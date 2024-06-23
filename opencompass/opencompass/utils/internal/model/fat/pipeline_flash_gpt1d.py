import torch
from colossalai.logging import get_dist_logger
from torch import nn
import torch
from colossalai.core import global_context as gpc

from colossalai import kernel
from colossalai.context.parallel_mode import ParallelMode
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
import inspect

from ..pipeline_utils import partition_uniform_with_embed

from colossalai.nn import init 
import math
from colossalai.context import ParallelMode, seed
from colossalai.utils.activation_checkpoint import checkpoint


from flash_attn.modules.embedding import ParallelGPT2Embeddings
from flash_attn.ops.fused_dense import ColumnParallelLinear

from einops import rearrange
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedMLP, ParallelFusedMLP
from flash_attn.ops.layer_norm import dropout_add_layer_norm
logger = get_dist_logger()


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class FlashTransformerLayer1D(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False,
                 fused_dropout_add_ln=True,
                 checkpoint=True,
                 layer_idx=0,
                 residual_in_fp32=False,
                 device=None):
        super().__init__()
        self.logger = get_dist_logger()
        self.prenorm = not apply_post_layer_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln

        head_dim = hidden_size//num_attention_heads
        self.mixer = ParallelMHA(embed_dim=hidden_size, num_heads=num_attention_heads, process_group=gpc.get_group(ParallelMode.PARALLEL_1D), 
                                 bias=True, dropout=drop_rate, softmax_scale=1/math.sqrt(head_dim), causal=True, layer_idx=layer_idx, 
                                 rotary_emb_dim=head_dim, rotary_emb_scale_base=0, use_flash_attn=True, checkpointing=False,
                                 sequence_parallel=False, device=device, dtype=dtype)
        
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.mlp = ParallelFusedMLP(hidden_size, int(hidden_size*mlp_ratio), out_features=hidden_size, activation='gelu_approx',
                                    process_group = gpc.get_group(ParallelMode.PARALLEL_1D), bias1=True, bias2=True,
                                    sequence_parallel=False, checkpoint_lvl=0, heuristic='auto',
                                    device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_add_ln is not installed'
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)
        self.residual_in_fp32 = residual_in_fp32
        self.return_residual = False  
        self.reset_parameters()
        self.checkpoint = checkpoint

    def reset_parameters(self):
        with torch.no_grad():
            with seed(ParallelMode.TENSOR):
                for name, param in self.mixer.named_parameters():
                    if param.ndim == 1:
                        # init.normal_(std=0.006)(param.data)
                        param.data.zero_()
                    elif 'Wqkv' in name:
                        init.normal_(std=0.006)(param.data)
                    else:
                        init.normal_(std=0.0015)(param.data)

                for name, param in self.mlp.named_parameters():
                    if param.ndim == 1:
                        param.data.zero_()
                    else:
                        if 'fc1' in name:
                            init.normal_(std=0.006)(param.data)
                        else:
                            init.normal_(std=0.0015)(param.data)

    def forward(self, hidden_states, residual=None):
        if self.checkpoint and self.training:
            return checkpoint(self._forward, False, hidden_states, residual)
        else:
            return self._forward(hidden_states, residual)

    def _forward(self, hidden_states=None, residual=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.dropout1(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                # self.logger.info(f'rank: {gpc.get_global_rank()}, residual:{residual[0, 0, :10].tolist()}')
            else:
                rowscale1 = None
                hidden_states, residual = dropout_add_layer_norm(
                    hidden_states, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            # self.logger.info(f'rank: {gpc.get_global_rank()}, after attention:{hidden_states[0, 0, :10].tolist()}')
            # exit()
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.dropout2(hidden_states)
                    residual = (dropped + residual) if residual is not None else dropped
                    # self.logger.info(f'rank: {gpc.get_global_rank()}, before norm:h:{hidden_states[0, 0, :10].tolist()}, d:{dropped[0, 0, :10].tolist()}, r:{residual[0, 0, :10].tolist()}')
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    # self.logger.info(f'rank: {gpc.get_global_rank()}, after norm:{hidden_states[0, 0, :10].tolist()}')
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    rowscale2 = None
                    hidden_states, residual = dropout_add_layer_norm(
                        hidden_states, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=True, residual_in_fp32=self.residual_in_fp32
                    )
                # self.logger.info(f'rank: {gpc.get_global_rank()}, before mlp:{hidden_states[0, 0, :10].tolist()}')
                hidden_states = self.mlp(hidden_states)
                # self.logger.info(f'rank: {gpc.get_global_rank()}, after mlp:{hidden_states[0, 0, :10].tolist()}')
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1(self.dropout1(mixer_out)
                                            + hidden_states).to(dtype=self.norm1.weight.dtype)
            else:
                rowscale1 = None
                hidden_states = dropout_add_layer_norm(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False, residual_in_fp32=self.residual_in_fp32
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.dropout2(mlp_out))
                                                + hidden_states).to(dtype=self.norm2.weight.dtype)
                else:
                    rowscale2 = None
                    hidden_states = dropout_add_layer_norm(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False, residual_in_fp32=self.residual_in_fp32
                    )
            return hidden_states


class FlashFusedPipelineGPT1D(nn.Module):
    def __init__(self,
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 vocab_size: int = 50304,
                 embed_drop_rate: float = 0.,
                 act_func: str = 'gelu',
                 mlp_ratio: int = 4.0,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 checkpoint: bool = False,
                 max_position_embeddings: int = -1,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False,
                 first: bool = False,
                 last: bool = False,
                 embed_split_hidden=False,
                 embed_grad_scale=0.1,
                 parallel_output=True,
                 start_layer_idx=0,
                 device=None):
        super().__init__()
        embedding = None
        norm = None
        head = None
        embed_cls = ParallelGPT2Embeddings
        head_cls = ColumnParallelLinear
        if embed_split_hidden:
            raise RuntimeError("Currently not support split in hidden dimension")
        if first:
            self.embedding = embed_cls(embed_dim=hidden_size, vocab_size=vocab_size, max_position_embeddings=-1, process_group=gpc.get_group(ParallelMode.TENSOR),
                 padding_idx=None, sequence_parallel=False, device=device, dtype=dtype)
            for name, param in self.embedding.named_parameters():
                with seed(ParallelMode.TENSOR):
                    fan_in, fan_out = param.shape
                    init.normal_(std=0.006)(param, fan_in=fan_in, fan_out=fan_out)
        self.embed_grad_scale = embed_grad_scale
        self.blocks = nn.ModuleList([
            FlashTransformerLayer1D(hidden_size = hidden_size,
                 num_attention_heads = num_attention_heads,
                 mlp_ratio = mlp_ratio,
                 drop_rate = drop_rate,
                 dtype = dtype,
                 layer_norm_epsilon = layer_norm_epsilon,
                 apply_post_layer_norm = apply_post_layer_norm, 
                 fused_dropout_add_ln = False,
                 checkpoint = checkpoint,
                 layer_idx = _+start_layer_idx,
                 residual_in_fp32 = False,
                 device=device)
            for _ in range(num_layers)
        ])
        if last:
            self.norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.head = head_cls(in_features=hidden_size, out_features=vocab_size, process_group=gpc.get_group(ParallelMode.TENSOR),
                 bias = False, sequence_parallel=False, device=device, dtype=dtype)
            for name, param in self.head.named_parameters():
                with seed(ParallelMode.TENSOR):
                    fan_in, fan_out = param.shape
                    init.normal_(std=0.006)(param, fan_in=fan_in, fan_out=fan_out)
        self.parallel_output = parallel_output
        self.apply_post_layer_norm = apply_post_layer_norm

    def forward(self, hidden_states=None, residual=None, input_ids=None, attention_mask=None):
        # attention_mask: 为1的地方需要attention
        if hasattr(self, 'embedding'):
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = self.embed_grad_scale*hidden_states + (1-self.embed_grad_scale)*hidden_states.detach()
        # logger.info(f'rank: {gpc.get_global_rank()}, embed:{hidden_states[0, 0, :10].tolist()}')
        if self.apply_post_layer_norm:
            for idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states)
        else:
            for idx, block in enumerate(self.blocks):
                hidden_states, residual = block(hidden_states, residual)
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, hidden:{hidden_states[0, 0, :10].tolist()}')

        if hasattr(self, 'norm'):
            hidden_states = self.head(self.norm(hidden_states))
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, head:{hidden_states[0, 0, :10].tolist()}')


        if self.apply_post_layer_norm:
            return hidden_states
        else:
            return hidden_states, residual


def _build_generic_gpt_pipeline_1d(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    if pipeline_size > 1:
        wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])  # 同步了一下参数，同时设置了pipeline_shared_module_pg，这样会在Engine初始化的时候设置PipelineSharedModuleGradientHandler
    else:
        wrapper = None

    all_parts = partition_uniform_with_embed(num_layers, pipeline_size, num_chunks)
    logger.info(f"The layer sharding is {all_parts}.", ranks=[0])
    parts = all_parts[pipeline_rank]

    models = []
    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = (end == num_layers and len(all_parts[-1])!=0)  # 如果最后一层没有内容的话，把最后的layer分给他
        kwargs['device'] = device
        kwargs['start_layer_idx'] = start
        chunk = FlashFusedPipelineGPT1D(**_filter_kwargs(FlashFusedPipelineGPT1D.__init__, kwargs)).to(device)

        if wrapper is not None:
            if start == 0:
                wrapper.register_module(chunk.embedding)
            elif end == num_layers:
                wrapper.register_module(chunk.head)
        models.append(chunk)
    torch.distributed.barrier()

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


def Flash_GPT2_exlarge_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False, 
                             num_layers=48, hidden_size=2048, vocab_size=50304, embed_grad_scale=0.1,
                             parallel_output=True, num_attention_heads=32, mlp_ratio=4.0, apply_post_layer_norm=False):
    cfg = dict(hidden_size=hidden_size, num_attention_heads=num_attention_heads, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden, vocab_size=vocab_size, 
               embed_grad_scale=embed_grad_scale, parallel_output=parallel_output, mlp_ratio=mlp_ratio, apply_post_layer_norm=apply_post_layer_norm)
    return _build_generic_gpt_pipeline_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)