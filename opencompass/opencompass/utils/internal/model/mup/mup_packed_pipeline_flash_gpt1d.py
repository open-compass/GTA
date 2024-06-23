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

from einops import rearrange
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedMLP, ParallelFusedMLP
from flash_attn.ops.layer_norm import dropout_add_layer_norm
from .packed_module import OneDParallelMHA, ScaleColumnParallelLinear
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward

### mup
import mup
from torch.nn.modules.conv import _ConvNd
logger = get_dist_logger()


def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

### mup
class MuScaleColumnParallelLinear(ScaleColumnParallelLinear, mup.MuReadout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def width_mult(self):
        assert hasattr(self.weight, 'infshape'), (
            'Please call set_base_shapes(...). If using torch.nn.DataParallel, '
            'switch to distributed training with '
            'torch.nn.parallel.DistributedDataParallel instead'
        )
        return self.weight.infshape.width_mult()
                    
    def forward(self, x):
        return super().forward(x / self.width_mult())

class PackedFlashTransformerLayer1D(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 mlp_ratio: int = 4,
                 attn_drop_rate: float = 0,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False,
                 fused_dropout_add_ln=True,
                 checkpoint=True,
                 layer_idx=0,
                 residual_in_fp32=False,
                 device=None,
                 no_bias=False,
                 mup=False,):
        super().__init__()
        self.logger = get_dist_logger()
        self.prenorm = not apply_post_layer_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln

        head_dim = hidden_size//num_attention_heads
        ### mup: modify softmax_scale
        if mup:
            softmax_scale = 1 / head_dim
        else:
            softmax_scale = 1/math.sqrt(head_dim)
        self.mixer = OneDParallelMHA(embed_dim=hidden_size, num_heads=num_attention_heads, process_group=gpc.get_group(ParallelMode.PARALLEL_1D), 
                                 bias=not no_bias, dropout=attn_drop_rate, softmax_scale=softmax_scale, causal=True, layer_idx=layer_idx, 
                                 rotary_emb_dim=head_dim, rotary_emb_scale_base=0, use_flash_attn=True, checkpointing=False,
                                 sequence_parallel=False, device=device, dtype=dtype)
        
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.mlp = ParallelFusedMLP(hidden_size, int(hidden_size*mlp_ratio), out_features=hidden_size, activation='gelu_approx',
                                    process_group = gpc.get_group(ParallelMode.PARALLEL_1D), bias1=not no_bias, bias2=not no_bias,
                                    sequence_parallel=False, checkpoint_lvl=0, heuristic='auto',
                                    device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        if no_bias:
            self.norm2.bias = None
            self.norm1.bias = None
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

    def forward(self, hidden_states, residual=None, cu_seqlens=None, indexes=None, inference_params=None):
        if self.checkpoint and self.training:
            return checkpoint(self._forward, False, hidden_states, residual, cu_seqlens, indexes, inference_params)
        else:
            return self._forward(hidden_states, residual, cu_seqlens, indexes, inference_params)

    def _forward(self, hidden_states=None, residual=None, cu_seqlens=None, indexes=None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor，长度比hidden_states 多一个
            indexes: 长度和hidden states一致，表示的是当前位置的position
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
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': (cu_seqlens[1:]-cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None, 
                            'indexes': indexes, 'inference_params': inference_params}
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
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': (cu_seqlens[1:]-cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None, 
                            'indexes': indexes, 'inference_params': inference_params}
            mixer_out = self.mixer(hidden_states, **mixer_kwargs)
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


class PackedFlashFusedPipelineGPT1D(nn.Module):
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
                 device=None,
                 no_bias=False,  # 来自Palm
                 initializer_range=1.0,
                 emb_scale=1.0,
                 mup=False,
                 ):
        super().__init__()
        embedding = None
        head = None
        
        self.initializer_range = initializer_range
        self.num_layers = num_layers
        self.emb_scale = emb_scale
        
        embed_cls = ParallelGPT2Embeddings
        if mup:
            head_cls = MuScaleColumnParallelLinear
        else:
            head_cls = ScaleColumnParallelLinear
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
            PackedFlashTransformerLayer1D(hidden_size = hidden_size,
                 num_attention_heads = num_attention_heads,
                 mlp_ratio = mlp_ratio,
                 attn_drop_rate=attn_drop_rate,
                 drop_rate = drop_rate,
                 dtype = dtype,
                 layer_norm_epsilon = layer_norm_epsilon,
                 apply_post_layer_norm = apply_post_layer_norm, 
                 fused_dropout_add_ln = False,
                 checkpoint = checkpoint,
                 layer_idx = _+start_layer_idx,  # 这个参数是用来在生成的时候做cache使用的
                 residual_in_fp32 = False,
                 device=device,
                 no_bias=no_bias,
                 mup=mup)
            for _ in range(num_layers)
        ])
        if last:
            if not apply_post_layer_norm:
                self.norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
                if no_bias:
                    self.norm.bias = None
            self.head = head_cls(in_features=hidden_size, out_features=vocab_size, process_group=gpc.get_group(ParallelMode.TENSOR),
                 bias = False, sequence_parallel=False, device=device, dtype=dtype, weight_scale=embed_grad_scale)
            for name, param in self.head.named_parameters():
                with seed(ParallelMode.TENSOR):
                    fan_in, fan_out = param.shape
                    init.normal_(std=0.006)(param, fan_in=fan_in, fan_out=fan_out)
        self.parallel_output = parallel_output
        self.apply_post_layer_norm = apply_post_layer_norm

    ### mup+GPT3 settings
    # http://arxiv.org/abs/2203.03466 Table 3
    # http://arxiv.org/abs/2005.14165 
    def init_weight_mup_GPT3(self, module, readout_zero_init=True, query_zero_init=True):
        if isinstance(module, MuScaleColumnParallelLinear) and readout_zero_init:
            # Output weights zero initilization.
            module.weight.data.zero_()
        elif isinstance(module, (nn.Linear, _ConvNd)):
            # Linear weights 
            mup.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            # bias weight
            if module.bias is not None:
                module.bias.data.zero_()
        # Default to GPT3
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # mup query layers zero init
        if isinstance(module, OneDParallelMHA) and query_zero_init:
            fanout, _ = module.Wqkv.weight.shape
            assert fanout % 3 == 0
            module.Wqkv.weight.data[:, :fanout//3] = 0
            
        # default to GPT3
        depth_std = self.initializer_range / math.sqrt(2 * self.num_layers)
        for name, p in module.named_parameters():
            ## TODO add DeepNorm, reference DeepNet & GLM130B
            if (("out_proj" in name) or ('fc2' in name)) and ('weight' in name):
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                mup.init.normal_(p, mean=0.0, std=depth_std)
                
    def forward(self, hidden_states=None, residual=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: 为1的地方需要attention
        if hasattr(self, 'embedding'):
            hidden_states = self.embedding(input_ids) * self.emb_scale
            if self.embed_grad_scale != 1:
                hidden_states = self.embed_grad_scale*hidden_states + (1-self.embed_grad_scale)*hidden_states.detach()
        if isinstance(cu_seqlens, list):
            assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)
            hidden_states = hidden_states.squeeze(0)  # 如果传入了 cu_seqlens ，说明是 packed 的状态，需要将 batch 为1的那一维直接 squeeze 掉
        if indexes is not None:
            assert len(indexes) == 1
            indexes = indexes[0]  # indexes 是用来指示在 packed 输入中各个 token 实际的 position id 。
        if self.apply_post_layer_norm:
            for idx, block in enumerate(self.blocks):
                # print(f'rank:{gpc.get_global_rank()}, idx:{idx}, hsz:{hidden_states[:, :5, :5].tolist()}')
                hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, indexes=indexes, inference_params=inference_params)
        else:
            for idx, block in enumerate(self.blocks):
                hidden_states, residual = block(hidden_states, residual, cu_seqlens=cu_seqlens, indexes=indexes, inference_params=inference_params)
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, hidden:{hidden_states[0, 0, :10].tolist()}')
        if hasattr(self, 'norm'):
            hidden_states = self.norm(hidden_states)
        if hasattr(self, 'head'):
            hidden_states = self.head(hidden_states)
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, head:{hidden_states[:, :10].argmax(dim=-1).tolist()}')
        # exit()
        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.PARALLEL_1D, dim=-1)
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
    if gpc.is_initialized(ParallelMode.GLOBAL):
        logger.info(f"The layer sharding is {all_parts}.", ranks=[0])
    parts = all_parts[pipeline_rank]

    models = []
    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = (end == num_layers and len(all_parts[-1])!=0)  # 如果最后一层没有内容的话，把最后的layer分给他
        kwargs['device'] = device
        kwargs['start_layer_idx'] = start
        chunk = PackedFlashFusedPipelineGPT1D(**_filter_kwargs(PackedFlashFusedPipelineGPT1D.__init__, kwargs)).to(device)

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


def Packed_Flash_GPT2_exlarge_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False, 
                             num_layers=48, hidden_size=2048, vocab_size=50304, embed_grad_scale=0.1,
                             parallel_output=True, num_attention_heads=32, mlp_ratio=4.0, apply_post_layer_norm=False, 
                             no_bias=False, mup=False, initializer_range=1.0, emb_scale=1.0):
    cfg = dict(hidden_size=hidden_size, num_attention_heads=num_attention_heads, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden, vocab_size=vocab_size, 
               embed_grad_scale=embed_grad_scale, parallel_output=parallel_output, mlp_ratio=mlp_ratio, apply_post_layer_norm=apply_post_layer_norm,
               no_bias=no_bias, mup=mup, initializer_range=initializer_range, emb_scale=emb_scale)
    return _build_generic_gpt_pipeline_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)