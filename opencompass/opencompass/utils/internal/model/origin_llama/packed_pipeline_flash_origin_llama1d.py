"""
在flash attention架构下复现 fair 的 llama
原始代码 https://github.com/facebookresearch/llama.git
"""
import torch
from colossalai.logging import get_dist_logger
from torch import nn
import torch
from colossalai.core import global_context as gpc

from colossalai.context.parallel_mode import ParallelMode
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
import inspect

from ..pipeline_utils import partition_uniform_with_embed, partition_uniform_with_embed2

from colossalai.nn import init 
import math
from colossalai.context import ParallelMode, seed
from colossalai.utils.activation_checkpoint import checkpoint


from flash_attn.modules.embedding import ParallelGPT2Embeddings

from einops import rearrange
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw
from flash_attn.ops.layer_norm import dropout_add_layer_norm
from ..packed_fat.packed_module import OneDParallelMHA, ScaleColumnParallelLinear
from .origin_llama_module import FeedForward
try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as RMSNorm
except:  # 防止没有安装直接就没办法 import 了
    pass
from ..llama.llama_module import RMSNorm  # 感觉应该是由于 apex 这个造成的不稳定, 因此norm就不要使用优化的了
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward
from .origin_llama_module import OneDOriginParallelMHA
logger = get_dist_logger()
from torch import Tensor
import torch.nn.functional as F


class Embedding1D(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim//gpc.tensor_parallel_size

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), dtype=dtype))

    def forward(self, input_: Tensor) -> Tensor:

        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)

        return output



def _filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def xavier_normal_tensor_parallel(tensor, partition_dim, gain=1, tp_degree=1):
    """对分布式的tensor进行初始化

    :param tensor: 需要初始化的tensor
    :param partition_dim: tensor是在哪个维度进行split的
    :param gain: _description_, defaults to 1
    :param tp_degree: _description_, defaults to 1
    :raises RuntimeError: _description_
    :return: _description_
    """
    assert len(tensor.shape) == 2
    fan_in, fan_out = tensor.shape
    if partition_dim == 0:
        fan_in *= tp_degree
    else:
        fan_out *= tp_degree

    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return torch.nn.init._no_grad_normal_(tensor, 0, std)


class PackedFlashOriginLLAMALayer1D(nn.Module):
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
                 deepnorm=False,
                 total_layers=-1,  # 如果不为1则是因为使用了deepnorm
                 norm_type='rmsnorm'
                 ):
        super().__init__()
        self.logger = get_dist_logger()
        self.prenorm = not apply_post_layer_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.total_layers = total_layers

        if deepnorm:
            assert total_layers!=-1, "When using deepnorm must pass in the total numer of layers"
            self.deepnorm_alpha = (2 * total_layers) ** 0.5  # deepnorm论文给的是0.25，但是GLM-130B代码给的是在adam优化器中使用0.5, 因为我们用adam这里就给fix了，TODO 可以做成可以调整的
            self.deepnorm_beta= (2 * total_layers) ** -0.5  # from LargeScale.megatron.model.utils:get_deepnorm_coefficients  根据是来源于https://kexue.fm/archives/8978
        else:
            self.deepnorm_alpha = 1.0
            self.deepnorm_beta = 1.0

        head_dim = hidden_size//num_attention_heads
        self.attention = OneDOriginParallelMHA(embed_dim=hidden_size, num_heads=num_attention_heads, process_group=gpc.get_group(ParallelMode.PARALLEL_1D), 
                                 bias=not no_bias, dropout=attn_drop_rate, softmax_scale=1/math.sqrt(head_dim), causal=True, layer_idx=layer_idx, 
                                 rotary_emb_dim=head_dim, rotary_emb_scale_base=0, use_flash_attn=True, checkpointing=False,
                                 sequence_parallel=False, device=device, dtype=dtype)
        
        self.dropout1 = nn.Dropout(drop_rate)
        if norm_type == 'rmsnorm':
            self.attention_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.feed_forward = FeedForward(hidden_size, int(hidden_size*mlp_ratio),        
                                    out_features=hidden_size, activation='gelu_approx',
                                    process_group = gpc.get_group(ParallelMode.PARALLEL_1D), bias=not no_bias,
                                    sequence_parallel=False, checkpoint_lvl=0, heuristic='auto',
                                    device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(drop_rate)
        if norm_type == 'rmsnorm':
            self.ffn_norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_add_ln is not installed'
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)
        self.residual_in_fp32 = residual_in_fp32  # 仅在 prenorm 下有意义
        self.return_residual = False
        self.deepnorm = deepnorm  # 在 prenorm 下使用 deepnorm 就仅等同于使用了deepnorm初始化，不会影响到训练的
        if deepnorm:
            self.deepnorm_reset_parameters()
        else:
            self.reset_parameters()
        self.checkpoint = checkpoint
        self.layer_idx = layer_idx

    def deepnorm_reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    # init.normal_(std=0.006)(param.data)
                    param.data.zero_()
                elif 'Wqkv' in name:
                    wq, wk, wv = param.data.chunk(3, dim=0)
                    xavier_normal_tensor_parallel(wq, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0)
                    xavier_normal_tensor_parallel(wk, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0)
                    xavier_normal_tensor_parallel(wv, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta)
                else:  # dense
                    xavier_normal_tensor_parallel(param.data, partition_dim=1, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and 'bias' in name:
                    param.data.zero_()
                else:
                    if 'w1' in name or 'w2' in name:
                        xavier_normal_tensor_parallel(param.data, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta)
                    else:  # w3
                        xavier_normal_tensor_parallel(param.data, partition_dim=1, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta)

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    # init.normal_(std=0.006)(param.data)
                    param.data.zero_()
                elif 'Wqkv' in name:
                    init.normal_(std=0.006)(param.data)
                else:
                    init.normal_(std=0.0015)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if param.ndim == 1 and 'bias' in name:
                    param.data.zero_()
                else:
                    if 'w1' in name or 'w2' in name:
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
                hidden_states = self.attention_norm(residual.to(dtype=self.attention_norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                # self.logger.info(f'rank: {gpc.get_global_rank()}, residual:{residual[0, 0, :10].tolist()}')
            else:
                rowscale1 = None
                hidden_states, residual = dropout_add_layer_norm(
                    hidden_states, residual, self.attention_norm.weight, self.attention_norm.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': (cu_seqlens[1:]-cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None, 
                            'indexes': indexes, 'inference_params': inference_params}
            hidden_states = self.attention(hidden_states, **mixer_kwargs)
            # self.logger.info(f'rank: {gpc.get_global_rank()}, after attention:{hidden_states[0, 0, :10].tolist()}')
            # exit()
            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.dropout2(hidden_states)
                    residual = (dropped + residual) if residual is not None else dropped
                    # self.logger.info(f'rank: {gpc.get_global_rank()}, before norm:h:{hidden_states[0, 0, :10].tolist()}, d:{dropped[0, 0, :10].tolist()}, r:{residual[0, 0, :10].tolist()}')
                    hidden_states = self.ffn_norm(residual.to(dtype=self.ffn_norm.weight.dtype))
                    # self.logger.info(f'rank: {gpc.get_global_rank()}, after norm:{hidden_states[0, 0, :10].tolist()}')
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                # self.logger.info(f'rank: {gpc.get_global_rank()}, before mlp:{hidden_states[0, 0, :10].tolist()}')
                hidden_states = self.feed_forward(hidden_states)
                # self.logger.info(f'rank: {gpc.get_global_rank()}, after mlp:{hidden_states[0, 0, :10].tolist()}')
            return hidden_states + residual
        else:
            assert residual is None
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': (cu_seqlens[1:]-cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None, 
                            'indexes': indexes, 'inference_params': inference_params}
            mixer_out = self.attention(hidden_states, **mixer_kwargs)
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.attention_norm(self.dropout1(mixer_out)
                                            + hidden_states * self.deepnorm_alpha).to(dtype=self.norm1.weight.dtype)
            else:
                rowscale1 = None
                hidden_states = dropout_add_layer_norm(
                    mixer_out, hidden_states * self.deepnorm_alpha, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False, residual_in_fp32=False
                )
            if not isinstance(self.feed_forward, nn.Identity):
                mlp_out = self.feed_forward(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.ffn_norm((self.dropout2(mlp_out))
                                                + hidden_states * self.deepnorm_alpha).to(dtype=self.norm2.weight.dtype)
                else:
                    rowscale2 = None
                    hidden_states = dropout_add_layer_norm(
                        mlp_out, hidden_states * self.deepnorm_alpha, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False, residual_in_fp32=False
                    )
            return hidden_states



class PackedFlashPipelineOriginLLAMA1D(nn.Module):
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
                 deepnorm=False,
                 total_layers=-1,  # 为deepnorm计算需要
                 residual_in_fp32=False,
                 norm_type='rmsnorm'
        ):
        super().__init__()
        embed_cls = Embedding1D
        head_cls = ScaleColumnParallelLinear
        if embed_split_hidden:
            raise RuntimeError("Currently not support split in hidden dimension")
        if first:
            # self.tok_embeddings = embed_cls(embed_dim=hidden_size, vocab_size=vocab_size, max_position_embeddings=-1, process_group=gpc.get_group(ParallelMode.TENSOR),
            #      padding_idx=None, sequence_parallel=False, device=device, dtype=dtype)
            self.tok_embeddings = embed_cls(num_embeddings=vocab_size, embedding_dim=hidden_size)
            for name, param in self.tok_embeddings.named_parameters():
                init.normal_(std=0.0052)(param)
        self.embed_grad_scale = embed_grad_scale
        self.layers = nn.ModuleList([
            PackedFlashOriginLLAMALayer1D(hidden_size = hidden_size,
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
                 residual_in_fp32 = residual_in_fp32,
                 device=device,
                 no_bias=no_bias,
                 deepnorm=deepnorm,
                 total_layers=total_layers,  # 为deepnorm计算需要
                 norm_type=norm_type
                 )
            for _ in range(num_layers)
        ])
        if last:
            if not apply_post_layer_norm:
                if norm_type == 'rmsnorm':
                    self.norm = RMSNorm(hidden_size, eps=layer_norm_epsilon)
                else:
                    self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.output = head_cls(in_features=hidden_size, out_features=vocab_size, process_group=gpc.get_group(ParallelMode.TENSOR),
                 bias = False, sequence_parallel=False, device=device, dtype=dtype, weight_scale=embed_grad_scale)
            for name, param in self.output.named_parameters():
                init.normal_(std=0.0052)(param)
        self.parallel_output = parallel_output
        self.apply_post_layer_norm = apply_post_layer_norm

    def forward(self, hidden_states=None, residual=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: 为1的地方需要attention
        if hasattr(self, 'tok_embeddings'):
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = self.embed_grad_scale*hidden_states + (1-self.embed_grad_scale)*hidden_states.detach()
            # if torch.distributed.get_rank() == 0:
            #     print(f"output {hidden_states[:, :, :3].tolist()}")

        if isinstance(cu_seqlens, list):
            assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)
            hidden_states = hidden_states.squeeze(0)  # 如果传入了 cu_seqlens ，说明是 packed 的状态，需要将 batch 为1的那一维直接 squeeze 掉
        if indexes is not None:
            assert len(indexes) == 1
            indexes = indexes[0]  # indexes 是用来指示在 packed 输入中各个 token 实际的 position id 。
        if self.apply_post_layer_norm:
            for idx, block in enumerate(self.layers):
                # print(f'rank:{gpc.get_global_rank()}, idx:{idx}, hsz:{hidden_states[:, :5, :5].tolist()}')
                hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, indexes=indexes, inference_params=inference_params)
        else:
            for idx, block in enumerate(self.layers):
                hidden_states = block(hidden_states, residual=None, cu_seqlens=cu_seqlens, indexes=indexes, inference_params=inference_params)
            # exit()
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, hidden:{hidden_states[0, 0, :10].tolist()}')
        if hasattr(self, 'norm'):
            hidden_states = self.norm(hidden_states)
        if hasattr(self, 'output'):
            hidden_states = self.output(hidden_states)
            # logger.info(f'rank: {gpc.get_global_rank()}, idx:{idx}, head:{hidden_states[:, :10].argmax(dim=-1).tolist()}')
        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.PARALLEL_1D, dim=-1)
        # if torch.distributed.get_rank() == 0:
        #     print(f"output {idx}, {hidden_states[:, :, :3].tolist()}")
        # exit()
        return hidden_states


def _build_generic_origin_llama_pipeline_1d(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
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
    kwargs['total_layers'] = num_layers
    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = (end == num_layers and len(all_parts[-1])!=0)  # 如果最后一层没有内容的话，把最后的layer分给他
        kwargs['device'] = device
        kwargs['start_layer_idx'] = start
        chunk = PackedFlashPipelineOriginLLAMA1D(**_filter_kwargs(PackedFlashPipelineOriginLLAMA1D.__init__, kwargs)).to(device)

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


def Packed_Flash_Origin_LLAMA_exlarge_pipeline_1D(num_chunks=1, checkpoint=False, dtype=torch.float, embed_split_hidden=False, 
                             num_layers=48, hidden_size=2048, vocab_size=50304, embed_grad_scale=0.1,
                             parallel_output=True, num_attention_heads=32, mlp_ratio=4.0, apply_post_layer_norm=False, 
                             no_bias=False, deepnorm=False, residual_in_fp32=False,
                             norm_type='rmsnorm', drop_rate=0, attn_drop_rate=0, model_type='llama'):
    assert model_type == 'llama', "Only support llama for this initilization"
    # residual_in_fp32 暂时无法使用，因为这个参数需要在 pipeline 之间传递的数据dtype不一致，需要对 colossalai 有比较大的修改
    cfg = dict(hidden_size=hidden_size, num_attention_heads=num_attention_heads, checkpoint=checkpoint,
               dtype=dtype, embed_split_hidden=embed_split_hidden, vocab_size=vocab_size, 
               embed_grad_scale=embed_grad_scale, parallel_output=parallel_output, mlp_ratio=mlp_ratio, apply_post_layer_norm=apply_post_layer_norm,
               no_bias=no_bias, deepnorm=deepnorm, residual_in_fp32=residual_in_fp32,
               norm_type=norm_type, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
    return _build_generic_origin_llama_pipeline_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)




