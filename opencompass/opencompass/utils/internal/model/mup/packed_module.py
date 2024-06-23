"""
由于需要修改 flash attention 中不少的组件，就放到当前这个地方了

"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
except ImportError:
    flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func = None, None

try:
    from flash_attn.ops.flash_attn_triton import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

try:
    from flash_attn.ops.fused_dense import FusedDense, ColumnParallelLinear, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    import ft_attention
except ImportError:
    ft_attention = None

from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention, SelfAttention, CrossAttention
from flash_attn.ops.fused_dense import fused_dense_func

from typing import Tuple
import math

import torch

from einops import rearrange, repeat

import rotary_emb
from flash_attn.layers.rotary import ApplyRotaryEmbQKV_ as LegacyApplyRotaryEmbQKV_

from dataclasses import dataclass, field
from typing import Optional, Union, Sequence, Callable
from flash_attn.modules.mha import _update_kv_cache
from torch import Tensor


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_sequence_len: int
    max_batch_size: int
    sequence_len_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    fused_ft_kernel: bool = False
    lengths_per_sample: Optional[Tensor] = None


class ApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        total, three, nheads, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, 's d -> s 1 d'),
                                rearrange(sin, 's d -> s 1 d'), q1, q2, False)
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(k1, k2, rearrange(cos_k, 's d -> s 1 d'),
                                rearrange(sin_k, 's d -> s 1 d'), k1, k2, False)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        # total, _, _, headdim = dqkv.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dq1, dq2, rearrange(cos, 's d -> s 1 d'),
                                rearrange(sin, 's d -> s 1 d'), dq1, dq2, True)
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dk1, dk2, rearrange(cos_k, 's d -> s 1 d'),
                                rearrange(sin_k, 's d -> s 1 d'), dk1, dk2, True)
        return dqkv, None, None, None, None


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
legacy_apply_rotary_embed_qkv = LegacyApplyRotaryEmbQKV_.apply


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """
        """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale_base = scale_base
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                 / (1.4 * dim) if scale_base > 0 else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)
        """
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1   # 如果是int说明走的是 eval_forward那个
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seqlen > self._seq_len_cached or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                          - seqlen // 2) / self.scale_base)
                scale = self.scale.to(device=power.device) ** rearrange(power, 's -> s 1')
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, indexes = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)
        if self.scale is None:
            return apply_rotary_emb_qkv_(
                qkv, self._cos_cached[indexes], self._sin_cached[indexes]
            )
        else:
            return apply_rotary_emb_qkv_(
                qkv, self._cos_cached[indexes], self._sin_cached[indexes],
                self._cos_k_cached[indexes], self._sin_k_cached[indexes], 
            )

    def eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return legacy_apply_rotary_embed_qkv(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:]
            )
        else:
            return legacy_apply_rotary_embed_qkv(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:], self._sin_k_cached[seqlen_offset:]
            )


class RoPEForMultiSeqWithSinusoid(nn.Module):

    def __init__(self, head_dim, max_len):
        super().__init__()
        self.head_dim = head_dim
        theta = torch.pow(1e4, -2 * (torch.linspace(0, head_dim-1, head_dim) // 2) / head_dim)   # [head_dim/2]
        theta = torch.arange(max_len).reshape(-1, 1, 1) * theta.reshape(1, 1, -1)  # L x 1 x head_dim/2
        sin_pos, cos_pos = torch.sin(theta), torch.cos(theta)  # [L x 1 x dim/2], [L x 1 x dim/2]
        self.register_buffer('sin_pos', sin_pos)
        self.register_buffer('cos_pos', cos_pos)

    def forward(self, q, k, indexes):
        """
        :param q: max_len * n_head * head_dim
        :param k: max_len * n_head * head_dim
        :param idx_list: max_len
        :return q_prime: q', q after rope, max_len * n_head * head_dim
        :return k_prime: k', k after rope, max_len * n_head * head_dim
        """
        sin_pos, cos_pos = self.sin_pos[indexes], self.cos_pos[indexes]

        # rotate_half_query [-q1,q0,-q3,q2......,-qd-1,qd-2]
        q_rotate_half = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        q_prime = q * cos_pos + q_rotate_half * sin_pos
        # rotate_half_key [-k1,k0,-k3,k2......,-kd-1,kd-2]
        k_rotate_half = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k)
        k_prime = k * cos_pos + k_rotate_half * sin_pos

        return q_prime, k_prime  # 这里的q'和k'是经过三角编码正常得到的，直接矩阵乘即可，无需考虑虚部实部


class OneDParallelMHA(nn.Module):
    """Multi-head self-attention and cross-attention
    支持输入为 total x hsz 形式的 causal attention
    """

    def __init__(self, embed_dim, num_heads, process_group, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, layer_idx=None, rotary_emb_dim=0,
                 rotary_emb_scale_base=0, use_flash_attn=False, checkpointing=False,
                 sequence_parallel=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base,
                                              device=device)
            # self.rotary_emb = RoPEForMultiSeqWithSinusoid(self.head_dim, max_len=2048)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError('fused_dense is not installed')
        self.Wqkv = ColumnParallelLinear(embed_dim, 3 * embed_dim, process_group, bias=bias,
                                         sequence_parallel=sequence_parallel, **factory_kwargs)
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                                     attention_dropout=dropout)
        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, process_group,
                                          sequence_parallel=sequence_parallel, **factory_kwargs)

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get('indexes', None) is not None:
            return self._packed_forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, '(b s) (three h d) -> b s three h d', s=seqlen, three=3,
                            d=self.head_dim)
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb.eval_forward(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv)
        else:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb.eval_forward(qkv, seqlen_offset=inference_params.sequence_len_offset)
            q = qkv[:, :, 0]
            assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
            kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
            # If we're processing the prompt, causal=None (use self.causal).
            # If we're decoding, then causal=False.
            causal = None if inference_params.sequence_len_offset == 0 else False
            context = self.inner_cross_attn(q, kv, causal=causal)
    
        if seqlen is None:
            context = rearrange(context, 'b s h d -> b s (h d)')
        else:
            context = rearrange(context, 'b s h d -> (b s) (h d)')
        out = self.out_proj(context)
        return out

    def _packed_forward(self, x, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)  # total x hsz'
        # if seqlen is None:
        #     qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, d=self.head_dim)
        # else:
        #     qkv = rearrange(qkv, '(b s) (three h d) -> b s three h d', s=seqlen, three=3,
        #                     d=self.head_dim)
        qkv = rearrange(qkv, 't (three h d) -> t three h d', three=3, d=self.head_dim)  # total x 3 x n_head x d
        qkv = self.rotary_emb(qkv, kwargs.pop('indexes'))
        if inference_params is None:
            # if self.rotary_emb_dim > 0:
            #     qkv = self.rotary_emb(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
        else:
            raise RuntimeError("Not support this right now")
        context = rearrange(context, 'b h d -> b (h d)')  # 回到之前的样子
        out = self.out_proj(context)
        return out


class ScaleColumnParallelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, process_group,
                 bias: bool = True, sequence_parallel=True, device=None, dtype=None, 
                 weight_scale=1) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f'out_features ({out_features}) must be divisible by '
                             f'world_size ({world_size})')
        super().__init__(in_features, out_features // world_size, bias=bias,
                         device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.weight_scale = weight_scale

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight*self.weight_scale + (1-self.weight_scale)*self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func(x, weight, self.bias, process_group=self.process_group,
                                sequence_parallel=self.sequence_parallel)


if __name__ == '__main__':
    
    pass