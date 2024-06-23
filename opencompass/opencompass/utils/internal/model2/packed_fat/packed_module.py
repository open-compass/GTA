"""
modified flash attention

"""

import torch
from einops import rearrange
from torch import nn

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
except ImportError:
    ColumnParallelLinear, RowParallelLinear = None, None


from dataclasses import dataclass, field
from typing import Optional, Tuple

import rotary_emb
import flash_attn
from flash_attn.layers.rotary import ApplyRotaryEmbQKV_ as LegacyApplyRotaryEmbQKV_
from flash_attn.modules.mha import (
    CrossAttention,
    FlashCrossAttention,
    FlashSelfAttention,
    SelfAttention,
    _update_kv_cache,
)
from flash_attn.ops.fused_dense import fused_dense_func

from torch import Tensor
import torch.nn.functional as F

# This is a hardcode, due to the BC-breaking of flash_attn v2.0
if flash_attn.__version__ == '2.0.0.post1':
    from flash_attn.flash_attn_interface import FlashAttnVarlenQKVPackedFunc as FlashAttnQKVPackedFunc
else:
    from flash_attn.flash_attn_interface import FlashAttnQKVPackedFunc


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
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, three, _, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), q1, q2, False)
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), k1, k2, False
        )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), dq1, dq2, True
        )
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), dk1, dk2, True
        )
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
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, indexes=0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)
        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[indexes], self._sin_cached[indexes])
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[indexes],
                self._sin_cached[indexes],
                self._cos_k_cached[indexes],
                self._sin_k_cached[indexes],
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
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )


class RoPEForMultiSeqWithSinusoid(nn.Module):
    """
    RoPEForMultiSeqWithSinusoid

    Args:
        head_dim (int): the dimention of each attention head
        max_len (int): the max lenght of sequence
    """

    def __init__(self, head_dim: int, max_len: int):
        super().__init__()
        self.head_dim = head_dim
        theta = torch.pow(1e4, -2 * (torch.linspace(0, head_dim - 1, head_dim) // 2) / head_dim)  # [head_dim/2]
        theta = torch.arange(max_len).reshape(-1, 1, 1) * theta.reshape(1, 1, -1)  # L x 1 x head_dim/2
        sin_pos, cos_pos = torch.sin(theta), torch.cos(theta)  # [L x 1 x dim/2], [L x 1 x dim/2]
        self.register_buffer("sin_pos", sin_pos)
        self.register_buffer("cos_pos", cos_pos)

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

        return q_prime, k_prime  # The q' and k' here are obtained normally through triangular encoding,
        # and can be directly multiplied by the matrix without considering the imaginary and real parts.


class OneDParallelMHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        bias (boolean): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                        output projection. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (boolean): Whether to use flash attention or not.If False, vanilla attention module will be used.
                                    False by default.
        checkpointing (boolean): Whether to use torch.utils.checkpointing to save VRAM or not. False by default.
        sequence_parallel (boolean): If True, we're doing Tensor Parallel with sequence parallelism. An all_gather_raw
                                    of x will be done before doing the matmul.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = False,
        checkpointing: bool = False,
        sequence_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.Wqkv = ColumnParallelLinear(
            embed_dim, 3 * embed_dim, process_group, bias=True, sequence_parallel=sequence_parallel, **factory_kwargs
        )  # according to https://spaces.ac.cn/archives/9577
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = RowParallelLinear(
            embed_dim, embed_dim, process_group, sequence_parallel=sequence_parallel, **factory_kwargs
        )

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params)

    def _forward(self, x, seqlen=None, inference_params=None):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        bsz, max_len, dim = x.shape
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, "(b s) (three h d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)
        if inference_params is None:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb.eval_forward(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv)
        else:
            if self.rotary_emb_dim > 0:
                if hasattr(inference_params, 'attention_mask') and inference_params.attention_mask is not None:
                    empties = inference_params.attention_mask[..., -1].sum(dim=-1)
                    moved_qkv = qkv.clone()
                    if inference_params.sequence_len_offset == 0:
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                moved_qkv[i][:-empties[i]] = qkv[i][empties[i]:]
                        moved_qkv = self.rotary_emb.eval_forward(moved_qkv,
                                                                 seqlen_offset=inference_params.sequence_len_offset)
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                qkv[i][empties[i]:] = moved_qkv[i][:-empties[i]]
                            else:
                                qkv[i] = moved_qkv[i]
                    else:
                        qkv = qkv.squeeze(1)
                        qkv = self.rotary_emb.forward(qkv,
                                                      inference_params.sequence_len_offset * torch.ones(qkv.size(0),
                                                                                                        dtype=torch.int,
                                                                                                        device=qkv.device) - empties)
                        qkv = qkv[:, None, ...]
                else:
                    qkv = self.rotary_emb.eval_forward(qkv, seqlen_offset=inference_params.sequence_len_offset)
            q = qkv[:, :, 0]
            kv = torch.stack([qkv[:, :, 1], qkv[:, :, 2]], dim=2)
            assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
            if hasattr(inference_params, 'window_size') and inference_params.window_size is not None:
                if inference_params.window_size <= inference_params.sequence_len_offset:
                    assert kv.size(1) == 1, 'update kv lenth more than 1'
                    inference_params.key_value_memory_dict[self.layer_idx][:,inference_params.keep_first:inference_params.window_size-1, ...] = \
                    inference_params.key_value_memory_dict[self.layer_idx][:,-(inference_params.window_size-1-inference_params.keep_first):, ...].clone()
                    inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
                    inference_params.sequence_len_offset = inference_params.window_size - 1

                    kv = _update_kv_cache(kv, inference_params, self.layer_idx)

                    inference_params.sequence_len_offset = inference_params.real_sequence_len_offset
                else:
                    kv = _update_kv_cache(kv, inference_params, self.layer_idx)
            else:
                kv = _update_kv_cache(kv, inference_params, self.layer_idx)
            
            causal = None if inference_params.sequence_len_offset == 0 else False
            if hasattr(inference_params, 'attention_mask') and inference_params.attention_mask is not None:
                if inference_params.sequence_len_offset == 0:  # 第一次进入，是个方阵
                    attn_mask = inference_params.attention_mask[:,None,...]
                    attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                    attn_mask4flsh = ~ attn_mask[:,:,-1,:].view(bsz, -1)
                    cu_seqlens = torch.concat([torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device), attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32)], dim=0)
                    cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                    seqlen_qkv = attn_mask4flsh.shape[-1]
                    total_qkv = qkv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(-1, qkv.shape[-3], qkv.shape[-2], qkv.shape[-1])
                    output = FlashAttnQKVPackedFunc.apply(total_qkv, cu_seqlens, seqlen_qkv, 0.0, None, True, False)
                    context = torch.zeros_like(q)
                    context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)
                    
                else:
                    attn_mask = inference_params.attention_mask[:,-1,:].view(bsz, 1, 1, -1)
                    if hasattr(inference_params, 'window_size') and inference_params.window_size is not None:
                        if inference_params.window_size <= inference_params.sequence_len_offset:
                            attn_mask = torch.concat([attn_mask[..., :inference_params.keep_first], 
                                                    attn_mask[..., -(inference_params.window_size-inference_params.keep_first):]], 
                                                    dim=-1)
                    import math
                    k, v = torch.chunk(kv, 2, dim=2)
                    k = k.squeeze(2)
                    v = v.squeeze(2)
                    scores = torch.einsum('blhd,bnhd->bhln', q, k)/math.sqrt(q.size(-1))
                    scores = scores.masked_fill(attn_mask, -65000.0)
                    scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                    context = torch.einsum('bhmn,bnhd->bmhd', scores, v)    
            else:
                context = self.inner_cross_attn(q, kv, causal=causal)

        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")
        out = self.out_proj(context)
        return out

    def _packed_forward(self, x, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)  # total x hsz'
        qkv = rearrange(qkv, "t (three h d) -> t three h d", three=3, d=self.head_dim)  # total x 3 x n_head x d
        qkv = self.rotary_emb(qkv, kwargs.pop("indexes"))
        if inference_params is None:
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
        else:
            raise RuntimeError("Not support this right now")
        context = rearrange(context, "b h d -> b (h d)")  # recover the shape
        out = self.out_proj(context)
        return out


class ScaleColumnParallelLinear(nn.Linear):
    """
    ScaleColumnParallelLinear.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        sequence_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.weight_scale = weight_scale

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func(
            x, weight, self.bias, process_group=self.process_group, sequence_parallel=self.sequence_parallel
        )
