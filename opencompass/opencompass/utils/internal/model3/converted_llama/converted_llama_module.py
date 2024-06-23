from typing import Optional, Union  # For comments

import torch
import torch.nn.functional as F
from colossalai.constants import IS_TENSOR_PARALLEL
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from einops import rearrange
from flash_attn.modules.mha import (
    CrossAttention,
    FlashCrossAttention,
    FlashSelfAttention,
    SelfAttention,
    _update_kv_cache,
)
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
from torch import nn

from ..packed_fat.packed_module import RotaryEmbedding


class OneDConvertedParallelMHA(nn.Module):
    """
    Multi-head self-attention and cross-attention
    The shape of causal attention matrix is total * hiddensize.

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
        use_flash_attn (boolean): Whether to use flash attention or not. If False, vanilla attention module
                                    will be used. False by default.
        checkpointing (boolean): Whether to use torch.utils.checkpointing to save VRAM or not. False by default.
        sequence_parallel (boolean): If True, we're doing Tensor Parallel with sequence parallelism. An all_gather_raw
                                    of x will be done before doing the matmul.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.

    Raises:
        ImportError: An ImportError is raised if ColumnParallelLinear or RowParallelLinear is None
        RuntimeError: An RuntimeError is raised if the inference_params is not None when calling _packed_forward

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
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        rot_embed_HF_impl: Optional[bool] = False,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpointing = checkpointing
        self.rot_embed_HF_impl = rot_embed_HF_impl

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base, device=device)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")

        self.Wqkv = ColumnParallelLinear(
            embed_dim, 3 * embed_dim, process_group, bias=bias, sequence_parallel=sequence_parallel, **factory_kwargs
        )

        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.wo = RowParallelLinear(
            embed_dim, embed_dim, process_group, sequence_parallel=sequence_parallel, bias=bias, **factory_kwargs
        )

        # need to assign tp attribute so that colossalai know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["wo", "Wqkv"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

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
        bsz, _, _ = x.shape
        qkv = self.Wqkv(x)

        if seqlen is None:
            qkv = rearrange(
                qkv, "b s (h three d) -> b s three h d", three=3, d=self.head_dim
            )  # Put `three` in the penultimate dimention to concatenate different TensorParallel conveniently
        else:
            qkv = rearrange(qkv, "(b s) (h three d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)

        # qkv shift, the rotary embedding in flash attention module in performed by separating
        # the front and back parts, while most of others are done by odd-even methods.
        if not self.rot_embed_HF_impl:
            qkv[:, :, 0] = torch.cat([qkv[..., 0, :, ::2], qkv[..., 0, :, 1::2]], dim=-1)
            qkv[:, :, 1] = torch.cat([qkv[..., 1, :, ::2], qkv[..., 1, :, 1::2]], dim=-1)

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
            kv = torch.stack([qkv[:, :, 1], qkv[:, :, 2]], dim=2)
            q = qkv[:, :, 0]

            assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
            kv = _update_kv_cache(kv, inference_params, self.layer_idx)
            causal = None if inference_params.sequence_len_offset == 0 else False
            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                import math

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                scores = torch.einsum("blhd,bnhd->bhln", q, k) / math.sqrt(q.size(-1))
                if inference_params.sequence_len_offset == 0:  # A square matrix upon the first entry
                    attn_mask = inference_params.attention_mask[:, None, ...]
                    attn_mask = torch.logical_or(torch.ones_like(scores, dtype=torch.bool).triu(diagonal=1), attn_mask)
                else:
                    attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)
                    assert scores.size(-2) == 1
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)  # bsz x h x L x L
                context = torch.einsum("bhmn,bnhd->bmhd", scores, v)
            else:
                context = self.inner_cross_attn(q, kv, causal=causal)
        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")
        out = self.wo(context)
        return out

    def _packed_forward(self, x, inference_params=None, **kwargs):
        """
        we delete seqlen=None for lint check, cause this arg is not used.

        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """

        qkv = self.Wqkv(x)  # length x dimension
        qkv = rearrange(qkv, "t (h three d) -> t three h d", three=3, d=self.head_dim)  # total x 3 x n_head x d
        # qkv shift
        # the rotary embedding in flash attention module in performed by separating the front and back parts, while
        # most of others are done by odd-even methods.
        if not self.rot_embed_HF_impl:
            qkv[:, 0] = torch.cat([qkv[:, 0, :, ::2], qkv[:, 0, :, 1::2]], dim=-1)
            qkv[:, 1] = torch.cat([qkv[:, 1, :, ::2], qkv[:, 1, :, 1::2]], dim=-1)
        qkv = self.rotary_emb(qkv, kwargs.pop("indexes"))
        if inference_params is None:
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
        else:
            raise RuntimeError("Not support this right now")
        context = rearrange(context, "b h d -> b (h d)")  # recover shape
        out = self.wo(context)
        return out


class FeedForward(nn.Module):
    """
    FeedForward, use SwiGLU by default.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        self.w3 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )
        # Add tensor parallel attribute so that colossalai know it is a tensor parallel module.
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["w1", "w2", "w3"]:
                for param in getattr(self, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """
    RMS Normarlization.

    Args:
        dim (int): the dimention of model.
        eps (float): bias term. 1e-6 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.

    """

    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)
