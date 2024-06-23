from torch import nn
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
import torch.nn.functional as F
import torch
try:
    from xformers.ops import swiglu
    from xformers.ops.swiglu_op import SwiGLUFusedOp
except:  # 防止没有安装直接就没办法 import 了
    pass
from flash_attn.ops.fused_dense import all_reduce
from ..packed_fat.packed_module import RotaryEmbedding
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention, SelfAttention, CrossAttention
from flash_attn.ops.fused_dense import fused_dense_func
from einops import rearrange, repeat
from flash_attn.modules.mha import _update_kv_cache
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MaskFeat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = gpc.get_local_rank(ParallelMode.TENSOR)
        self.world_size = gpc.get_world_size(ParallelMode.TENSOR)
        
    def split_mask_model_parallel(self, mask):
        # mask [1, 1, h] -> [1, 1, h1], [1,1,h2], ...
        if self.world_size == 1:
            return mask
        mask_dim = mask.size(-1)
        assert (mask_dim % self.world_size) == 0, f'mask_dim: {mask_dim}, world_size: {self.world_size}'
        out_dim = mask_dim // self.world_size
        return mask[..., self.rank*out_dim:(self.rank+1)*out_dim]

    def forward(self, x, mask, input_is_parallel=False, print=False):
        if input_is_parallel:
            mask = self.split_mask_model_parallel(mask)
        if x.dim() == 2:
            mask = mask.squeeze(0)
        mask = mask.type_as(x)
        return x * mask.detach()

class OneDScaleParallelMHA(nn.Module):
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
        self.mask_feat = MaskFeat()

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, 'rotary_emb is not installed'
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, scale_base=rotary_emb_scale_base,
                                              device=device)
            # self.rotary_emb = RoPEForMultiSeqWithSinusoid(self.head_dim, max_len=2048)
        # rotary_emb = precompute_freqs_cis(self.head_dim, 2048)
        # self.register_buffer('rotary_emb', rotary_emb)

        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError('fused_dense is not installed')
        # self.wq = ColumnParallelLinear(embed_dim, embed_dim, process_group, bias=bias,
        #                                  sequence_parallel=sequence_parallel, **factory_kwargs)
        # self.wk = ColumnParallelLinear(embed_dim, embed_dim, process_group, bias=bias,
        #                                  sequence_parallel=sequence_parallel, **factory_kwargs)
        # self.wv = ColumnParallelLinear(embed_dim, embed_dim, process_group, bias=bias,
        #                                  sequence_parallel=sequence_parallel, **factory_kwargs)
        self.Wqkv = ColumnParallelLinear(embed_dim, 3 * embed_dim, process_group, bias=bias,
                                         sequence_parallel=sequence_parallel, **factory_kwargs)
        
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                                     attention_dropout=dropout)
        # output projection always have the bias (for now)
        self.wo = RowParallelLinear(embed_dim, embed_dim, process_group,
                                          sequence_parallel=sequence_parallel,
                                           bias=bias, **factory_kwargs)

    def forward(self, x, feat_mask=None, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get('indexes', None) is not None:
            return self._packed_forward(x=x, feat_mask=feat_mask, seqlen=seqlen, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, feat_mask=feat_mask, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, feat_mask=None, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        bsz, max_len, dim = x.shape
        # q, k, v = self.wq(x), self.wk(x), self.wv(x)  # bsz x max_len x dim
        # qkv = torch.stack([q, k, v], dim=2).reshape(bsz, max_len, -1)  # bsz x max_len x 3dim
        qkv = self.Wqkv(x)
        # qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, 'b s (h three d) -> b s three h d', three=3, d=self.head_dim)  # 这里将 three 放在倒数第二维是为了方便之后 concat 不同的tp
        else:
            qkv = rearrange(qkv, '(b s) (h three d) -> b s three h d', s=seqlen, three=3,
                            d=self.head_dim)
        # if torch.distributed.get_rank() == 0:
        #     print(qkv[:, :, 0, :3, :3] - q.reshape(bsz, max_len, -1, self.head_dim)[:, :, :3, :3])
        #     print(self.Wqkv.weight[0, :4])
        # exit()
        # qkv shift，因为 flash attention 的 RoPE 是前半部分和后半部分分开的方式进行RoPE的，而其它大部分是通过奇偶进行
        qkv[:, :, 0] = torch.cat([qkv[..., 0, :, ::2], qkv[..., 0, :, 1::2]], dim=-1)
        qkv[:, :, 1] = torch.cat([qkv[..., 1, :, ::2], qkv[..., 1, :, 1::2]], dim=-1)

        if inference_params is None:
            if self.rotary_emb_dim > 0:
                # q, k = apply_rotary_emb(qkv[:, :, 0], qkv[:, :, 1], freqs_cis=self.rotary_emb[:max_len])
                # qkv = torch.stack([q, k, qkv[:,:,1]], dim=2)
                qkv = self.rotary_emb.eval_forward(qkv)
            if not self.checkpointing:
                context = self.inner_attn(qkv)
            else:
                context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv)
        else:
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb.eval_forward(qkv, seqlen_offset=inference_params.sequence_len_offset)
                # q, k = apply_rotary_emb(qkv[:, :, 0], qkv[:, :, 1], freqs_cis=self.rotary_emb[inference_params.sequence_len_offset:inference_params.sequence_len_offset+max_len])
            kv = torch.stack([qkv[:, :, 1], qkv[:, :, 2]], dim=2)
            q = qkv[:, :, 0]
            assert self.layer_idx is not None, 'Generation requires layer_idx in the constructor'
            kv = _update_kv_cache(kv, inference_params, self.layer_idx)
            causal = None if inference_params.sequence_len_offset == 0 else False
            if hasattr(inference_params, 'attention_mask') and inference_params.attention_mask is not None:
                import math
                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                scores = torch.einsum('blhd,bnhd->bhln', q, k)/math.sqrt(q.size(-1))
                if inference_params.sequence_len_offset == 0:  # 第一次进入，是个方阵
                    attn_mask = inference_params.attention_mask[:,None,...]
                    attn_mask = torch.logical_or(torch.ones_like(scores, dtype=torch.bool).triu(diagonal=1), attn_mask)
                else:
                    attn_mask = inference_params.attention_mask[:,-1,:].view(bsz, 1, 1, -1)
                    assert scores.size(-2) == 1
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum('bhmn,bnhd->bmhd', scores, v)
            else:
                context = self.inner_cross_attn(q, kv, causal=causal)
        if seqlen is None:
            context = rearrange(context, 'b s h d -> b s (h d)')
        else:
            context = rearrange(context, 'b s h d -> (b s) (h d)')
        # import os
        # if os.environ.get('print'):
        #     if torch.distributed.get_rank() == 0:
        #         print(f"after attn: {context[:, :, :3].tolist()}")
        #     exit()
        
        if context.dim() == 2:
            feat_mask = feat_mask.view(1, -1)
        context = self.mask_feat(context, feat_mask, input_is_parallel=True)
        out = self.wo(context)
        out = self.mask_feat(out, feat_mask)
        return out

    def _packed_forward(self, x, feat_mask=None, seqlen=None, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        # q, k, v = self.wq(x), self.wk(x), self.wv(x)  # max_len x dim
        # q = q.reshape(-1, self.num_heads, self.head_dim)
        # v = v.reshape(-1, self.num_heads, self.head_dim)
        # k = k.reshape(-1, self.num_heads, self.head_dim)
        # qkv = torch.stack([q, k, v], dim=1)  # length x 3 x n_head x head_dim
        qkv = self.Wqkv(x)  # length x dimension
        # if seqlen is None:
        #     qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, d=self.head_dim)
        # else:
        #     qkv = rearrange(qkv, '(b s) (three h d) -> b s three h d', s=seqlen, three=3,
        #                     d=self.head_dim)
        qkv = rearrange(qkv, 't (h three d) -> t three h d', three=3, d=self.head_dim)  # total x 3 x n_head x d
        # qkv shift，因为 flash attention 的 RoPE 是前半部分和后半部分分开的方式进行RoPE的，而其它大部分是通过奇偶进行
        qkv[:, 0] = torch.cat([qkv[:, 0, :, ::2], qkv[:, 0, :, 1::2]], dim=-1)
        qkv[:, 1] = torch.cat([qkv[:, 1, :, ::2], qkv[:, 1, :, 1::2]], dim=-1)
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

        if context.dim() == 2:
            feat_mask = feat_mask.view(1, -1)
        context = self.mask_feat(context, feat_mask, input_is_parallel=True)
        out = self.wo(context)
        out = self.mask_feat(out, feat_mask)

        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        # dim: int,
        # hidden_dim: int,
        in_features, hidden_features, out_features=None, activation='gelu_approx',
        process_group = None, bias=True, sequence_parallel=False, checkpoint_lvl=0, 
        heuristic='auto', device=None, dtype=None,
        multiple_of: int = 256  # 是多少的倍数
    ):
        super().__init__()
        # hidden_dim = int(2 * in_features / 3)
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            # in_features, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w3 = ColumnParallelLinear(
            in_features, hidden_features, process_group, bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.w2 = RowParallelLinear(
            # hidden_dim, out_features, bias=False, input_is_parallel=True, init_method=lambda x: x
            hidden_features, out_features, process_group, bias = bias, sequence_parallel=False, device=device, dtype=dtype
        )
        self.mask_feat = MaskFeat()

    def forward(self, x, feat_mask=None, ffn_mask=None):
        # The input x and the output w2 share the same mask.
        # The hidden feature w1*w3 uses a different mask.
        # Assume the input x is masked before: x = [x_1, ..., x_c, 0, ..., 0]
        w1 = F.silu(self.w1(x))
        w3 = self.w3(x)
        # if hidden_dim is also enlarged
        h = self.mask_feat(w1 * w3, ffn_mask, input_is_parallel=True)
        w2 = self.mask_feat(self.w2(h), feat_mask) # [y_1, ..., y_c, 0, ..., 0]
        return w2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x, feat_mask):
        if x.dim() == 2:
            feat_mask = feat_mask.squeeze(0)
        act_num = feat_mask.sum(dim=-1, keepdim=True).to(x)
        # act_num = (feat_mask * feat_mask).sum(dim=-1, keepdim=True)
        o = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / act_num + self.eps)
        return o

    def forward(self, x, feat_mask=None):
        output = self._norm(x.float(), feat_mask).type_as(x)
        return output * self.weight
