import enum

import torch
import torch.nn.functional as F
from torch import nn

try:
    from colossalai._C import scaled_masked_softmax
except ImportError:
    from colossalai.kernel.op_builder.scaled_masked_softmax import (
        ScaledMaskedSoftmaxBuilder,
    )

    scaled_masked_softmax = ScaledMaskedSoftmaxBuilder().load()

try:
    from colossalai._C import scaled_upper_triangle_masked_softmax
except ImportError:
    from colossalai.kernel.op_builder.scaled_upper_triangle_masked_softmax import (
        ScaledUpperTrainglemaskedSoftmaxBuilder,
    )

    scaled_upper_triang_masked_softmax = ScaledUpperTrainglemaskedSoftmaxBuilder().load()


class ScaleMaskSoftmax(nn.Module):
    """
    Fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: Flag to indicate if input in fp16 data format.
        input_in_bf16: Flag to indicate if input in bf16 data format.
        attn_mask_type: Attention mask type (pad or causal)
        scaled_masked_softmax_fusion: Flag to indicate user want to use softmax fusion
        softmax_in_fp32: If True, softmax in performed at fp32 precision.
        scale: Scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, cur_input, mask):
        # The unnecessary position in the mask is 1, remember
        # [b, np, sq, sk]
        assert input.dim() == 4

        return self.forward_torch_softmax(cur_input, mask)

    def forward_torch_softmax(self, cur_input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            cur_input = cur_input.float()

        if self.scale is not None:
            cur_input = cur_input * self.scale

        mask_output = cur_input.masked_fill(mask, -50000.0) if mask is not None else cur_input
        probs = F.softmax(mask_output, dim=-1)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence

        1.  Scale the tensor.
        2.  Apply upper triangular mask (typically used in gpt models).
        3.  Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triangle_masked_softmax.forward(inputs, scale_t[0])

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triangle_masked_softmax.backward(output_grads, softmax_results, scale_t[0])

        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence

        1.  Scale the tensor.
        2.  Apply the mask.
        3.  Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class FusedScaleMaskSoftmax(nn.Module):
    """
    Fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: Flag to indicate if input in fp16 data format.
        input_in_bf16: Flag to indicate if input in bf16 data format.
        attn_mask_type: Attention mask type (pad or causal)
        scaled_masked_softmax_fusion: Flag to indicate user want to use softmax fusion
        mask_func: Mask function to be applied.
        softmax_in_fp32: If True, softmax in performed at fp32 precision.
        scale: Scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, cur_input, mask):
        # [b, np, sq, sk]
        assert cur_input.dim() == 4

        if self.is_kernel_available(mask, *cur_input.size()):
            return self.forward_fused_softmax(cur_input)
        else:
            return self.forward_torch_softmax()

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and (mask is not None or self.attn_mask_type == AttnMaskType.causal)  # mask tensor must not be None
            and 16 < sk <= 2048  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 2048:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, cur_input):
        b, np, sq, sk = cur_input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            cur_input = cur_input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(cur_input, scale)
            return probs.view(b, np, sq, sk)
        else:
            raise RuntimeError("Should not run into here")

    def forward_torch_softmax(self):
        raise RuntimeError("Should not run here")

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        try:
            import colossalai._C.scaled_masked_softmax
        except ImportError:
            raise RuntimeError("ScaledMaskedSoftmax requires cuda extensions")  # pylint: disable=W0707

        return colossalai._C.scaled_masked_softmax.get_batch_per_block(sq, sk, b, np)
