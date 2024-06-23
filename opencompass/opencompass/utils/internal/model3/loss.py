from colossalai import nn as col_nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
from torch import nn


class GPTLMLoss(nn.Module):
    """
    Loss function for GPT Language Model.
    """

    def __init__(self, parallel_output=True):
        super().__init__()
        if parallel_output:
            self.loss_fn = col_nn.CrossEntropyLoss(
                reduction=True
            )  # The loss in this place is bound to the gather_output initialized by VocabParallelClassifier1D
        else:
            # Here, the output will gather output is set in the model, so use ordinary loss
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, labels):
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(
            shift_logits, shift_labels
        )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        return loss


class FlashGPTLMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, parallel_output=True, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                if gpc.get_global_rank() == 0:
                    print(f"use label_smoothing: {label_smoothing}")
        else:
            label_smoothing = 0
        self.label_smoothing = label_smoothing

        if parallel_output:
            self.loss_fn = FlashCrossEntropyLoss(
                reduction="mean",
                inplace_backward=True,
                process_group=gpc.get_group(ParallelMode.PARALLEL_1D),
                label_smoothing=label_smoothing,
            )  # The loss in this place is bound to the gather_output initialized by VocabParallelClassifier1D
        else:
            # Here, the output will gather output is set in the model, so use ordinary loss
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, *args):
        if len(args) == 3:
            # residual is to match prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(
            shift_logits, shift_labels
        )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        return loss
