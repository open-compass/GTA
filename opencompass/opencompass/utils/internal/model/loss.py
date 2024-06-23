from torch import nn
from colossalai import nn as col_nn
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


class GPTLMLoss(nn.Module):

    def __init__(self, parallel_output=True):
        super().__init__()
        if parallel_output:
            self.loss_fn = col_nn.CrossEntropyLoss(reduction=True)  # 这个地方的loss和VocabParallelClassifier1D初始化的gather_output是绑定的
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')  # 这里由于在model中设置了输出会gather output，所以使用普通的 loss

    def forward(self, logits, labels, **kwargs):
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(shift_logits, shift_labels)  # 这里不需要考虑ignore_index的问题，是由于loss计算会通过计算范围来计算的，而-100一定在这个范围外，所以没有问题

        return loss


class FlashGPTLMLoss(nn.Module):

    def __init__(self, parallel_output=True):
        super().__init__()
        if parallel_output:
            self.loss_fn = FlashCrossEntropyLoss(reduction='mean', inplace_backward=True, process_group=gpc.get_group(ParallelMode.PARALLEL_1D))  # 这个地方的loss和VocabParallelClassifier1D初始化的gather_output是绑定的
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')  # 这里由于在model中设置了输出会gather output，所以使用普通的 loss

    def forward(self, *args):
        if len(args)==3:
            # residual 是为了配合prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # 使用postnorm的情况
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(shift_logits, shift_labels)  # 这里不需要考虑ignore_index的问题，是由于loss计算会通过计算范围来计算的，而-100一定在这个范围外，所以没有问题

        return loss