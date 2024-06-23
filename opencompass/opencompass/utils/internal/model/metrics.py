import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage


class AccPerplex:
    def __init__(self, device, tp_pg, dp_pg):
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_log_probs = torch.Tensor([0]).to(device=device)
        self.tp_pg = tp_pg
        self.dp_pg = dp_pg
        self.tp_local_rank = torch.distributed.get_rank(self.tp_pg)

    def __call__(self, logits, labels):
        return self.update(logits, labels)

    def update(self, logits, labels):
        with torch.no_grad():
            if isinstance(logits, (list, tuple)):
                logits = logits[0]  # 说明是带有residual的
            logits = logits.detach().clone()
            labels = labels.detach().clone()
            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            pred_shift = self.tp_local_rank*logits.shape[-1]  # 根据当前的rank有一个shift，因为logits是被切分掉了

            acc = (shift_labels == (shift_logits.argmax(dim=-1)+pred_shift)).sum()
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)
            mask = shift_labels.ne(-100)
            self.right += acc  # 这里不需要 masked_fill 是由于 -100 反正也取不到
            self.total += mask.sum()

            # perplexity 的计算
            logits_max = torch.max(shift_logits, dim=-1)[0]
            torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=self.tp_pg)
            # Subtract the maximum value.
            shift_logits = shift_logits.sub(logits_max.unsqueeze(dim=-1))

            # Get the partition's vocab indecies
            partition_vocab_size = shift_logits.size()[-1]
            vocab_start_index = partition_vocab_size * self.tp_local_rank
            vocab_end_index = vocab_start_index + partition_vocab_size

            # Create a mask of valid vocab ids (1 means it needs to be masked).
            target_mask = (shift_labels < vocab_start_index) | (shift_labels >= vocab_end_index)
            masked_target = shift_labels - vocab_start_index
            masked_target[target_mask] = 0

            # Get predicted-logits = logits[target].
            # For Simplicity, we convert logits to a 2-D tensor with size
            # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
            logits_2d = shift_logits.view(-1, partition_vocab_size)
            masked_target_1d = masked_target.view(-1)
            arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
            predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
            predicted_logits_1d = predicted_logits_1d.clone().contiguous()
            predicted_logits = predicted_logits_1d.view_as(shift_labels)  # bsz x max_len
            predicted_logits[target_mask] = 0.0
            # All reduce is needed to get the chunks from other GPUs.
            torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)

            pred_exp_logits = torch.exp(predicted_logits)
            # Sum of exponential of logits along vocab dimension across all GPUs.
            sum_exp_logits = torch.exp(shift_logits).sum(dim=-1)
            torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=self.tp_pg)

            total_log_probs = -(pred_exp_logits/sum_exp_logits).log().masked_fill(shift_labels.eq(-100), 0).sum()
            self.total_log_probs += total_log_probs

    def get_metric(self, reset=True):
        if is_no_pp_or_last_stage() and self.dp_pg is not None:
            torch.distributed.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            torch.distributed.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)
            torch.distributed.all_reduce(self.total_log_probs, op=torch.distributed.ReduceOp.SUM, group=self.dp_pg)

        acc = round((self.right/self.total).item(), 4)
        perplexity = round(torch.exp(self.total_log_probs / self.total).item(), 4)
        # res['perplexity'] = round(perplexity.item(), 4)
        if reset:
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_log_probs.fill_(0)
        return {'acc': acc, 'perplexity': perplexity}
