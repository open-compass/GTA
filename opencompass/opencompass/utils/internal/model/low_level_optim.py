from colossalai.zero.sharded_optim.low_level_optim import LowLevelZeroOptimizer, compute_norm, flatten, release_param_grad 
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.megatron_timers import timer
from colossalai.logging import get_dist_logger
import torch.distributed as dist

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.monitor_and_alert import send_alert_message, get_process_rank


logger = get_dist_logger()

class ModifiedLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    def _step(self, closure=None):
        assert closure is None, 'closure is not supported by step()'

        # check for overflow
        found_inf = self._check_overflow()
        # 因为 compute norm 的时候可能遇到 inf
        timer('cal_norm').start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=self._param_store.get_fp16_params_by_rank_group(group_id=group_id,
                                                                               rank=self._local_rank))
                if norm_group == -1:
                    timer('cal_norm').stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup
        self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            if get_process_rank() == 0:
                send_alert_message(message=f"Overflow occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, None

        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        global_norm = 0
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)

            # create flat gradient for the flat fp32 params
            fp16_avg_grads = gradients
            flat_fp16_avg_grads = flatten(fp16_avg_grads)

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert param_shape == flat_fp32_avg_grads.shape, \
                f'fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}'

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm>0:
            global_norm = sum(norm_groups)**0.5
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)
        timer('cal_norm').stop()
        # update the parameters
        timer('step').start()
        
        # to avoid modify engine.update(), we use envvar to pass arguments
        enable_skip = os.environ.get('ENABLE_SKIP_PARAM_UPDT')
        grad_norm_baseline = float(os.environ.get('GRAD_NORM_BASE'))
        grad_norm_max = float(os.environ.get('GRAD_NORM_MAX'))
        grad_norm_ref = max(grad_norm_baseline, grad_norm_max)
        if enable_skip == "True" and (global_norm/loss_scale) > grad_norm_ref:
            #skip weight update if normalized gradient increased steeply
            timer('step').stop()
            logger.warning(f"skip weight update because normalized "
                           f"gradient({global_norm/loss_scale}) > reference ({grad_norm_ref}).", ranks=[0])
            # encode grad_norm as -99.0 to indicate this case
            return False, -99.0

        self.optim.step()
        # release the fp32 grad
        release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())

        # update fp16 partition updated by the current rank
        for group_id in range(len(self._fp16_param_groups)):
            fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=self._local_rank, group_id=group_id)
            fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
            fp16_param.data.copy_(fp32_param)

        # broadcast the updated model weights
        handles = []
        for group_id in range(self.num_param_groups):
            for rank in range(self._world_size):
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)
                rank = gpc.get_ranks_in_group(ParallelMode.DATA)[rank]  # need to convert to the global rank
                handle = dist.broadcast(fp16_param, src=rank, group=self._dp_group, async_op=True)
                handles.append(handle)

        for handle in handles:
            handle.wait()
        timer('step').stop()
        # 这里可能不需要 update gradients , 是由于在初始化中使用 sync_params 函数，所以保持了同步
        return True, global_norm/loss_scale
