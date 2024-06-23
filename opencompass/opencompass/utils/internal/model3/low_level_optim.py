import collections
import math
import os
import sys
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.megatron_timers import timer
from colossalai.zero.sharded_optim._utils import get_grad_accumulate_object
from colossalai.zero.sharded_optim.low_level_optim import (
    LowLevelZeroOptimizer,
    _calc_l2_norm,
    compute_norm,
    flatten,
    is_model_parallel_parameter,
    release_param_grad,
)
from torch.optim import Optimizer

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def compute_layer_norm(gradients, parameters, parameter_names, norm_type=2):
    """Get the norm of each layer.

    Arguments:
        gradients (Iterable[Tensor]): The gradient value
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to
        parameter_names (Iterable[str]): The name of the parameter each gradient corresponds to
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Dict: Total norm of the parameters in each layers.
    """
    assert norm_type == 2, "Only support l2 norm in layer now"
    # Norm parameters.
    norm_type = float(norm_type)

    for param in parameters:
        if hasattr(param, "colo_attr") and param.colo_attr.sharded_data_tensor.is_sharded:
            raise RuntimeError("Currently not support Zero3")

    tensor_parallel_grads = []
    tensor_parallel_grads_names = []
    for g, p, n in zip(gradients, parameters, parameter_names):
        # TODO consider the pipeline shared parameter

        if (
            gpc.is_initialized(ParallelMode.PIPELINE)
            and hasattr(p, "pipeline_shared_module_pg")
            and dist.get_rank(p.pipeline_shared_module_pg) == 0
        ):  # if shared between different pipe, only count 0
            tensor_parallel_grads.append(g.data.float())
            tensor_parallel_grads_names.append(n)
            # if is_model_parallel_parameter(p):
            #     tensor_parallel_grads.append(g.data.float())
            # else:
            #     no_tensor_parallel_grads.append(g.data.float())
        elif (
            gpc.is_initialized(ParallelMode.PIPELINE)
            and hasattr(p, "pipeline_shared_module_pg")
            and dist.get_rank(p.pipeline_shared_module_pg) != 0
        ):
            continue
        elif (
            gpc.is_initialized(ParallelMode.TENSOR)
            and not is_model_parallel_parameter(p)
            and gpc.get_local_rank(ParallelMode.TENSOR) == 0
        ):  # if not used in each chunk, such as layernorm
            # no_tensor_parallel_grads.append(g.data.float())
            tensor_parallel_grads.append(g.data.float())
            tensor_parallel_grads_names.append(n)
        elif is_model_parallel_parameter(p):
            # reductor = (gpc.get_world_size(ParallelMode.TENSOR) / getattr(p, NUM_PARTITIONS))**(1 / norm_type)
            tensor_parallel_grads.append(g.data.float())
            tensor_parallel_grads_names.append(n)
        elif gpc.get_local_rank(ParallelMode.TENSOR) != 0:
            continue
        else:
            raise RuntimeError("Should not arrive here")

    layers_grads = defaultdict(list)
    layers_norms = defaultdict(float)
    for idx, name in enumerate(tensor_parallel_grads_names):
        if "layers" in name:
            layer_idx = int(name.split(".")[2])
            layers_grads[layer_idx].append(tensor_parallel_grads[idx])
        elif "embeddings" in name:
            layers_grads["embeddings"].append(tensor_parallel_grads[idx])
        elif "output" in name:
            layers_grads["output"].append(tensor_parallel_grads[idx])
        elif "norm" in name:
            layers_grads["norm"].append(tensor_parallel_grads[idx])

    for layer_idx in layers_grads:
        layers_norms[layer_idx] = (_calc_l2_norm(layers_grads[layer_idx]) ** norm_type).item()

    objs = [None for _ in range(gpc.get_world_size(ParallelMode.DATA))]
    dist.all_gather_object(objs, layers_norms, group=gpc.get_group(ParallelMode.DATA))

    layers_norms = {
        k: math.sqrt(v) for k, v in dict(sum(map(collections.Counter, objs), collections.Counter())).items()
    }
    return layers_norms


class ModifiedLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    """
    Modified Low Level Zero Optimizer.
    """

    def _has_inf_or_nan(self, tensor):
        try:
            tensor_mean = float(tensor.mean())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if tensor_mean == float("inf") or tensor_mean == -float("inf"):
                return True
            return False

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                if avg_grad is not None and self._has_inf_or_nan(avg_grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce over MODEL ranks
        # dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MODEL))

        # all-reduce over all ranks
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL))

        return self._found_overflow.item() > 0

    def _step(self, closure=None):
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm
        timer("cal_norm").start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=self._param_store.get_fp16_params_by_rank_group(
                        group_id=group_id, rank=self._local_rank
                    ),
                )
                if norm_group == -1:
                    timer("cal_norm").stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup
        self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            from utils.monitor_and_alert import get_process_rank, send_alert_message

            if get_process_rank() == 0:
                send_alert_message(message="Overflow occurs, please check it.")
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
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm > 0:
            global_norm = sum(norm_groups) ** 0.5
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)
        timer("cal_norm").stop()
        # update the parameters
        timer("step").start()

        # to avoid modify engine.update(), we use envvar to pass arguments
        enable_skip = os.environ.get("ENABLE_SKIP_PARAM_UPDT", "False")
        if enable_skip == "True":
            grad_norm_baseline = float(os.environ.get("GRAD_NORM_BASE"))
            grad_norm_max = float(os.environ.get("GRAD_NORM_MAX"))
            grad_norm_ref = max(grad_norm_baseline, grad_norm_max)
            if (global_norm / loss_scale) > grad_norm_ref:
                # skip weight update if normalized gradient increased steeply
                timer("step").stop()
                from utils.logger import LLM_LOGGER as logger

                logger.warning(
                    f"skip weight update because normalized gradient({global_norm/loss_scale}) \
> reference ({grad_norm_ref}).",
                    ranks=[0],
                )
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
        timer("step").stop()
        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        return True, global_norm / loss_scale


class NoPPModifiedLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    """
    Modified Low Level Zero Optimizer without pipeline parallel, and support overlap_communication.
    """

    def backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

        timer("dp_sync").start()
        dist.barrier()
        timer("dp_sync").stop()

        # 由于删掉了 ddp 的同步，这里需要手动同步一下
        if NoPPModifiedLowLevelZeroOptimizer.do_reduce_and_sync_grad():
            timer("sync_grad").start()
            self._reduce_grad()
            self.sync_grad()
            timer("sync_grad").stop()

        # clear reduced grads
        # (WGT): There should be no need to add a judgment on accumulate_grad in this place.
        # When we skip the stage of reduce_grad() + sync_grad(), emptying an already empty
        # param_store will not have any effect.
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()

    def _has_inf_or_nan(self, tensor):
        try:
            tensor_mean = float(tensor.mean())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if tensor_mean == float("inf") or tensor_mean == -float("inf"):
                return True
            return False

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                if avg_grad is not None and self._has_inf_or_nan(avg_grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce over MODEL ranks
        # dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.MODEL))

        # all-reduce over all ranks
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.GLOBAL))

        return self._found_overflow.item() > 0

    def _step(self, closure=None):
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm
        timer("cal_norm").start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=self._param_store.get_fp16_params_by_rank_group(
                        group_id=group_id, rank=self._local_rank
                    ),
                )
                if norm_group == -1:
                    timer("cal_norm").stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup
        self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            from utils.monitor_and_alert import get_process_rank, send_alert_message

            if get_process_rank() == 0:
                send_alert_message(message="Overflow occurs, please check it.")
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
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm > 0:
            global_norm = sum(norm_groups) ** 0.5
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)
        timer("cal_norm").stop()
        # update the parameters
        timer("step").start()

        # to avoid modify engine.update(), we use envvar to pass arguments
        enable_skip = os.environ.get("ENABLE_SKIP_PARAM_UPDT", "False")
        if enable_skip == "True":
            grad_norm_baseline = float(os.environ.get("GRAD_NORM_BASE"))
            grad_norm_max = float(os.environ.get("GRAD_NORM_MAX"))
            grad_norm_ref = max(grad_norm_baseline, grad_norm_max)
            if (global_norm / loss_scale) > grad_norm_ref:
                # skip weight update if normalized gradient increased steeply
                timer("step").stop()
                from utils.logger import LLM_LOGGER as logger

                logger.warning(
                    f"skip weight update because normalized "
                    f"gradient({global_norm/loss_scale}) > reference ({grad_norm_ref}).",
                    ranks=[0],
                )
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
        timer("step").stop()
        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        return True, global_norm / loss_scale

    @staticmethod
    def do_reduce_and_sync_grad():
        step_count = int(os.environ["STEP_COUNT"])
        accumulate_grad = int(os.environ["GRAD_ACC_NUM"])

        if accumulate_grad == 1:
            # When args.accumulate_grads == 1, we always sync grad.
            return True
        else:
            return (step_count + 1) % accumulate_grad == 0

    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    # determines the reduction destionation rank
                    # this is only valid for stage 2
                    # dst_rank = None means using all-reduce
                    # else using reduce
                    if self._partition_grads:
                        reduce_rank = self._param_store.get_param_rank(param)
                    else:
                        reduce_rank = None

                    def _define_and_attach(param, reduce_rank):
                        # get the AccumulateGrad object of the param itself
                        accum_grad_obj = get_grad_accumulate_object(param)
                        self._grad_store.add_accumulate_grad_object(accum_grad_obj)

                        reduction_func = partial(
                            self._reduce_and_remove_grads_by_bucket, param=param, reduce_rank=reduce_rank
                        )

                        # define hook
                        # NOT IMPORTANT BUT GOOD TO KNOW:
                        # args here is not grad, but allow_unreacable and accumulate_grad
                        def reduce_grad_hook(*args):  # pylint: disable=W0613
                            # (WGT): Skip reduce hook when accumulate_grad is triggered.
                            # accumulate_grad is the command line parameter in utils.py,
                            # not to be confused with accumulate_grad in low_level_optimzer.
                            if NoPPModifiedLowLevelZeroOptimizer.do_reduce_and_sync_grad():
                                reduction_func()

                        accum_grad_obj.register_hook(reduce_grad_hook)

                    _define_and_attach(param, reduce_rank)


class LowLevelZeroOptimizerWithLayerNorms(ModifiedLowLevelZeroOptimizer):
    """
    Modified Low Level Zero Optimizer with recording layer gradient norms.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_scale=2**16,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        hysteresis=2,
        max_scale: int = 2**24,
        clip_grad_norm=0.0,
        verbose=False,
        reduce_bucket_size=1024 * 1024,
        communication_dtype=None,
        overlap_communication=False,
        partition_grad=False,
        dp_parallel_mode=ParallelMode.DATA,
        mp_parallel_mode=ParallelMode.MODEL,
        cpu_offload=False,
        forced_dtype=None,
        named_parameters=None,
    ):
        self.named_parameters = named_parameters
        super().__init__(
            optimizer=optimizer,
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            clip_grad_norm=clip_grad_norm,
            verbose=verbose,
            reduce_bucket_size=reduce_bucket_size,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            partition_grad=partition_grad,
            dp_parallel_mode=dp_parallel_mode,
            mp_parallel_mode=mp_parallel_mode,
            cpu_offload=cpu_offload,
            forced_dtype=forced_dtype,
        )

    def _partition_param_list(self, param_list):
        """
        Partition the parameter list into chunks of roughly equal size.
        """
        new_param_list = []
        assert isinstance(self.named_parameters, dict)
        assert len(param_list) == len(self.named_parameters)
        for i, name in enumerate(self.named_parameters.keys()):
            new_param_list.append([i, self.named_parameters[name], name])

        numel_per_rank = [0 for _ in range(self._world_size)]
        params_per_rank = [[] for _ in range(self._world_size)]
        params_info_per_rank = [[] for _ in range(self._world_size)]

        # partititon the parameters in a greedy fashion
        new_sorted_params = sorted(new_param_list, key=lambda x: x[1].numel(), reverse=True)
        for idx, param, name in new_sorted_params:
            # allocate this parameter to the rank with
            # the smallest numel for load balancing purpose
            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            numel_per_rank[rank_to_go] += param.numel()
            params_per_rank[rank_to_go].append(param)
            params_info_per_rank[rank_to_go].append(str(idx) + "_" + name)

        # Core info to be used in the compute_layer_norm method.
        self._layer_grad_info = params_info_per_rank[self._local_rank]

        if self._verbose:
            self._logger.info(
                f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}",
                ranks=[0],
                parallel_mode=self._dp_parallel_mode,
            )
        return params_per_rank

    def get_layer_grad_norm_groups(self, loss_scale=1.0):
        """
        Compute the gradient norms of each layer.
        """
        layer_norm_groups = []
        for group_id in range(self.num_param_groups):
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                layer_norm_group = compute_layer_norm(
                    gradients=gradients,
                    parameters=self._param_store.get_fp16_params_by_rank_group(
                        group_id=group_id, rank=self._local_rank
                    ),
                    parameter_names=self._layer_grad_info,
                )
                layer_norm_group = {k: v / loss_scale for k, v in layer_norm_group.items()}
                layer_norm_groups.append(layer_norm_group)

        return layer_norm_groups

    def _step(self, closure=None):
        assert closure is None, "closure is not supported by step()"

        # check for overflow
        found_inf = self._check_overflow()
        # Because you may encounter inf when computing norm
        timer("cal_norm").start()
        norm_groups = []
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=self._param_store.get_fp16_params_by_rank_group(
                        group_id=group_id, rank=self._local_rank
                    ),
                )
                if norm_group == -1:
                    timer("cal_norm").stop()
                    found_inf = True
                    break
                norm_groups.append(norm_group)

        loss_scale = float(self.loss_scale.item())  # backup

        self.grad_scaler.update(found_inf)
        # update loss scale if overflow occurs
        if found_inf:
            from utils.monitor_and_alert import get_process_rank, send_alert_message

            if get_process_rank() == 0:
                send_alert_message(message="Overflow occurs, please check it.")
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, None
        self.layer_norm_groups = self.get_layer_grad_norm_groups(loss_scale=loss_scale)

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
            assert (
                param_shape == flat_fp32_avg_grads.shape
            ), f"fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}"

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm > 0:
            global_norm = sum(norm_groups) ** 0.5
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)
        timer("cal_norm").stop()
        # update the parameters
        timer("step").start()

        # to avoid modify engine.update(), we use envvar to pass arguments
        enable_skip = os.environ.get("ENABLE_SKIP_PARAM_UPDT")
        grad_norm_baseline = float(os.environ.get("GRAD_NORM_BASE"))
        grad_norm_max = float(os.environ.get("GRAD_NORM_MAX"))
        grad_norm_ref = max(grad_norm_baseline, grad_norm_max)
        if enable_skip == "True" and (global_norm / loss_scale) > grad_norm_ref:
            # skip weight update if normalized gradient increased steeply
            timer("step").stop()
            from utils.logger import LLM_LOGGER as logger

            logger.warning(
                f"skip weight update because normalized gradient({global_norm/loss_scale}) \
> reference ({grad_norm_ref}).",
                ranks=[0],
            )
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
        timer("step").stop()
        # update gradients may not be needed here, because the sync_params function is used in initialization,
        # so synchronization is maintained
        return True, global_norm / loss_scale
