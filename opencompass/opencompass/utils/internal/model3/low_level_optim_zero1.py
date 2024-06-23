import os
import sys
from functools import partial

import torch
import torch.distributed as dist
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import is_using_pp
from colossalai.utils.cuda import get_current_device
from colossalai.utils.megatron_timers import timer
from colossalai.zero.sharded_optim._utils import (
    get_grad_accumulate_object,
    has_inf_or_nan,
    release_param_grad,
    sync_param,
)
from colossalai.zero.sharded_optim.bookkeeping import (
    BucketStore,
    GradientStore,
    ParameterStore,
)
from colossalai.zero.sharded_optim.low_level_optim import (
    LowLevelZeroOptimizer,
    compute_norm,
    flatten,
)
from torch.optim import Optimizer

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class NoPPModifiedLowLevelZeroOptimizer(LowLevelZeroOptimizer):
    """
    Modified Low Level Zero Optimizer without pipeline parallel, and support overlap_communication.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        # grad scaler config
        initial_scale=2**16,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        hysteresis=2,
        max_scale: int = 2**24,
        # grad clipping
        clip_grad_norm=0.0,
        verbose=False,
        # communication
        reduce_bucket_size=1024 * 1024,
        communication_dtype=None,
        overlap_communication=False,
        # stage 2
        partition_grad=False,
        dp_parallel_mode=ParallelMode.DATA,
        mp_parallel_mode=ParallelMode.MODEL,
        # cpu offload
        cpu_offload=False,
        # forced dtype
        forced_dtype=None,
    ):
        assert partition_grad is False, "unsupport zero2 or zero3"
        super().__init__(optimizer=optimizer)
        if is_using_pp() and overlap_communication:
            raise RuntimeError(
                "The pipeline parallelism is not compatible with overlap_communication, "
                "please set overlap_communication=False if you want to use the pipeline parallelism."
            )
        if is_using_pp() and partition_grad:
            raise RuntimeError(
                "The pipeline parallelism is not compatible with Zero2, "
                "please set partition_grad=False if you want to use the pipeline parallelism."
            )

        assert partition_grad is False, "NoPPModifiedLowLevelZeroOptimizer not support partition_grad by now."

        self._dtype = self.optim.param_groups[0]["params"][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # stage 2
        self._partition_grads = partition_grad

        # cpu_offload
        self._cpu_offload = cpu_offload

        # get process groups
        self._dp_parallel_mode = dp_parallel_mode
        self._mp_parallel_mode = mp_parallel_mode
        self._local_rank = gpc.get_local_rank(dp_parallel_mode)
        self._world_size = gpc.get_world_size(dp_parallel_mode)

        self._dp_group = gpc.get_group(dp_parallel_mode)
        if gpc.is_initialized(mp_parallel_mode) and gpc.get_world_size(mp_parallel_mode) > 1:
            self._mp_group = gpc.get_group(mp_parallel_mode)
        else:
            self._mp_group = None

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            verbose=verbose,
        )
        self._found_overflow = torch.FloatTensor([0]).to(get_current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group["params"]
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self._dp_parallel_mode)
        self._grad_store = GradientStore(self._dp_parallel_mode)
        self._bucket_store = BucketStore(self._dp_parallel_mode)

        # (WGT): We need to record the rank in which parameter groups are not assigned parameters.
        self.param_group_has_params = []
        self.param_group_no_params_ranks = []
        self.padding_grad = torch.zeros([32], dtype=self._dtype, device=get_current_device())
        self.padding_tensor = torch.zeros([32], dtype=self._dtype, device=get_current_device())

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            # assign parameters to ranks
            # the params in the list are sorted
            params_per_rank, no_params_ranks = self._partition_param_list(group_params)
            self.param_group_no_params_ranks.append(no_params_ranks)
            self.param_group_has_params.append(self._local_rank not in no_params_ranks)

            # store the mapping between param to rank
            # each param should belong to only one rank
            for rank, params in enumerate(params_per_rank):
                # (WGT): Check whether any rank is not assigned params.
                if len(params) != 0:
                    self._param_store.add_fp16_param_list_by_rank_group(rank, group_id, params)
                    for param in params:
                        self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            # move_tensor(params, device='cpu')
            for param in group_params:
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._world_size):
                # (WGT): No flat fp16 buffer is allocated if the process has no parameters.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                    with torch.no_grad():
                        flat_tensor = flatten(tensor_list)
                    flat_tensor = flat_tensor.data.cuda()
                    self._param_store.add_flat_fp16_param_by_rank_group(rank, group_id, flat_tensor)
                    sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            # (WGT): No flat fp32 buffer is allocated if the process has no parameters.
            if self.param_group_has_params[group_id]:
                fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(self._local_rank, group_id)
                fp32_flat_current_rank = fp16_flat_current_rank.float()
                device = "cpu" if self._cpu_offload else get_current_device()
                fp32_flat_current_rank = fp32_flat_current_rank.to(device)
                fp32_flat_current_rank.requires_grad = True
                self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

                # need to replace the params in the `params` field in the optimizer
                # so that when the optimizer calls step(), it only updates the tensors
                # managed by this data parallel rank
                param_group["params"] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)
        assert len(self._fp16_param_groups) != 0

        # (WGT): If a rank is not assigned any arguments, 'has_params' is False.
        self.has_params = sum(self.param_group_has_params) != 0

        # intialize communication stream for
        # communication-compuation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

    def _partition_param_list(self, param_list):
        params_per_rank = [[] for _ in range(self._world_size)]
        numel_per_rank = [0 for _ in range(self._world_size)]
        no_params_ranks = []

        # partititon the parameters in a greedy fashion
        sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)
        for param in sorted_params:
            # allocate this parameter to the rank with
            # the smallest numel for load balancing purpose
            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            numel_per_rank[rank_to_go] += param.numel()

        # (WGT): Check whether any rank is not assigned to parameters.
        for rank, params in enumerate(params_per_rank):
            if len(params) == 0:
                no_params_ranks.append(rank)

        if self._verbose:
            self._logger.info(
                f"Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}",
                ranks=[0],
                parallel_mode=self._dp_parallel_mode,
            )

        return params_per_rank, set(no_params_ranks)

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
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                    if avg_grad is not None and has_inf_or_nan(avg_grad):
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
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
                parameters = self._param_store.get_fp16_params_by_rank_group(group_id=group_id, rank=self._local_rank)
            else:
                # (WGT): In order to prevent collection communication from hanging,
                # we need to involve rank that are not assigned parameters in compute_norm(),
                # so we give them a fp16 vector of 0 values.
                gradients = [self.padding_grad]
                parameters = [self.padding_tensor]

            if self._clip_grad_norm > 0:
                # this norm is before scaling, it will be very large
                norm_group = compute_norm(
                    gradients=gradients,
                    parameters=parameters,
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
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if not self.param_group_has_params[group_id]:
                continue
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

        # (WGT): The following operations are performed only on the rank to which parameters are assigned.
        if len(single_grad_partition_groups) != 0:
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

        # (WGT): For those ranks that are not assigned parameters, we just wait for other ranks
        # to send them updated their own parameters.
        if self.has_params:
            self.optim.step()
            # release the fp32 grad
            release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())
            # update fp16 partition updated by the current rank
            for group_id in range(len(self._fp16_param_groups)):
                if self.param_group_has_params[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                        rank=self._local_rank, group_id=group_id
                    )
                    fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                    fp16_param.data.copy_(fp32_param)

        # broadcast the updated model weights
        handles = []
        for group_id in range(self.num_param_groups):
            for rank in range(self._world_size):
                # (WGT): The following operations are performed only on the rank to which parameters are assigned.
                if rank not in self.param_group_no_params_ranks[group_id]:
                    fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)
                    grank = gpc.get_ranks_in_group(ParallelMode.DATA)[rank]  # need to convert to the global rank
                    assert grank == rank, f"{grank} == {rank}"
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

    def sync_grad(self):
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, _ in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        avg_gradients = self._grad_store._averaged_gradients
        for group_id in range(self.num_param_groups):
            # (WGT): The following operations are performed only on the rank to which parameters are assigned.
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                param_group = self._param_store.get_fp16_params_by_rank_group(self._local_rank, group_id)

                if group_id not in avg_gradients:
                    avg_gradients[group_id] = []

                param_idx = 0
                for param in param_group:
                    if param.grad is not None:
                        if len(avg_gradients[group_id]) == param_idx:
                            avg_gradients[group_id].append(param.grad)
                        else:
                            avg_gradients[group_id][param_idx].add_(param.grad)
                        param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    # TODO 需要加入state_dict方法
    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        # TODO 需要考虑没有 grad_scaler 的情况
        states["grad_scaler"] = grad_scaler

        # 传入的 optimizer 的 state , todo 如果这个optimizer还没跑过就state dict的话，可能会导致其中的一些东西不存在，可能会在之后报错？
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        # 自身管理的 fp32 的权重部分
        flat_fp32_weights = {}
        for group_id, param in self._fp32_flat_param_groups_of_current_rank.items():
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                assert param.grad is None
                flat_fp32_weights[group_id] = param
        states["flat_fp32_weights"] = flat_fp32_weights

        # TODO 应该还需要有一些sanity check的内容

        # TODO 需要考虑出现 dp 数量变化的情况

        return states

    def load_state_dict(self, states):
        # TODO 需要考虑出现 dp 数量变化的情况

        # TODO 需要考虑没有 loss_scaler 的情况
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)

        # load optimizer
        optim_states = states["base_optim_states"]
        self.optim.load_state_dict(optim_states)

        # fp32 权重
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_flat_param_groups_of_current_rank)
        for group_id, param in flat_fp32_weights.items():
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                _param = self._fp32_flat_param_groups_of_current_rank[group_id]
                assert _param.shape == param.shape
                _param.data.copy_(param.data)

        # 需要对model的进行赋值
        for group_id in range(len(self._fp16_param_groups)):
            if self._local_rank not in self.param_group_no_params_ranks[group_id]:
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(
                    rank=self._local_rank, group_id=group_id
                )
                fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
                fp16_param.data.copy_(fp32_param)  # 自动也就改变了 model 那边的值了
