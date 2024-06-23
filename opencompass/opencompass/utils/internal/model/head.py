#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from collections import OrderedDict
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.kernel import LayerNorm
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.utils.checkpointing import (
    broadcast_state_dict,
    gather_tensor_parallel_state_dict,
    partition_tensor_parallel_state_dict,
)
from colossalai.utils.cuda import get_current_device

from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.colossalai_layer._utils import ColossalaiModule
from colossalai.nn.layer.utils import divide, set_tensor_parallel_attribute_by_partition
from colossalai.nn.layer.vanilla import VanillaLayerNorm, VanillaPatchEmbedding
from colossalai.nn.layer.parallel_1d._operation import linear_with_async_comm
from colossalai.nn.layer.parallel_1d._utils import (
    gather_forward_split_backward,
    get_parallel_input,
    reduce_grad,
    reduce_input,
    set_parallel_input,
    split_forward_gather_backward,
)

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


# @LAYERS.register_module
class VocabParallelClassifier1D(ParallelLayer):
    r"""ColLinear with given weight. Classifier of 1D parallelism.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 gather_output: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 grad_scale:float=1  # 是否使用清华GLM-130B中提到的对梯度进行scale的trick，不为1的话就是需要scale
                 ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gather_output = gather_output
        self.parallel_input = get_parallel_input()

        # Divide the weight matrix along the last dimension.
        self.num_classes_per_partition = divide(num_classes, gpc.tensor_parallel_size)

        # Parameters.
        # Initialize weight.
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(torch.empty(self.num_classes_per_partition, self.in_features, **factory_kwargs))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.empty(self.num_classes_per_partition, **factory_kwargs))
        else:
            self.bias = None
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        set_parallel_input(False)
        env.vocab_parallel = True
        self.grad_scale = grad_scale

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def _load_from_global_state_dict(self, state_dict, prefix, *args):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        local_state = partition_tensor_parallel_state_dict(local_state,
                                                           ParallelMode.PARALLEL_1D,
                                                           dims={
                                                               weight_key: 0,
                                                               bias_key: 0
                                                           },
                                                           partition_states={
                                                               weight_key: True,
                                                               bias_key: True
                                                           })
        super()._load_from_global_state_dict(local_state, prefix, *args)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict()
        if self.has_weight:
            local_state[weight_key] = self.weight
        if self.bias is not None:
            local_state[bias_key] = self.bias
        local_state = gather_tensor_parallel_state_dict(local_state,
                                                        ParallelMode.PARALLEL_1D,
                                                        dims={
                                                            weight_key: 0,
                                                            bias_key: 0
                                                        },
                                                        partition_states={
                                                            weight_key: True,
                                                            bias_key: True
                                                        },
                                                        keep_vars=keep_vars)
        destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in VocabParallelClassifier1D forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input_.shape, self.weight.shape, self.weight.shape[-1])
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, ParallelMode.PARALLEL_1D)
        # Matrix multiply.
        weight = self.weight
        if self.grad_scale != 1:
            weight = self.grad_scale*weight + (1-self.grad_scale)*weight.detach()

        output_parallel = F.linear(input_parallel, weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, ParallelMode.PARALLEL_1D, dim=-1)
        else:
            output = output_parallel
        return output
