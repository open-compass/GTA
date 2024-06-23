#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
from colossalai import kernel
from colossalai import nn as col_nn
from colossalai.core import global_context as gpc
from colossalai.kernel.jit import (
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
)
from colossalai.nn import init
from colossalai.nn.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.utils import ACT2FN, divide
from colossalai.utils.activation_checkpoint import checkpoint
from torch import Tensor, nn

from .common import RotaryEmbedding
from .scale_softmax import AttnMaskType, FusedScaleMaskSoftmax
from .utils import apply_rotary_pos_emb

__all__ = ["DeepFusedGPTTransformerLayer1D"]


class DeepFusedGPTTransformerLayer1D(ParallelLayer):
    """
    1D Deep Fused GPT Transformer Layer.

    Args:
        hidden_size (int): The size of hidden states.
        num_attention_heads (int): The number of attention head.
        act_func (str): The activation function, default is "gelu".
        mlp_ratio (int): The MLP ratio, default is 4.0.
        attention_dropout_prob (float): The attention dropout rate, default is 0.0.
        hidden_dropout_prob (float): The hidden layer dropout rate, default is 0.0.
        dtype (torch.dtype): The data type.
        checkpoint (bool): If save checkpoint, default is False.
        layer_norm_epsilon (float): The epsilon of layer normalization, default is 1e-5.
        apply_post_layer_norm (bool): If apply post layer normalization, default is False.

    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        act_func: str = "gelu",
        mlp_ratio: float = 4.0,
        attention_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        dtype=None,
        checkpoint: bool = False,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
    ):
        super().__init__()

        self.checkpoint = checkpoint
        self.dtype = dtype
        self.norm1 = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.apply_post_layer_norm = apply_post_layer_norm
        self.in_features = hidden_size

        # attention part
        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)
        self.query_key_value = Linear1D_Col(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
            weight_initializer=init.normal_(std=0.006),
            bias_initializer=init.zeros_(),
        )
        self.attention_dropout = col_nn.Dropout(attention_dropout_prob)
        self.dense = Linear1D_Row(
            hidden_size,
            hidden_size,
            dtype=dtype,
            parallel_input=True,
            weight_initializer=init.normal_(std=0.006 / math.sqrt(2 * 8)),
            bias_initializer=init.zeros_(),
            skip_bias_add=True,
        )
        self.dropout = hidden_dropout_prob
        self.rotary_emb = RotaryEmbedding(self.attention_head_size)
        self.register_buffer("pos_emb", None, persistent=False)
        self.softmax = FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=None,
            softmax_in_fp32=True,
            scale=1 / math.sqrt(self.attention_head_size),
        )

        self.norm2 = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        # MLP part

        self.mlp_ratio = mlp_ratio
        self.act = ACT2FN[act_func]
        # Project to mlp_ratio * h.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            dtype=dtype,
            gather_output=False,
            skip_bias_add=False,
            weight_initializer=init.normal_(std=0.006),
            bias_initializer=init.zeros_(),
        )
        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            dtype=dtype,
            parallel_input=True,
            weight_initializer=init.normal_(std=0.006 / math.sqrt(2 * 8)),
            bias_initializer=init.zeros_(),
            skip_bias_add=True,
        )
        self.train()

    def train(self, mode: bool = True):
        if mode:
            self.bias_add_dropout = bias_dropout_add_fused_train
        else:
            self.bias_add_dropout = bias_dropout_add_fused_inference
        return super().train(mode)

    def eval(self):
        self.bias_add_dropout = bias_dropout_add_fused_inference
        return super().eval()

    def softmax_forward(self, attention_scores, attention_mask):
        return self.softmax(attention_scores, attention_mask)

    def get_rotary_embedding(self, seq, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq:
            return self.pos_emb[:seq]

        pos_emb = self.rotary_emb(seq, device=device)
        delattr(self, "pos_emb")
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def _forward(self, hidden_states, attention_mask) -> Tensor:
        if not self.apply_post_layer_norm:
            residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states
        query_key_value = self.query_key_value(hidden_states)  # bsz x max_len x 3d

        new_qkv_shape = query_key_value.shape[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.attention_head_size,
        )  # bsz x max_len x n_head' x  3d'
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))  # bsz x n_head x length x hsz
        query_layer, key_layer, value_layer = torch.chunk(query_key_value, 3, dim=-1)

        # apply posititon embedding
        positions = self.get_rotary_embedding(query_key_value.shape[2], query_key_value.device)
        query_layer, key_layer = map(lambda t: apply_rotary_pos_emb(positions, t), (query_layer, key_layer))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = self.softmax_forward(attention_scores, attention_mask)

        attention_scores = attention_scores.type(value_layer.dtype)

        attention_probs = self.attention_dropout(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output, bias = self.dense(context_layer)

        hidden_states = self.bias_add_dropout(output, bias, residual, self.dropout)

        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states

        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = self.act(intermediate_output)

        output, bias = self.dense_2(intermediate_output)
        hidden_states = self.bias_add_dropout(output, bias, residual, self.dropout)

        output = (hidden_states, attention_mask)
        return output

    def forward(self, hidden_states, attention_mask):
        if self.checkpoint and self.training:
            return checkpoint(self._forward, False, hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)


class DeepFusedGPTTransformerLayer(nn.Module):
    """
    Deep Fused GPT Transformer Layer.

    Args:
        hidden_size (int): The size of hidden states.
        num_attention_heads (int): The number of attention head.
        act_func (str): The activation function, default is "gelu".
        mlp_ratio (int): The MLP ratio, default is 4.0.
        attention_dropout_prob (float): The attention dropout rate, default is 0.0.
        hidden_dropout_prob (float): The hidden layer dropout rate, default is 0.0.
        dtype (torch.dtype): The data type.
        checkpoint (bool): If save checkpoint, default is False.
        layer_norm_epsilon (float): The epsilon of layer normalization, default is 1e-5.
        apply_post_layer_norm (bool): If apply post layer normalization, default is False.

    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        act_func: str = "gelu",
        mlp_ratio: float = 4.0,
        attention_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        dtype=None,
        checkpoint: bool = False,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.apply_post_layer_norm = apply_post_layer_norm
        self.in_features = hidden_size

        # attention part
        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.query_key_value = nn.Linear(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
        )
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.dense = nn.Linear(
            hidden_size,
            hidden_size,
            dtype=dtype,
        )
        self.dropout = hidden_dropout_prob
        self.rotary_emb = RotaryEmbedding(self.attention_head_size)
        self.register_buffer("pos_emb", None, persistent=False)
        self.softmax = FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            mask_func=None,
            softmax_in_fp32=True,
            scale=1 / math.sqrt(self.attention_head_size),
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        # MLP part

        self.mlp_ratio = mlp_ratio
        self.act = ACT2FN[act_func]
        # Project to mlp_ratio * h.
        self.dense_1 = nn.Linear(self.in_features, int(self.mlp_ratio * self.in_features), dtype=dtype)
        # Project back to h.
        self.dense_2 = nn.Linear(int(self.mlp_ratio * self.in_features), self.in_features, dtype=dtype)

    def train(self, mode: bool = True):
        return super().train(mode)

    def eval(self):
        return super().eval()

    def softmax_forward(self, attention_scores, attention_mask):
        return self.softmax(attention_scores, attention_mask)

    def get_rotary_embedding(self, seq, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq:
            return self.pos_emb[:seq]

        pos_emb = self.rotary_emb(seq, device=device)
        delattr(self, "pos_emb")
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def _forward(self, hidden_states, attention_mask) -> Tensor:
        if not self.apply_post_layer_norm:
            residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states

        query_key_value = self.query_key_value(hidden_states)  # bsz x max_len x 3d

        new_qkv_shape = query_key_value.shape[:-1] + (
            -1,
            3 * self.attention_head_size,
        )  # bsz x max_len x n_head' x  3d'
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))  # bsz x n_head x length x hsz
        query_layer, key_layer, value_layer = torch.chunk(query_key_value, 3, dim=-1)

        # apply posititon embedding
        positions = self.get_rotary_embedding(query_key_value.shape[2], query_key_value.device)
        query_layer, key_layer = map(lambda t: apply_rotary_pos_emb(positions, t), (query_layer, key_layer))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = self.softmax_forward(attention_scores, attention_mask)

        attention_scores = attention_scores.type(value_layer.dtype)

        attention_probs = self.attention_dropout(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        attention_output = self.dense(context_layer)

        hidden_states = residual + attention_output

        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states

        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = self.act(intermediate_output)

        output = self.dense_2(intermediate_output)
        hidden_states = residual + output

        output = (hidden_states, attention_mask)
        return output

    def forward(self, hidden_states, attention_mask):
        if self.checkpoint and self.training:
            return checkpoint(self._forward, False, hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)
