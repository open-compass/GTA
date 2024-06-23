import os
import sys

import torch
from colossalai import kernel
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import init
from colossalai.nn.layer import VocabParallelEmbedding1D
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from torch import nn

from .fused_gpt1d import DeepFusedGPTTransformerLayer, DeepFusedGPTTransformerLayer1D
from .gpt1d import FusedGPTTransformerLayer1D
from .head import VocabParallelClassifier1D
from .pipeline_utils import partition_uniform_with_embed
from .utils import filter_kwargs

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class GenericPipelineGPT(nn.Module):
    """
    Generic Pipeline GPT Module.

    Args:
        embedding: The embedding layer, default is None.
        blocks: The middle layers, default is None.
        norm: The norm layer, default is None.
        head: The head layer, default is None.

    """

    def __init__(self, embedding=None, blocks=None, norm=None, head=None) -> None:
        super().__init__()
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.head = head
        assert blocks is not None
        if norm is not None or head is not None:
            assert norm is not None and head is not None

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids=input_ids)

        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


class GenericGPT(nn.Module):
    """
    Generic GPT Module.

    Args:
        embedding: The embedding layer, default is None.
        blocks: The middle layers, default is None.
        norm: The norm layer, default is None.
        head: The head layer, default is None.
        embed_grad_scale: The scale factor for embedding gradient, default is 0.1.

    """

    def __init__(self, embedding=None, blocks=None, norm=None, head=None, embed_grad_scale=0.1) -> None:
        super().__init__()
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.head = head
        assert blocks is not None
        if norm is not None or head is not None:
            assert norm is not None and head is not None

        self.embed_grad_scale = embed_grad_scale

    def forward(self, input_ids=None, attention_mask=None):
        # attention_mask: The place where it is 1 needs attention
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.blocks):
            hidden_states, attention_mask = block(hidden_states, attention_mask)

        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


class FusedPipelineGPT1D(GenericPipelineGPT):
    """
    Fused Pipeline 1D GPT Module.

    Args:
        num_layers (int): The number of layers, default is 12.
        hidden_size (int): The size of hidden states, default is 768.
        num_attention_heads (int): The number of attention head, default is 12.
        vocab_size (int): The size of the vocabulary, default is 50304.
        act_func (str): The activation function, default is "gelu".
        mlp_ratio (int): The MLP ratio, default is 4.0.
        attn_drop_rate (float): The attention dropout rate, default is 0.0.
        drop_rate (float): The dropout rate, default is 0.0.
        dtype (torch.dtype): The data type, default is torch.float.
        checkpoint (bool): If save checkpoint, default is False.
        layer_norm_epsilon (float): The epsilon of layer normalization, default is 1e-5.
        apply_post_layer_norm (bool): If apply post layer normalization, default is False.
        first (bool): If include the first layer, default is False.
        last (bool): If include the last layer, default is False.
        embed_split_hidden (bool): If embedding split hidden dimension, default is False.
        embed_grad_scale (float): The scale factor of embedding gradient, default is 0.1.
        parallel_output (bool): If parallel output, default is True.

    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        act_func: str = "gelu",
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
        first: bool = False,
        last: bool = False,
        embed_split_hidden=False,
        embed_grad_scale=0.1,
        parallel_output=True,
    ):
        embedding = None
        norm = None
        head = None
        embed_cls = VocabParallelEmbedding1D
        head_cls = VocabParallelClassifier1D
        if embed_split_hidden:
            raise RuntimeError("Currently not support split in hidden dimension")

        if first:
            embedding = embed_cls(vocab_size, hidden_size, dtype=dtype, weight_initializer=init.normal_(std=0.006))
        self.embed_grad_scale = embed_grad_scale
        blocks = nn.ModuleList(
            [
                FusedGPTTransformerLayer1D(
                    hidden_size,
                    num_attention_heads,
                    act_func=act_func,
                    mlp_ratio=mlp_ratio,
                    attention_dropout_prob=attn_drop_rate,
                    hidden_dropout_prob=drop_rate,
                    dtype=dtype,
                    checkpoint=checkpoint,
                    max_position_embeddings=1024,
                    layer_norm_epsilon=layer_norm_epsilon,
                    apply_post_layer_norm=apply_post_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        if last:
            norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            head = head_cls(
                in_features=hidden_size,
                num_classes=vocab_size,
                dtype=dtype,
                gather_output=not parallel_output,
                bias=False,
                grad_scale=self.embed_grad_scale,
                weight_initializer=init.normal_(std=0.006),
            )
        super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        # attention_mask: The place where it is 1 needs attention
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


class DeepFusedPipelineGPT1D(GenericPipelineGPT):
    """
    Deep Fused Pipeline 1D GPT Module.

    Args:
        num_layers (int): The number of layers, default is 12.
        hidden_size (int): The size of hidden states, default is 768.
        num_attention_heads (int): The number of attention head, default is 12.
        vocab_size (int): The size of the vocabulary, default is 50304.
        act_func (str): The activation function, default is "gelu".
        mlp_ratio (int): The MLP ratio, default is 4.0.
        attn_drop_rate (float): The attention dropout rate, default is 0.0.
        drop_rate (float): The dropout rate, default is 0.0.
        dtype (torch.dtype): The data type, default is torch.float.
        checkpoint (bool): If save checkpoint, default is False.
        layer_norm_epsilon (float): The epsilon of layer normalization, default is 1e-5.
        apply_post_layer_norm (bool): If apply post layer normalization, default is False.
        first (bool): If include the first layer, default is False.
        last (bool): If include the last layer, default is False.
        embed_split_hidden (bool): If embedding split hidden dimension, default is False.
        embed_grad_scale (float): The scale factor of embedding gradient, default is 0.1.
        parallel_output (bool): If parallel output, default is True.

    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        act_func: str = "gelu",
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
        first: bool = False,
        last: bool = False,
        embed_split_hidden=False,
        embed_grad_scale=0.1,
        parallel_output=True,
    ):
        embedding = None
        norm = None
        head = None
        embed_cls = VocabParallelEmbedding1D
        head_cls = VocabParallelClassifier1D
        if embed_split_hidden:
            raise RuntimeError("Currently not support split in hidden dimension")

        if first:
            embedding = embed_cls(vocab_size, hidden_size, dtype=dtype, weight_initializer=init.normal_(std=0.006))
        self.embed_grad_scale = embed_grad_scale
        blocks = nn.ModuleList(
            [
                DeepFusedGPTTransformerLayer1D(
                    hidden_size,
                    num_attention_heads,
                    act_func=act_func,
                    mlp_ratio=mlp_ratio,
                    attention_dropout_prob=attn_drop_rate,
                    hidden_dropout_prob=drop_rate,
                    dtype=dtype,
                    checkpoint=checkpoint,
                    layer_norm_epsilon=layer_norm_epsilon,
                    apply_post_layer_norm=apply_post_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        if last:
            norm = kernel.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            head = head_cls(
                in_features=hidden_size,
                num_classes=vocab_size,
                dtype=dtype,
                gather_output=not parallel_output,
                bias=False,
                grad_scale=self.embed_grad_scale,
                weight_initializer=init.normal_(std=0.006),
            )
        super().__init__(embedding=embedding, blocks=blocks, norm=norm, head=head)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        # attention_mask: The place where it is 1 needs attention
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.blocks):
            hidden_states, attention_mask = block(hidden_states, attention_mask)

        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


def _build_generic_gpt_pipeline_1d(module_cls, num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    if pipeline_size > 1:
        wrapper = PipelineSharedModuleWrapper(
            [0, pipeline_size - 1]
        )  # Synchronize the parameters and set the pipeline_shared_module_pg at the same time,
        # so that the PipelineSharedModuleGradientHandler will be set when the Engine is initialized.
    else:
        wrapper = None

    all_parts = partition_uniform_with_embed(num_layers, pipeline_size, num_chunks)
    # all_parts = partition_without_last_head(num_layers, pipeline_size, num_chunks)

    from utils.logger import LLM_LOGGER as logger

    logger.info(f"The layer sharding is {all_parts}.", ranks=[0])

    parts = all_parts[pipeline_rank]

    models = []
    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If the last layer has no content, assign the last layer to him
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0

        chunk = module_cls(**filter_kwargs(module_cls.__init__, kwargs)).to(device)

        if wrapper is not None:
            if start == 0:
                wrapper.register_module(chunk.embedding)
            elif end == num_layers:
                wrapper.register_module(chunk.head)
        models.append(chunk)

    if len(parts) == 0:  # For the special case of taking out the head alone to make a layer
        kwargs["num_layers"] = 0
        kwargs["first"] = False
        kwargs["last"] = True
        chunk = module_cls(**filter_kwargs(module_cls.__init__, kwargs)).to(device)
        if wrapper is not None:
            wrapper.register_module(chunk.head)
        models.append(chunk)

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    numel = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()

    return model


def _build_gpt_pipeline_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    if hasattr(gpc.config, "use_deep_fused") and not gpc.config.get("use_deep_fused", False):
        model = FusedPipelineGPT1D
        raise RuntimeError("not causal masking")
    else:
        model = DeepFusedPipelineGPT1D
    return _build_generic_gpt_pipeline_1d(model, num_layers, num_chunks, device, **kwargs)


def GPT2_exlarge_pipeline_1D(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=0.1,
    parallel_output=True,
    num_attention_heads=32,
    mlp_ratio=4.0,
    apply_post_layer_norm=False,
):
    cfg = dict(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        checkpoint=checkpoint,
        dtype=dtype,
        embed_split_hidden=embed_split_hidden,
        vocab_size=vocab_size,
        embed_grad_scale=embed_grad_scale,
        parallel_output=parallel_output,
        mlp_ratio=mlp_ratio,
        apply_post_layer_norm=apply_post_layer_norm,
    )
    return _build_gpt_pipeline_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)


def GPT2_exlarge(
    dtype=torch.float,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=0.1,
    num_attention_heads=32,
    mlp_ratio=4.0,
):
    blocks = nn.ModuleList(
        [
            DeepFusedGPTTransformerLayer(
                hidden_size,
                num_attention_heads,
                act_func="gelu",
                mlp_ratio=mlp_ratio,
                attention_dropout_prob=0,
                hidden_dropout_prob=0,
                dtype=dtype,
                checkpoint=False,
                layer_norm_epsilon=1e-5,
                apply_post_layer_norm=False,
            )
            for _ in range(num_layers)
        ]
    )
    embed = nn.Embedding(vocab_size, hidden_size)
    norm = nn.LayerNorm(hidden_size)
    head = nn.Linear(hidden_size, vocab_size, bias=False)
    assert head.weight.shape == embed.weight.shape
    head.weight.data = embed.weight.data
    return GenericGPT(embedding=embed, blocks=blocks, norm=norm, head=head, embed_grad_scale=embed_grad_scale)
