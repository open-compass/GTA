import math
import os
import sys
from typing import Optional

import torch
from colossalai.constants import IS_TENSOR_PARALLEL
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import init
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward
from colossalai.utils.activation_checkpoint import checkpoint
from flash_attn.modules.embedding import ParallelGPT2Embeddings
from flash_attn.modules.mlp import ParallelFusedMLP
from flash_attn.ops.layer_norm import dropout_add_layer_norm
from torch import nn

from ..model_utils import Embedding1D
from ..pipeline_utils import partition_uniform_with_embed2
from ..utils import (
    filter_kwargs,
    scaled_init_method_normal,
    xavier_normal_tensor_parallel,
)
from .packed_module import OneDParallelMHA, ScaleColumnParallelLinear

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class PackedFlashTransformerLayer1D(nn.Module):
    """
    1D Packed Flash Transformer Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        apply_post_layer_norm (bool): Whether to use postnorm(True) or prenorm(False). False by default.
        fused_dropout_add_ln (bool): Whether to use dropout_add_layer_norm. True by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        no_bias (bool): Whether to use bias in multihead attention and feedforward. False by default
        deepnorm (bool): Whether to use deepnorm or not. False by default
        total_layers (int): The total numer of layers. Needed when using deepnorm. -1 by default.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        checkpoint: bool = True,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        no_bias: bool = False,
        deepnorm: bool = False,
        total_layers: int = -1,
        use_scaled_init: bool = True,
    ):
        super().__init__()
        self.prenorm = not apply_post_layer_norm
        self.fused_dropout_add_ln = fused_dropout_add_ln

        if deepnorm:
            assert total_layers != -1, "When using deepnorm must pass in the total numer of layers"
            self.deepnorm_alpha = (2 * total_layers) ** 0.5  # refer to the code of GLM-130B
            self.deepnorm_beta = (
                2 * total_layers
            ) ** -0.5  # from LargeScale.megatron.model.utils:get_deepnorm_coefficients
            # refer to: https://kexue.fm/archives/8978
        else:
            self.deepnorm_alpha = 1.0
            self.deepnorm_beta = 1.0

        head_dim = hidden_size // num_attention_heads
        self.mixer = OneDParallelMHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            process_group=gpc.get_group(ParallelMode.PARALLEL_1D),
            dropout=attn_drop_rate,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            use_flash_attn=True,
            checkpointing=False,
            sequence_parallel=False,
            device=device,
            dtype=dtype,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.mlp = ParallelFusedMLP(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            activation="gelu_approx",
            process_group=gpc.get_group(ParallelMode.PARALLEL_1D),
            bias1=not no_bias,
            bias2=not no_bias,
            sequence_parallel=False,
            checkpoint_lvl=0,
            heuristic="auto",
            device=device,
            dtype=dtype,
        )
        # need to assign tp attribute so that colossalai know it is tensor parallel module

        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            module = getattr(self, "mlp")
            for name in ["fc1", "fc2"]:
                for param in getattr(module, name).parameters():
                    setattr(param, IS_TENSOR_PARALLEL, True)

        self.dropout2 = nn.Dropout(drop_rate)

        self.use_scaled_init = use_scaled_init
        if gpc.get_global_rank() == 0:
            print(
                f"use_scaled_init: {use_scaled_init}",
                flush=True,
            )

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        if no_bias:
            self.norm2.bias = None
            self.norm1.bias = None
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, "dropout_add_ln is not installed"
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.deepnorm = deepnorm
        if deepnorm:
            self.deepnorm_reset_parameters()
        else:
            self.reset_parameters()

        self.checkpoint = checkpoint

    def deepnorm_reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    wq, wk, wv = param.data.chunk(3, dim=0)
                    xavier_normal_tensor_parallel(
                        wq, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0
                    )
                    xavier_normal_tensor_parallel(
                        wk, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=1.0
                    )
                    xavier_normal_tensor_parallel(
                        wv, partition_dim=0, tp_degree=gpc.get_world_size(ParallelMode.TENSOR), gain=self.deepnorm_beta
                    )
                else:  # dense
                    xavier_normal_tensor_parallel(
                        param.data,
                        partition_dim=1,
                        tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                        gain=self.deepnorm_beta,
                    )

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                else:
                    if "fc1" in name:
                        xavier_normal_tensor_parallel(
                            param.data,
                            partition_dim=0,
                            tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                            gain=self.deepnorm_beta,
                        )
                    else:
                        xavier_normal_tensor_parallel(
                            param.data,
                            partition_dim=1,
                            tp_degree=gpc.get_world_size(ParallelMode.TENSOR),
                            gain=self.deepnorm_beta,
                        )

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "Wqkv" in name:
                    init.normal_(std=0.006)(param.data)
                else:
                    if self.use_scaled_init:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        init.normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                else:
                    if self.use_scaled_init:
                        if "fc1" in name:
                            init.normal_(std=0.006)(param.data)
                        else:
                            scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        if "fc1" in name:
                            init.normal_(std=0.006)(param.data)
                        else:
                            init.normal_(std=0.0015)(param.data)

    def forward(
        self, hidden_states, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
        if self.checkpoint and self.training:
            return checkpoint(
                self._forward, False, hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen
            )
        else:
            return self._forward(hidden_states, residual, cu_seqlens, indexes, inference_params, max_seqlen)

    def _forward(
        self, hidden_states=None, residual=None, cu_seqlens=None, indexes=None, inference_params=None, max_seqlen=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.dropout1(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                rowscale1 = None
                hidden_states, residual = dropout_add_layer_norm(
                    hidden_states,
                    residual,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                )
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)

            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.dropout2(hidden_states)
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    rowscale2 = None
                    hidden_states, residual = dropout_add_layer_norm(
                        hidden_states,
                        residual,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.dropout2.p if self.training else 0.0,
                        self.norm2.eps,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                    )

                hidden_states = self.mlp(hidden_states)

            return hidden_states + residual
        else:
            assert residual is None
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": (cu_seqlens[1:] - cu_seqlens[:-1]).max().item() if cu_seqlens is not None else None,
                "indexes": indexes,
                "inference_params": inference_params,
            }
            mixer_out = self.mixer(hidden_states, **mixer_kwargs)
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1(self.dropout1(mixer_out) + hidden_states * self.deepnorm_alpha).to(
                    dtype=self.norm1.weight.dtype
                )
            else:
                rowscale1 = None
                hidden_states = dropout_add_layer_norm(
                    mixer_out,
                    hidden_states * self.deepnorm_alpha,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.dropout1.p if self.training else 0.0,
                    self.norm1.eps,
                    rowscale=rowscale1,
                    prenorm=False,
                    residual_in_fp32=False,
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.dropout2(mlp_out)) + hidden_states * self.deepnorm_alpha).to(
                        dtype=self.norm2.weight.dtype
                    )
                else:
                    rowscale2 = None
                    hidden_states = dropout_add_layer_norm(
                        mlp_out,
                        hidden_states * self.deepnorm_alpha,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.dropout2.p if self.training else 0.0,
                        self.norm2.eps,
                        rowscale=rowscale2,
                        prenorm=False,
                        residual_in_fp32=False,
                    )
            return hidden_states


class PackedFlashFusedPipelineGPT1D(nn.Module):
    """
    1D Packed Flash Fused Pipeline GPT.

    Args:
        num_layers (int): The number of layer. 12 by default
        hidden_size (int): The size of hidden state. 768 by default
        num_attention_heads (int): The number of attention head. 12 by default
        vocab_size (int): The size of vocabulary. 50304 by default
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default
        dtype (torch.dtype): The type of data. torch.float by default
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        checkpoint_fraction (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 1.0 by default
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-6 by default.
        apply_post_layer_norm (bool): Whether to use postnorm(True) or prenorm(False). False by default.
        first (bool): Whether input embedding layer or not. False by default
        last (bool): Whether output embedding layer or not. False by default
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    True by default
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default
        device (Optional[Union[str, torch.device]]): The device will be used. None by default
        no_bias (bool): Whether the bias is needed for linears. False by default
        deepnorm (bool): Whether to use deepnorm or not. False by default
        total_layers (int): The total number of layers. Will be used in deepnorm. -1 by default
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
    """

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        checkpoint: bool = False,
        checkpoint_fraction: float = 1.0,
        layer_norm_epsilon: float = 1e-5,
        apply_post_layer_norm: bool = False,
        first: bool = False,
        last: bool = False,
        embed_split_hidden: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        no_bias: bool = False,
        deepnorm: bool = False,
        total_layers: int = -1,
        residual_in_fp32: bool = False,
        use_scaled_init: bool = True,
    ):
        super().__init__()

        if checkpoint_fraction <= 0:
            checkpoint = False
        if not checkpoint:
            checkpoint_fraction = 0
        checkpoint_layer_num = num_layers * checkpoint_fraction
        if embed_split_hidden:
            embed_cls = Embedding1D
        else:
            embed_cls = ParallelGPT2Embeddings  # split in vocab
        head_cls = ScaleColumnParallelLinear
        if first:
            if embed_split_hidden:
                self.embedding = embed_cls(num_embeddings=vocab_size, embedding_dim=hidden_size)
            else:
                self.embedding = embed_cls(
                    embed_dim=hidden_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=-1,
                    process_group=gpc.get_group(ParallelMode.TENSOR),
                    padding_idx=None,
                    sequence_parallel=False,
                    device=device,
                    dtype=dtype,
                )
            for _, param in self.embedding.named_parameters():
                init.normal_(std=0.0052)(param)
        self.embed_grad_scale = embed_grad_scale
        self.blocks = nn.ModuleList(
            [
                PackedFlashTransformerLayer1D(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    no_bias=no_bias,
                    deepnorm=deepnorm,
                    total_layers=total_layers,  # For deepnorm
                    use_scaled_init=use_scaled_init,
                )
                for lid in range(num_layers)
            ]
        )
        if last:
            if not apply_post_layer_norm:
                self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
                if no_bias:
                    self.norm.bias = None
            self.head = head_cls(
                in_features=hidden_size,
                out_features=vocab_size,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                bias=False,
                sequence_parallel=False,
                device=device,
                dtype=dtype,
                weight_scale=embed_grad_scale,
            )
            for _, param in self.head.named_parameters():
                init.normal_(std=0.0052)(param)
        self.parallel_output = parallel_output
        self.apply_post_layer_norm = apply_post_layer_norm

        # need to assign tp attribute so that colossalai know it is tensor parallel module
        if gpc.get_world_size(ParallelMode.TENSOR) > 1:
            for name in ["embedding", "head"]:
                if hasattr(self, name):
                    for param in getattr(self, name).parameters():
                        setattr(param, IS_TENSOR_PARALLEL, True)

    def forward(self, hidden_states=None, cu_seqlens=None, input_ids=None, indexes=None, inference_params=None):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "embedding"):
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )
        if isinstance(cu_seqlens, list):
            assert len(cu_seqlens) == 1
            cu_seqlens = cu_seqlens[0].to(hidden_states.device)
            # the batch dimension with a size of 1 should be directly squeezed off.
        if cu_seqlens is not None:
            hidden_states = hidden_states.squeeze(0)  # If cu_seqlens is passed in，it indicated a packed state，
            cu_seqlens = cu_seqlens.squeeze(0)

        if indexes is not None:
            assert len(indexes) == 1
            indexes = indexes[
                0
            ]  # The indexes are used to indicate the actual position IDs of each token in the packed input.
        max_seqlen = None
        if cu_seqlens is not None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                indexes=indexes,
                inference_params=inference_params,
                max_seqlen=max_seqlen,
            )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        if not self.parallel_output:
            hidden_states = gather_forward_split_backward(hidden_states, ParallelMode.PARALLEL_1D, dim=-1)
        return hidden_states


def _build_generic_gpt_pipeline_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pipeline_size = 1
        pipeline_rank = 0

    # if pipeline_size > 1:
    #     wrapper = PipelineSharedModuleWrapper(
    #         [0, pipeline_size - 1]
    #     )  # synchronize the parameters and set pipeline_shared_module_pg at the same time
    #     # This will set the PipelineSharedModuleGradientHandler during Engine initialization
    # else:
    #     wrapper = None

    all_parts = partition_uniform_with_embed2(num_layers, pipeline_size, num_chunks)
    if gpc.is_initialized(ParallelMode.GLOBAL):
        from utils.logger import LLM_LOGGER as logger

        logger.info(f"The layer sharding is {all_parts}.", ranks=[0])
    parts = all_parts[pipeline_rank]

    models = []
    kwargs["total_layers"] = num_layers

    if kwargs["checkpoint"] is False:
        # if pipeline_rank == 0:
        #     kwargs["checkpoint"] = True
        #     kwargs["checkpoint_fraction"] = 0.2
        # elif pipeline_rank == 1:
        #     kwargs["checkpoint"] = True
        #     kwargs["checkpoint_fraction"] = 0.1
        # else:
        #     kwargs["checkpoint"] = True
        #     kwargs["checkpoint_fraction"] = 0.01
        pass
    else:
        kwargs["checkpoint_fraction"] = 1.0

    for start, end in parts:
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        kwargs["last"] = (
            end == num_layers and len(all_parts[-1]) != 0
        )  # If there is no content in the final layer, assign the last layer.
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        chunk = PackedFlashFusedPipelineGPT1D(**filter_kwargs(PackedFlashFusedPipelineGPT1D.__init__, kwargs)).to(
            device
        )

        # if wrapper is not None:
        #     if start == 0:
        #         wrapper.register_module(chunk.embedding)
        #     elif end == num_layers:
        #         wrapper.register_module(chunk.head)
        models.append(chunk)
    torch.distributed.barrier()

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model


def Packed_Flash_GPT2_exlarge_pipeline_1D(
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
    no_bias=False,
    deepnorm=False,
    residual_in_fp32=False,
    drop_rate=0,
    attn_drop_rate=0,
    model_type="flash",
    use_scaled_init: bool = True,
):
    assert model_type == "flash", "Only flash model type is allowed"
    # residual_in_fp32 cannot be used temporarily because this parameter requires inconsistent data types to
    # be passed between pipelines, which requires significant modifications to colossalai.
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
        no_bias=no_bias,
        deepnorm=deepnorm,
        residual_in_fp32=residual_in_fp32,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        use_scaled_init=use_scaled_init,
    )
    return _build_generic_gpt_pipeline_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
