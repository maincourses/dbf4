# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy
import os

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    init_int8_linear_kernel,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_int8_linear

logger = init_logger(__name__)


class CompressedTensorsW8A8Int8(CompressedTensorsScheme):
    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.w8a8_strategy = envs.VLLM_W8A8_BACKEND
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.logical_widths = output_partition_sizes

        self.kernel = init_int8_linear_kernel(
            is_channelwise=(self.strategy == QuantizationStrategy.CHANNEL),
            is_static_input_scheme=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
            module_name=self.__class__.__name__,
        )

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        input_zero_point = None
        input_scale = None
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )

        layer.register_parameter("input_zero_point", input_zero_point)
        layer.register_parameter("input_scale", input_scale)
        if not hasattr(layer, "azp_adj"):
            layer.register_parameter("azp_adj", None)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None, input_quant_args: list[torch.Tensor] | None = None, silu_quant_args: list[torch.Tensor] | None = None, **_
    ) -> torch.Tensor:
        # return self.kernel.apply_weights(layer, x, bias)
        return apply_int8_linear(input=x,
                                 weight=layer.weight,
                                 weight_scale=layer.weight_scale,
                                 input_scale=layer.input_scale,
                                 input_zero_point=layer.input_zero_point,
                                 azp_adj=layer.azp_adj,
                                 bias=bias,
                                 w8a8_strategy=self.w8a8_strategy,
                                 input_quant_args=input_quant_args,
                                 silu_quant_args=silu_quant_args)
