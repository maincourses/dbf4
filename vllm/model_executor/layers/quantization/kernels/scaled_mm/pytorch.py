# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from packaging import version

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
try:
    from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
    from lmslim.layers.gemm.fp8_utils import triton_scaled_mm_fp8
except ImportError:
    print("INFO: Please updata lmslim if you want to use fp8_utils.\n")
from vllm import envs

class TorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    Base class for FP8 linear kernels using Torch.
    Each subclass represents a kernel variant for
    specific device capabilities and torch versions.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_cpu()):
            return False, "requires ROCm, CUDA or CPU."

        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."

        return True, None

    def get_output_padding(self) -> int | None:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        # We also don't pad when using torch.compile,
        # as it breaks with dynamic shapes.
        #
        # The perf gain is still relevant as of 16/1/2026
        # torch version == 2.9.0. More details in the link below:
        # https://github.com/vllm-project/vllm/issues/32269
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
        return 17 if pad_output else None


class PerTensorTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return False, "requires per tensor activation and weight scales."
        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = torch._scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
        # A fix for discrepancy in scaled_mm which returns tuple
        # for torch < 2.5 and a single value in torch >= 2.5
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)


class RowWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        from vllm.platforms.rocm import on_mi3xx

        if not on_mi3xx():
            return False, "requires MI3xx."

        if compute_capability is not None and compute_capability < 94:
            return False, "requires compute capability 94 and above."

        if not version.parse(torch.__version__) >= version.parse("2.7"):
            return False, "requires pytorch version >=2.7."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if c.out_dtype == torch.float16:
            # hipblaslt rowwise _scaled_mm only supports BFloat16
            return False, "supports BFloat16 output data type only."

        if per_tensor_activation_scales or per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        #  Note:
        #  For now it has only been validated on ROCm platform.
        #  fp8 rowwise scaling in torch._scaled_mm is introduced in
        #  https://github.com/pytorch/pytorch/pull/144432 using
        #  hipBLASLt and ROCm 6.3, which only exists in torch 2.7 and above.
        #
        #  For CUDA platform please validate if the torch._scaled_mm supports
        #  rowwise scaled GEMM before using it

        # Fused GEMM_DQ Rowwise GEMM
        output = torch._scaled_mm(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.t(),
            bias=bias,
        )

        return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)


class ChannelWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        m = A.shape[0]
        k = A.shape[1]
        n = B.shape[0]

        if envs.VLLM_W8A8_BACKEND == 3:
            _, output = hipblaslt_w8a8_channelwise_gemm(
                a=A,
                b=B,
                scale_a=As,
                scale_b=Bs,
                m=m,
                n=n,
                k=k,
                transpose_flag="NT",
                out_dtype=out_dtype,
                bias=bias,
            )
            return output.view(m, n)
        else:
            output = triton_scaled_mm_fp8(
                A,
                B,
                scale_a=As,
                scale_b=Bs,
                out_dtype=out_dtype,
                bias=bias,
            )
            return output.view(*output_shape)
