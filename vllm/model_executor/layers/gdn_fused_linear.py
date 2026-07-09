# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime fused GDN projection helpers for Qwen3.5 on ROCm."""

import torch

from vllm import envs
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.utils.torch_utils import direct_register_custom_op

_QWEN35_QKVZ_SHAPE = (16384, 5120)
_QWEN35_BA_SHAPE = (96, 5120)
_QWEN35_ROWS_PER_BLOCK = 2
_LLMM1_TWO_WEIGHTS_OP = None
_LLMM1_TWO_WEIGHTS_PROBED = False


def _get_llmm1_two_weights_op():
    global _LLMM1_TWO_WEIGHTS_OP, _LLMM1_TWO_WEIGHTS_PROBED
    if _LLMM1_TWO_WEIGHTS_PROBED:
        return _LLMM1_TWO_WEIGHTS_OP
    _LLMM1_TWO_WEIGHTS_PROBED = True
    try:
        import vllm._rocm_C  # noqa: F401

        _LLMM1_TWO_WEIGHTS_OP = torch.ops._rocm_C.LLMM1TwoWeights
    except (AttributeError, ImportError, RuntimeError):
        _LLMM1_TWO_WEIGHTS_OP = None
    return _LLMM1_TWO_WEIGHTS_OP


def gdn_qkvz_ba_fused_linear_impl(
    x: torch.Tensor,
    weight_qkvz: torch.Tensor,
    weight_ba: torch.Tensor,
    rows_per_block: int,
) -> torch.Tensor:
    x_view = x.reshape(-1, x.size(-1))
    if not x_view.is_contiguous():
        x_view = x_view.contiguous()
    out = torch.ops._rocm_C.LLMM1TwoWeights(
        weight_qkvz, weight_ba, x_view, rows_per_block
    )
    return out.reshape(*x.shape[:-1], weight_qkvz.shape[0] + weight_ba.shape[0])


def gdn_qkvz_ba_projection_impl(
    x: torch.Tensor,
    weight_qkvz: torch.Tensor,
    weight_ba: torch.Tensor,
    rows_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_view = x.reshape(-1, x.size(-1))
    if x_view.shape[0] == 1:
        if not x_view.is_contiguous():
            x_view = x_view.contiguous()
        fused = torch.ops._rocm_C.LLMM1TwoWeights(
            weight_qkvz, weight_ba, x_view, rows_per_block
        )
        qkvz, ba = fused.split([weight_qkvz.shape[0], weight_ba.shape[0]], dim=-1)
        return qkvz.reshape(*x.shape[:-1], weight_qkvz.shape[0]), ba.reshape(
            *x.shape[:-1], weight_ba.shape[0]
        )

    # Keep prefill on the same unquantized GEMM dispatcher as the original
    # MergedColumnParallelLinear path, while preserving the custom op in the
    # compiled graph for decode.
    gemm = dispatch_unquantized_gemm()
    return gemm(None, x, weight_qkvz, None), gemm(None, x, weight_ba, None)


def gdn_qkvz_ba_fused_linear_fake(
    x: torch.Tensor,
    weight_qkvz: torch.Tensor,
    weight_ba: torch.Tensor,
    rows_per_block: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_qkvz.shape[0] + weight_ba.shape[0]))


def gdn_qkvz_ba_projection_fake(
    x: torch.Tensor,
    weight_qkvz: torch.Tensor,
    weight_ba: torch.Tensor,
    rows_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.new_empty((*x.shape[:-1], weight_qkvz.shape[0])),
        x.new_empty((*x.shape[:-1], weight_ba.shape[0])),
    )


direct_register_custom_op(
    op_name="gdn_qkvz_ba_fused_linear",
    op_func=gdn_qkvz_ba_fused_linear_impl,
    fake_impl=gdn_qkvz_ba_fused_linear_fake,
)

direct_register_custom_op(
    op_name="gdn_qkvz_ba_projection",
    op_func=gdn_qkvz_ba_projection_impl,
    fake_impl=gdn_qkvz_ba_projection_fake,
)


def maybe_gdn_qkvz_ba_fused_linear(
    x: torch.Tensor,
    in_proj_qkvz: torch.nn.Module,
    in_proj_ba: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    from vllm.platforms.rocm import on_gfx936

    weight_qkvz = getattr(in_proj_qkvz, "weight", None)
    weight_ba = getattr(in_proj_ba, "weight", None)
    if weight_qkvz is None or weight_ba is None:
        return None
    if getattr(in_proj_qkvz, "bias", None) is not None:
        return None
    if getattr(in_proj_ba, "bias", None) is not None:
        return None
    if not envs.VLLM_ROCM_USE_SKINNY_GEMM or not on_gfx936():
        return None
    if _get_llmm1_two_weights_op() is None:
        return None
    if x.dtype not in (torch.float16, torch.bfloat16):
        return None
    if x.dtype != weight_qkvz.dtype or x.dtype != weight_ba.dtype:
        return None
    if not x.is_cuda or not weight_qkvz.is_cuda or not weight_ba.is_cuda:
        return None
    if not weight_qkvz.is_contiguous():
        return None
    if not weight_ba.is_contiguous():
        return None
    if tuple(weight_qkvz.shape) != _QWEN35_QKVZ_SHAPE:
        return None
    if tuple(weight_ba.shape) != _QWEN35_BA_SHAPE:
        return None

    return torch.ops.vllm.gdn_qkvz_ba_projection(
        x, weight_qkvz, weight_ba, _QWEN35_ROWS_PER_BLOCK
    )
