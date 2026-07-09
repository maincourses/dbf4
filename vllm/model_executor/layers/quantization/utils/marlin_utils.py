# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types
from .quant_utils import pack_cols, unpack_cols
  
logger = init_logger(__name__)

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: bool | None = None,
    include_fp_type: bool = True,
    device_capability: int | None = None,
):
    if current_platform.is_cpu():
        return _query_cpu_marlin_supported_quant_types(has_zp, include_fp_type)

    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    if device_capability < 75:
        return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        )
        types1 = query_marlin_supported_quant_types(
            True, include_fp_type, device_capability
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res


def _query_cpu_marlin_supported_quant_types(
    has_zp: bool | None = None,
    include_fp_type: bool = True,
):
    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = _query_cpu_marlin_supported_quant_types(
            False,
            include_fp_type,
        )
        types1 = _query_cpu_marlin_supported_quant_types(
            True,
            include_fp_type,
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias, only supports 4-bits for now
        res = [scalar_types.uint4b8]
        return res


def _check_marlin_supported(
    quant_type: ScalarType,
    group_size: int | None,
    has_zp: bool,
    device_capability: int | None = None,
) -> tuple[bool, str | None]:
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} "
            f"are supported (for group_size = {group_size}, "
            f"device_capability = {device_capability}, zp = {has_zp}).",
        )
    if group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        return (
            False,
            f"Marlin does not support group_size = {group_size}. "
            f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
            "are supported.",
        )

    return True, None


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: int | None = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
    quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )


def check_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> tuple[bool, str | None]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, e.__str__()
    return True, None


#暂不支持marlinlinear
def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    if current_platform.is_rocm():
        return False
    output_size_per_partition = (
        getattr(layer, "output_size_per_partition", None) or layer.output_size
    )
    input_size_per_partition = (
        getattr(layer, "input_size_per_partition", None) or layer.input_size
    )

    # return check_marlin_supports_shape(
    #     output_size_per_partition=output_size_per_partition,
    #     input_size_per_partition=input_size_per_partition,
    #     input_size=layer.input_size,
    #     group_size=group_size,
    # )[0]
    return False

def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    # if current_platform.is_rocm():
    #     return False
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition
    # apply_router_weight_on_input is not supported for moe marlin
    supports_router_weight = not layer.apply_router_weight_on_input

    #暂时只支持bw
    device_name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    supports_device = "BW" in device_name
    # gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
    # down: (n, k) = (hidden_size, intermediate_size_per_partition)
    # moe marlin requires n % 128 == 0 and k % 64 == 0
    supports_shape = (
        hidden_size % 128 == 0
        and intermediate_size_per_partition % max(64, group_size) == 0
    )

    supports_group_size = group_size in [64] #暂时只支持64
    return supports_shape and supports_group_size and supports_router_weight


def marlin_moe_intermediate_size(w1_packed: torch.Tensor, w2_packed: torch.Tensor):
    """
    Given Marlin packed weight matrices w1_packed, and w2_packed,
    return the MoE intermediate size N
    """
    marlin_tile_size = 16
    return w2_packed.size(1) * marlin_tile_size


def marlin_make_workspace(
    output_size_per_partition: int, device: torch.device
) -> torch.Tensor:
    max_workspace_size = (
        output_size_per_partition // GPTQ_MARLIN_MIN_THREAD_N
    ) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(
        max_workspace_size, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_make_empty_zp(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


# def get_scale_perms():
#     scale_perm: list[int] = []
#     for i in range(8):
#         scale_perm.extend([i + 8 * j for j in range(8)])
#     scale_perm_single: list[int] = []
#     for i in range(4):
#         scale_perm_single.extend(
#             [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
#     return scale_perm, scale_perm_single


# def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
#                           group_size: int) -> torch.Tensor:

#     scale_perm, scale_perm_single = get_scale_perms()
#     if group_size < size_k and group_size != -1:
#         s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
#     else:
#         s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
#     s = s.reshape((-1, size_n)).contiguous()

#     return s
def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(16): # 遍历列方向不同scale的 8个线程
        scale_perm.extend([i + 16 * j for j in range(8)]) # 插入 8 个数据块中 对应位置的索引
    return scale_perm


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False # [56, 512]  7168  512  128
) -> torch.Tensor:
    # 将[128, 128](fp16) B矩阵中 每个[16, 16]计算块中的对应位置的 zero值 放到一起 
    scale_perm = get_scale_perms()
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_act_int8_process_scales(s: torch.Tensor):
    a_scales_scale_factor = 1 / 4096 * s.max().float()
    s = s / s.max() * 4096
    s = s.round().to(torch.int16).view(s.dtype)
    return s, a_scales_scale_factor


def marlin_moe_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size, is_a_8bit)
    return output


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int, is_a_8bit: bool = False
) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    # 和 scale 使用一致的重排逻辑，将[128, 128](fp16) B矩阵中 每个[16, 16]计算块中的对应位置的 zero值 放到一起 
    scale_perm = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # uint4 混排
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    if not is_a_8bit:
        # uint4打包成 int32
        zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits, is_a_8bit)
    return marlin_zp


def moe_awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(
            q_zp_packed[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        logger.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible."
        )


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        return
    logger.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."
    )


def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    if not envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


_quant_fp8_method: QuantFP8 | None = None


def get__quant_fp8_method() -> QuantFP8:
    global _quant_fp8_method
    if _quant_fp8_method is None:
        _quant_fp8_method = QuantFP8(False, GroupShape.PER_TOKEN)
    return _quant_fp8_method


def get_marlin_input_dtype(prefix: str | None = None):
    if envs.VLLM_MARLIN_INPUT_DTYPE is None:
        return
    elif envs.VLLM_MARLIN_INPUT_DTYPE.lower() == "int8":
        return torch.int8
    elif envs.VLLM_MARLIN_INPUT_DTYPE.lower() == "fp8":
        if not current_platform.is_device_capability(
            89
        ) and not current_platform.is_device_capability(120):
            raise ValueError(
                "Marlin W4A8-FP8 only support SM89 or SM120 device "
                "(It is slower than Marlin W4A16 on other devices). "
                "You can consider using W4A8-INT8 instead"
                "(set VLLM_MARLIN_INPUT_DTYPE=int8)."
            )

        _ = get__quant_fp8_method()
        return torch.float8_e4m3fn
    else:
        return


def marlin_quant_input(x: torch.Tensor, quant_dtype: torch.dtype):
    x = x.reshape(-1, x.shape[-1])
    if quant_dtype == torch.int8:
        return per_token_quant_int8(x)
    elif quant_dtype == torch.float8_e4m3fn:
        return get__quant_fp8_method()(x)
    else:
        raise ValueError(f"unsupported quant_dtype {quant_dtype}")


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    wtype: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    input_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    a_scales = None
    if input_dtype == torch.int8:
        assert wtype == scalar_types.uint4b8, (
            "W8A8-INT8 is not supported by marlin kernel."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)
        a_scales = a_scales * input_global_scale
    elif input_dtype == torch.float8_e4m3fn:
        assert wtype == scalar_types.uint4b8, (
            "INT8 weight + FP8 activation is not supported."
        )

        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)

    output = ops.marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        a_scales,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    a_scales = None
    if input_dtype == torch.int8:
        assert quant_type == scalar_types.uint4, (
            "W8A8-INT8 is not supported by marlin kernel."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)
        a_scales = a_scales * input_global_scale
    elif input_dtype == torch.float8_e4m3fn:
        assert quant_type == scalar_types.uint4, (
            "INT8 weight + FP8 activation is not supported."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)

    output = ops.marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        a_scales,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


def merge_scales_zeros(marlin_s: torch.Tensor, marlin_zp: torch.Tensor,
                                    data_num_0: int, data_num_1: int) -> torch.Tensor:                           
    """
    合并两个 Tensor, 每行交替取 data_num_0 个 float16 和 data_num_1 个 int32。
    要求：
      - marlin_s 每行长度能被 data_num_0 整除
      - marlin_zp 每行长度能被 data_num_1 整除
      - 合并后的总字节数必为 4 的倍数

    返回:
        [N, M] 的 int32 Tensor(行数一致，列数已对齐)
    """
    assert marlin_s.shape[0] == marlin_zp.shape[0], "Batch size mismatch"
    assert marlin_s.dtype == torch.float16
    assert marlin_zp.dtype == torch.int32

    N, D0 = marlin_s.shape
    _, D1 = marlin_zp.shape

    assert D0 % data_num_0 == 0, "marlin_s 每行必须能被 data_num_0 整除"
    assert D1 % data_num_1 == 0, "marlin_zp 每行必须能被 data_num_1 整除"

    s_block_count = D0 // data_num_0
    zp_block_count = D1 // data_num_1
    assert s_block_count == zp_block_count

    total_blocks = s_block_count

    # 转为字节视图
    s_bytes = marlin_s.view(torch.uint8).reshape(N, -1)
    zp_bytes = marlin_zp.view(torch.uint8).reshape(N, -1)

    # 每行的合并结果
    merged_rows = []

    for i in range(N):
        s_row = s_bytes[i]
        zp_row = zp_bytes[i]
        s_ptr = 0
        zp_ptr = 0
        merged = []

        for _ in range(total_blocks):
            # 如果 s 还有剩余 block，就取
            if s_ptr < s_row.numel():
                chunk_s = s_row[s_ptr: s_ptr + data_num_0 * 2]  # float16 = 2 字节
                merged.append(chunk_s)
                s_ptr += data_num_0 * 2

            # 如果 zp 还有剩余 block，就取
            if zp_ptr < zp_row.numel():
                chunk_zp = zp_row[zp_ptr: zp_ptr + data_num_1 * 4]  # int32 = 4 字节
                merged.append(chunk_zp)
                zp_ptr += data_num_1 * 4

        # 合并所有字节，并直接转换为 int32
        merged_bytes = torch.cat(merged)
        # assert merged_bytes.numel() % 4 == 0, "最终字节长度必须是4的倍数"
        merged_int32 = merged_bytes.view(torch.int32)
        merged_rows.append(merged_int32)

    # 所有合并行长度一致，可以直接堆叠
    result = torch.stack(merged_rows)
    return result

def awq_marlin_moe_permute_sz(
            s : torch.Tensor,
            z : torch.Tensor,
            size_k: int,
            size_n: int,
            ) -> torch.Tensor:
    num_experts = s.shape[0]
    
    # output = torch.empty((num_experts, size_k // 16, size_n//2 + size_n//8),
    #                     device=z.device,
    #                     dtype=z.dtype)

    outputs = []
    for e in range(num_experts):
        out_sz = merge_scales_zeros(s[e], z[e], 128, 16)
        outputs.append(out_sz)
    return torch.stack(outputs, dim=0)
