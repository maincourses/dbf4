# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import W8a8GetCacheJSON
from vllm import envs

try:
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n") 
    
W8A8_TRITONJSON=W8a8GetCacheJSON()


def get_w8a8_int8_marlin_weights(
         weight,
         k_tile=64):
    # 7168, 512
    weight = weight.T
    size_k, size_n = weight.shape
    assert size_k // k_tile
    weight = weight.reshape(size_k // k_tile, k_tile, size_n)
    weight = weight.transpose(1, 2)
    weight = weight.reshape(size_k // k_tile, size_n * k_tile)

    return weight


def w8a8_nt_kpack2_marlin_weight(w8a8_w, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert w8a8_w.dtype == torch.int8, "w8a8_w 必须是 int8 类型"
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    w8a8_w = w8a8_w.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
    w8a8_w = w8a8_w.permute((0, 2, 1, 3)).contiguous()
    w8a8_w = w8a8_w.reshape((size_n // k_tile, size_k * k_tile))
    return w8a8_w

def weight8bit_nt_kpack2_marlin1(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                k_tile1=4,
                                n_tile=16,
                                n_tile1=16):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"


        q = weight.reshape((size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        # q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
        q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"


        q = weight.reshape((E, size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((0, 1, 4, 2, 5, 3, 6)).contiguous()
        q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q

def sparse_cutlass_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_sparse_scaled_mm_supported(capability)


def cutlass_fp8_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_fp8(capability)


def cutlass_block_fp8_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_scaled_mm_supports_block_fp8(capability)


def cutlass_group_gemm_supported() -> bool:
    if not current_platform.is_cuda():
        return False

    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()

    return ops.cutlass_group_gemm_supported(capability)


CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: float | torch.Tensor
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    #
    # Extra note: upon weight reloading weight_scale.ndim == 0
    unfused_module_in_checkpoint = (
        weight_scale.ndim != 0
        and weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            # Skip any component with zero width.
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = ops.scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    input_zero_point: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None  = None,
    bias: torch.Tensor | None = None,
    w8a8_strategy: int | None = 0,
    input_quant_args: list[torch.Tensor] | None = None, 
    silu_quant_args: list[torch.Tensor] | None = None, **_
):
    # ops.scaled_int8_quant supports both dynamic and static quant.
    # * dynamic, layer.input_scale is None and x_scale computed from x.
    # * static, layer.input_scale is scalar and x_scale is input_scale.
    if envs.USE_FUSED_RMS_QUANT and input_quant_args is not None:
        assert len(input_quant_args) == 2
        assert input_quant_args[0] is not None
        assert input_quant_args[1] is not None
        x_zp =None
        x_q, x_scale = input_quant_args
    elif envs.USE_FUSED_RMS_QUANT and silu_quant_args is not None:
        assert len(silu_quant_args) == 2
        assert silu_quant_args[0] is not None
        assert silu_quant_args[1] is not None
        x_zp =None
        x_q, x_scale = silu_quant_args
    else:
        symmetric = azp_adj is None
        if input_scale is None and input_zero_point is None and symmetric is True:
            x_q, x_scale=per_token_quant_int8(input)
            x_zp =None
        else:
            x_q, x_scale, x_zp = ops.scaled_int8_quant(input,
                                                       input_scale,
                                                       input_zero_point,
                                                       symmetric=symmetric)

    if x_zp is not None:
        # Currently, static is always per-tensor and dynamic is per-token
        static = input_zero_point is not None
        azp = None if static else x_zp
        return ops.cutlass_scaled_mm_azp(x_q,
                                         weight,
                                         scale_a=x_scale,
                                         scale_b=weight_scale,
                                         out_dtype=input.dtype,
                                         azp_adj=azp_adj,
                                         azp=azp,
                                         bias=bias)
    if w8a8_strategy==1:
        m=x_q.shape[0]
        k=x_q.shape[1]
        n=weight.shape[1]
        #print("m:{},n:{},k:{}".format(m,n,k))
        if len(W8A8_TRITONJSON.triton_json_dict)==0:
            best_config=None
            
        elif f"1_{n}_{k}" in  W8A8_TRITONJSON.triton_json_dict:
            if m<=16:
                m_=m
                #best_config=W8A8_TRITONJSON.triton_json_dict[0][f"{m}_{n}_{k}"]
            elif m<=64:
                m_= (m + 3) & -4 #取值到最近的4的倍数
            elif m<=160:
                m_=(m + 7) & -8
                
            elif m<200: #256
                m_=160
            elif m<480: #512
                m_=256
            elif m<960: #1024
                m_=512
            elif m<2048:
                m_=1024
            elif m<4096:
                m_=2048
            elif m<6000:
                m_=4096
            else:
                m_=8192  

            best_config=W8A8_TRITONJSON.triton_json_dict[f"{m_}_{n}_{k}"]
            
        else: 
            best_config=None
            
        return ops.triton_scaled_mm(
                x_q,
                weight,
                scale_a=x_scale,
                scale_b=weight_scale,
                out_dtype=input.dtype,
                bias=bias,best_config=best_config)
    elif w8a8_strategy==2:
        return ops.cutlass_scaled_mm(
                x_q,
                weight,
                scale_a=x_scale,
                scale_b=weight_scale,
                out_dtype=input.dtype,
                bias=bias)
    elif w8a8_strategy==3:
        # x_q: shape (m, k)     stride (k, 1)
        # weight: shape (n, k)  stride (k, 1)
        return ops.blaslt_scaled_mm(x_q,
                                    weight,
                                    scale_a=x_scale,
                                    scale_b=weight_scale,
                                    out_dtype=input.dtype,
                                    bias=bias)
    else:
        return ops.rocblas_scaled_mm(
                x_q,
                weight,
                scale_a=x_scale,
                scale_b=weight_scale,
                out_dtype=input.dtype,
                bias=bias) 


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale
