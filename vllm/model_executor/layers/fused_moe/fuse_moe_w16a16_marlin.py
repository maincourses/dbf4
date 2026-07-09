import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import torch
import triton
import triton.language as tl
import lmslim.envs as lsenvs

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_cache
if use_lightop:
    from lightop import moe_gemm_marlin_w16a16, get_moe_cuda_marlin_config_w16a16
    from lightop import op as op
@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@triton.jit
def _moe_sum_reduce_kernel(
        input_ptr,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        output_ptr,
        output_stride_0,
        output_stride_1,
        token_num: int,
        topk_num: int,
        hidden_dim: int,
        routed_scaling_factor: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
        input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    if token_num <= 32:
        BLOCK_M = 1
        BLOCK_DIM = 512
        NUM_STAGE = 2
        num_warps = 4

    elif token_num <= 128:
        BLOCK_M = 1
        BLOCK_DIM = 1024
        NUM_STAGE = 0
        num_warps = 2

    elif token_num <= 4096:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 0
        num_warps = 2
    else:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 2
        num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return

def moe_reduce_dispatch(
    intermediate_cache3: torch.Tensor,
    out_hidden_states: torch.Tensor,
    begin_chunk_idx: int,
    end_chunk_idx: int,
    routed_scaling_factor: float,
    shared_output: Optional[torch.Tensor] = None,
):
    inter_cache_view = intermediate_cache3.view(*intermediate_cache3.shape)
    n = intermediate_cache3.shape[0]

    # 根据 n 大小选择不同的 reduce 实现
    if 1 <= n <= 4:
        moe_sum_reduce_torch_compile(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
    elif 4 < n <= 1024:
        moe_sum_reduce_triton(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], 1.0)
    elif 1024 < n <= 32768:
        ops.moe_sum_opt1(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
    else:
        ops.moe_sum(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
        
    # 根据 shared_output 是否存在决定怎么更新
    if shared_output is not None:
        out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor).add_(shared_output[begin_chunk_idx:end_chunk_idx])
    else:
        out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)

def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

def moe_align_block_size_lightop(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ep_size: int = 8,
    num_token: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    local_num_experts = num_experts // ep_size
  
    if num_token:
        if num_token < block_size:
            max_num_tokens_padded = min(topk_ids.numel() * block_size, topk_ids.numel() + local_num_experts * (block_size - 1))
        else:
            max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        sorted_ids = torch.full((max_num_tokens_padded,), fill_value=topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    else:
        max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.int32,
                                 device=topk_ids.device)
        sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    if expert_map is not None:
        expert_ids = torch.zeros((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    else:
        expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.zeros((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    op.moe_align_block_size(topk_ids, 
                            num_experts, 
                            block_size, 
                            sorted_ids,
                            expert_ids, 
                            num_tokens_post_pad,
                            expert_map,
                            expert_mask,
                            num_local_tokens,
                            expert_map is not None)
    return sorted_ids, expert_ids, num_tokens_post_pad



def fused_experts_impl_w16a16_marlin(hidden_states: torch.Tensor,
                                     w1_marlin: torch.Tensor,
                                     w2_marlin: torch.Tensor,
                                     topk_weights: torch.Tensor,
                                     topk_ids: torch.Tensor,
                                     cache13: torch.Tensor,
                                     inplace: bool = False,
                                     activation: str = "silu",
                                     apply_router_weight_on_input: bool = False,
                                     global_num_experts: int = -1,
                                     expert_map: Optional[torch.Tensor] = None,
                                     use_nn_moe: Optional[bool] = False,
                                     routed_scaling_factor: Optional[float] = 1.0,
                                     shared_output: Optional[torch.Tensor] = None,
                                     num_local_tokens: Optional[torch.Tensor] = None,
                                     expect_m: Optional[int] = -1,
                                     ):

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1_marlin.is_contiguous(), "Packed weights1 must be contiguous"
    assert w2_marlin.is_contiguous(), "Packed weights2 must be contiguous"
    # 当前只支持 bf16 fp16
    assert hidden_states.dtype in [torch.bfloat16,torch.float16]
    compute_type = hidden_states.dtype
    assert use_lightop, (
        "only BW and set LMSLIM_USE_LIGHTOP=1 support Marlin W16A16 MoE")

    num_tokens, K = hidden_states.shape
    
    # Packed weights store the same number of elements as the original layout,
    # but reshaped/reordered for Marlin kernels:
    # - w1_marlin: [E, K/16, (2N)*16]
    # - w2_marlin: [E, N/16, K*16]
    E, k_div16, twoN_times16 = w1_marlin.shape
    K_w1 = k_div16 * 16
    assert K_w1 == K, f"w1_marlin K mismatch: {K_w1} vs {K}"
    assert twoN_times16 % 16 == 0
    twoN = twoN_times16 // 16
    assert twoN % 2 == 0
    N = twoN // 2

    E2, n_div16, k_times16 = w2_marlin.shape
    assert E2 == E, f"w2_marlin E mismatch: {E2} vs {E}"
    K_w2 = k_times16 // 16
    assert K_w2 == K, f"w2_marlin K mismatch: {K_w2} vs {K}"
    assert n_div16 * 16 == N, f"w2_marlin N mismatch: {n_div16 * 16} vs {N}"
   
    if global_num_experts == -1:
        global_num_experts = E

    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1

    N2 = 2 * N
    intermediate_cache1 = cache13[:M * top_k_num * N2].view(-1, N2)
    intermediate_cache3 = cache13[:M * top_k_num * K]

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N),
                                      device=hidden_states.device,
                                      dtype=compute_type)
    is_ep = expert_map is not None
    expert_mask=None
    ep_size=None
    if is_ep:
        expert_mask = torch.zeros((CHUNK_SIZE, top_k_num), dtype=torch.bool,  device=hidden_states.device, requires_grad=False)
        ep_size = global_num_experts // E

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states, dtype=compute_type)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break
        
        
        bs = tokens_in_chunk
        if num_local_tokens is not None and expect_m != -1:
            bs = expect_m

        intermediate_cache3 = intermediate_cache3.view(-1, K)
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk * top_k_num]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk * top_k_num]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk * top_k_num]
        # import logging
        # logger = logging.getLogger(__name__)
        # if not status:
        #     logger.info("lightop unsupport this size E:%s, N:%s, K:%s", E, N, K)
        config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w16a16(
            E,
            bs,
            N2,
            K,
            K,
            N,
            top_k_num,
            device_name,
            num_cus,
            hidden_states.dtype)
        assert status, f'lightop unsupport this size E:{E}, N:{N}, K:{K}'
        # Align with vLLM's default config handling for W16A16.
        # if "BLOCK_SIZE_M" not in config_marlin_0:
        #     config_marlin_0["BLOCK_SIZE_M"] = 16
        # if "BLOCK_SIZE_M" not in config_marlin_1:
        #     config_marlin_1["BLOCK_SIZE_M"] = config_marlin_0["BLOCK_SIZE_M"]
        # if "MODE" not in config_marlin_0:
        #     config_marlin_0["MODE"] = 412
        # if "MODE" not in config_marlin_1:
        #     config_marlin_1["MODE"] = 411
        # if "DELTA" not in config_marlin_0:
        #     config_marlin_0["DELTA"] = 1
        # if "DELTA" not in config_marlin_1:
        #     config_marlin_1["DELTA"] = 1

        block_size_m = config_marlin_0["BLOCK_SIZE_M"]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        if num_local_tokens is None:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, block_size_m,
                                    global_num_experts, expert_map=expert_map,
                                    expert_mask = expert_mask[:tokens_in_chunk] if is_ep else None))
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size_lightop(curr_topk_ids, block_size_m, global_num_experts, 
                                            expert_map = expert_map, 
                                            expert_mask = expert_mask[begin_chunk_idx:end_chunk_idx] if is_ep else None,
                                            num_local_tokens = num_local_tokens,
                                            ep_size=ep_size))

        # GEMM1: hidden_states * w1 -> intermediate_cache1
        moe_gemm_marlin_w16a16(
            curr_hidden_states,
            w1_marlin,
            intermediate_cache1,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            config_marlin_0,
        )
        if (envs.VLLM_USE_FUSE_SILU_AND_MUL
                and intermediate_cache1.dtype == intermediate_cache2.dtype
                == torch.float16):
            from lightop import fuse_silu_and_mul
            fuse_silu_and_mul(intermediate_cache1, intermediate_cache2)
        else:
            torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

        # GEMM2: intermediate_cache2 * w2, apply routing weights here.
        moe_gemm_marlin_w16a16(
            intermediate_cache2,
            w2_marlin,
            intermediate_cache3,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            config_marlin_1,
        )
        intermediate_cache3 = intermediate_cache3.view(-1, top_k_num, K)

        if envs.VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD and shared_output is not None:
            from lightop import op as op
            factor = (
                float(routed_scaling_factor)
                if routed_scaling_factor is not None
                else 1.0
            )
            op.moe_sum(
                input=intermediate_cache3.view(*intermediate_cache3.size()),
                output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                bias=shared_output[begin_chunk_idx:end_chunk_idx],
                expert_mask=None,
                num_local_tokens=None,
                factor=factor,
            )
        elif envs.VLLM_USE_LIGHTOP_MOE_SUM:
            from lightop import op as op
            op.moe_sum(input=intermediate_cache3.view(*intermediate_cache3.size()),
                output=out_hidden_states[begin_chunk_idx:end_chunk_idx], bias=None, 
                expert_mask=None, num_local_tokens=None, factor=1.0)
        else:
            ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.size()),
                            out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states
