# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for GPTQ."""
import functools
from typing import Optional

import torch
try:
    import lightop
except Exception:
    print("INFO: Please install lightop if you want to infer awq of marlin.\n") 

import vllm.envs as envs
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    moe_align_block_size, try_get_optimal_moe_config)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new, maybe_warn_marlin_atomic_add)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_cache
def get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        return scalar_types.uint4 if num_bits == 4 else scalar_types.uint8
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128

def fused_marlin_moe_int4(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168
                     w1_scale_zero: torch.Tensor,
                     w2_scale_zero: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,      
                     expert_map: Optional[torch.Tensor] = None,
                     g_idx1: Optional[torch.Tensor] = None,
                     g_idx2: Optional[torch.Tensor] = None,
                     sort_indices1: Optional[torch.Tensor] = None,
                     sort_indices2: Optional[torch.Tensor] = None,
                     w1_zeros: Optional[torch.Tensor] = None,
                     w2_zeros: Optional[torch.Tensor] = None,
                     workspace: Optional[torch.Tensor] = None,
                     num_bits: int = 4,
                     is_k_full: bool = True,
                     inplace: bool = False) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # quant_type = ScalarType.from_id(quant_type_id)
    # assert quant_type in [
    #     scalar_types.uint4, scalar_types.uint8b128, scalar_types.uint4b8,
    #     scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f
    # ]

    # bit4_scalar_types = [
    #     scalar_types.uint4, scalar_types.uint4b8, scalar_types.float4_e2m1f
    # ]
    # num_bits = 4 if quant_type in bit4_scalar_types else 8

    # Check constraints.
    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    # assert num_bits in [4]
    assert num_bits in [4]

    num_tokens, K = hidden_states.shape # 32, 7168
    E = w1.shape[0] # 256
    N = w2.shape[1] * 16 # 256
    topk = topk_ids.shape[1] # 8
    
    #暂时固定为16384
    #CHUNK_SIZE = 16384
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)

    if workspace is None:
        sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count
        workspace = torch.zeros(sms * 3,
                                dtype=torch.int,
                                device=hidden_states.device,
                                requires_grad=False)

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)
    if global_num_experts == -1:
        global_num_experts = E
    intermediate_cache2 = torch.empty( 
        (M * topk, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if envs.VLLM_USE_GLOBAL_CACHE13:
        intermediate_cache13 = get_moe_cache(topk, N, K, device=hidden_states.device, dtype=hidden_states.dtype)
    else:    
        intermediate_cache13 = torch.empty(
            (M * topk * max(2 * N, K), ),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    intermediate_cache1 = intermediate_cache13[:M * topk * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[:M * topk * K] 
    intermediate_cache3 = intermediate_cache3.view(-1, K) 

    use_atomic_add = hidden_states.dtype == torch.half or \
        torch.cuda.get_device_capability(hidden_states.device)[0] >= 9

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):

        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break
        intermediate_cache3 = intermediate_cache3.view(-1, K) 
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk * topk, :]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk * topk, :]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk * topk, :]
            M = tokens_in_chunk

        # Select block_size_m
        for block_size_m in [16, 32, 48, 64, 80]:
            if M * topk / E / block_size_m < 0.9:
                break

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(curr_topk_ids, block_size_m, global_num_experts, expert_map)

        intermediate_cache1 = lightop.moe_marlin_w4a16(
            curr_hidden_states, 
            intermediate_cache1,
            w1,
            w1_scale_zero,
            g_idx1,
            sort_indices1,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            curr_topk_weights,
            block_size_m,
            topk,
            False,
            expert_map is not None,
            M, 
            2 * N,
            K,
            is_k_full,
            use_atomic_add,
            True,
            False
        )

        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

        intermediate_cache3 = lightop.moe_marlin_w4a16(
            intermediate_cache2,
            intermediate_cache3,
            w2,
            w2_scale_zero,
            g_idx2,
            sort_indices2,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            curr_topk_weights,
            block_size_m,
            1,
            True,
            expert_map is not None,
            M * topk,
            K,
            N,
            is_k_full,
            use_atomic_add,
            True,
            False
            ).view(-1, topk, K)
        
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states

def fused_marlin_moe_int4_fake(
                     hidden_states: torch.Tensor, # 32, 7168 
                     w1: torch.Tensor, # 256, 512, 7168 --> 32*8, 512 --> 32*8, 256
                     w2: torch.Tensor, # 256, 256, 7168
                     w1_scale_zero: torch.Tensor,
                     w2_scale_zero: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     global_num_experts: int = -1,      
                     expert_map: Optional[torch.Tensor] = None,
                     g_idx1: Optional[torch.Tensor] = None,
                     g_idx2: Optional[torch.Tensor] = None,
                     sort_indices1: Optional[torch.Tensor] = None,
                     sort_indices2: Optional[torch.Tensor] = None,
                     w1_zeros: Optional[torch.Tensor] = None,
                     w2_zeros: Optional[torch.Tensor] = None,
                     workspace: Optional[torch.Tensor] = None,
                     num_bits: int = 4,
                     is_k_full: bool = True,
                     inplace: bool = False) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_marlin_moe_int4",
    op_func=fused_marlin_moe_int4,
    mutates_args=[],
    fake_impl=fused_marlin_moe_int4_fake,
)