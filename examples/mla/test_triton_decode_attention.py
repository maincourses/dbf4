# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import torch
import triton
from triton_decode_attention import decode_attentionv1_fwd, decode_attentionv2_fwd

def cdiv(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("B", [1])
# @pytest.mark.parametrize("L", [100])
@pytest.mark.parametrize("L", [1,100,400,700,1000,1300,1600,1900,2200,2500,2800,3100,3400,3700,4000,4300,4600,4900,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,15000,15500,16000,16500,17000,17500,18000,18500,19000,19500,20000,20500,21000,21500,22000,22500,23000,23500,24000,24500,25000,25500,26000,26500,27000,27500,28000,28500,29000,29500,30000,30500,31000,31500,32000,32500])
@pytest.mark.parametrize("H_Q", [4, 8, 16])
@pytest.mark.parametrize("H_KV", [1])
@pytest.mark.parametrize("D_QK", [576])
@pytest.mark.parametrize("D_V", [512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [16]) 

def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 4

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE) #这里为向上取整,65,(1027+16-1)//16
    req_to_page = torch.randint(0,
                                CACHE_SIZE // PAGE_SIZE,
                                (B, num_pages_per_batch, 1), #shape为（B, num_pages_per_batch, 1）的tensor,大小取值为0 至cache_size//page_size
                                device="cuda")
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE) 
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device="cuda").view(
        1, 1, -1)
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()


    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device="cuda")
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device="cuda")

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

    b_seq_len = torch.full((B, ), seq_len, device="cuda")
    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device="cuda",
    )
    b_req_idx = torch.arange(B, device="cuda").to(torch.int32)
    # Call the original implementation.
    decode_attentionv2_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )


    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    o1 = torch.zeros_like(o)
    configs = {
                  "v2_tc": {"stage1": {}, "stage2": {}},
                  "v1_2stages_tc": {"stage1": {}, "stage2": {}},
              }
    ms = {
         "v1_2stages_tc": 10000.0,
         "v2_tc": 10000.0,
     }
    final_best_config = {
                            "kernel_kind": "",
                            "best_config": {},
                            "best_us": 0.0,
                        }
    v2_tc_stage1_best_config, v2_tc_stage2_best_config = decode_attentionv2_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )
    assert torch.allclose(o, o1, atol=1e-2, rtol=1e-2)
    quantiles = [0.5, 0.2, 0.8]
    v2_tc_ms, v2_tc_min_ms, v2_tc_max_ms = triton.testing.do_bench(lambda:
    decode_attentionv2_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,

    ), quantiles=quantiles)
    for key, value in v2_tc_stage1_best_config.kwargs.items():
        configs["v2_tc"]["stage1"][key] = value
    configs["v2_tc"]["stage1"]["num_stages"] = v2_tc_stage1_best_config.num_stages
    configs["v2_tc"]["stage1"]["num_warps"] = v2_tc_stage1_best_config.num_warps
    for key, value in v2_tc_stage2_best_config.kwargs.items():
        configs["v2_tc"]["stage2"][key] = value
    configs["v2_tc"]["stage2"]["num_stages"] = v2_tc_stage2_best_config.num_stages
    configs["v2_tc"]["stage2"]["num_warps"]  = v2_tc_stage2_best_config.num_warps
    ms["v2_tc"] = v2_tc_ms
    print(f"v2_tc best configs is {configs['v2_tc']}")
    print("print mla decode attention v2 kernel [B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE] min cost :",[B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE], v2_tc_ms)
    o2 = torch.zeros_like(o)

    v1_tc_stage1_best_config, v1_tc_stage2_best_config = decode_attentionv1_fwd(
        q,
        k_buffer,
        v_buffer,
        o2,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
    )
    assert torch.allclose(o, o2, atol=1e-2, rtol=1e-2)
    
    v1_tc_ms, v1_tc_min_ms, v1_tc_max_ms = triton.testing.do_bench(lambda:
    decode_attentionv1_fwd(
        q,
        k_buffer,
        v_buffer,
        o1,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,

    ), quantiles=quantiles)
    for key, value in v1_tc_stage1_best_config.kwargs.items():
        configs["v1_2stages_tc"]["stage1"][key] = value
    configs["v1_2stages_tc"]["stage1"]["num_stages"] = v1_tc_stage1_best_config.num_stages
    configs["v1_2stages_tc"]["stage1"]["num_warps"] = v1_tc_stage1_best_config.num_warps
    configs["v1_2stages_tc"]["stage1"]["num_ldmatrixes"] = v1_tc_stage1_best_config.num_ldmatrixes
    for key, value in v1_tc_stage2_best_config.kwargs.items():
        configs["v1_2stages_tc"]["stage2"][key] = value
    configs["v1_2stages_tc"]["stage2"]["num_stages"] = v1_tc_stage2_best_config.num_stages
    configs["v1_2stages_tc"]["stage2"]["num_warps"]  = v1_tc_stage2_best_config.num_warps
    configs["v1_2stages_tc"]["stage2"]["num_ldmatrixes"] = v1_tc_stage1_best_config.num_ldmatrixes
    ms["v1_2stages_tc"] = v1_tc_ms 
    min_key, min_ms = min(ms.items(), key=lambda x: x[1])
    final_best_config["kernel_kind"] = min_key
    final_best_config["best_config"] = configs[min_key]
    final_best_config["best_us"] = min_ms * 1000
    print(f"v1_2stages_tc best configs is {configs['v1_2stages_tc']}")

    print("print mla decode attention v1 kernel [B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE] min cost :",[B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE], v1_tc_ms)
    print(f"Tuned_decode_attention choose {min_key} kernel, min cost {min_ms} ms, best config of {min_key} kernel is {configs[min_key]}")
    assert torch.allclose(o, o2, atol=1e-2, rtol=1e-2)


    #**************save config**************#
    batch = b_req_idx.shape[0]
    mean_seq_len = int((b_seq_len.sum() / max(1, batch)).item())

    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if "K100_AI" in device_name:
        # return f"QH={QH}_KVH={KVH}_QKD={QKD}_VD={VD}_{cache_dtype}_K100AI.json"
        file_name = f"QH={H_Q}_KVH={H_KV}_QKD={D_QK}_VD={D_V}_fp16_K100AI.json"
    elif "BW" in device_name:
        # return f"QH={QH}_KVH={KVH}_QKD={QKD}_VD={VD}_{cache_dtype}_BW.json"
        file_name = f"QH={H_Q}_KVH={H_KV}_QKD={D_QK}_VD={D_V}_fp16_BW.json"
    else:
        raise ValueError(f"Unsurpport device name: {device_name}")
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            config_info = json.load(file)
    else:
        config_info = {}

    # 如果 config_info 中没有当前的 batch，初始化它为一个空字典
    # if f"{batch}" not in config_info:
    #     config_info[f"{batch}"] = {}

    # 把新的 mean_seq_len 配置加入到当前 batch 中
    # config_info[f"{batch}"][f"{mean_seq_len}"] = final_best_config
    config_info[f"{mean_seq_len}"] = final_best_config

    # 保存最佳配置
    with open(file_name, 'w') as file:
        json.dump(config_info, file, indent=1)
    #**************save config**************#

