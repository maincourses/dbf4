import torch
import numpy as np

try:
    from lightop import awq_marlin_repack_w4a8
    use_lightop = False
except Exception:
    use_lightop = False

def unpack_int8_to_int4(tensor_int8: torch.Tensor) -> torch.Tensor:
    """
    将[N, K//2]大小的torch.int8 Tensor，转换为[N, K]大小的torch.int32 Tensor。
    每个int8包含两个int4，分别提取到int32的低4位，其余位为0。

    Args:
        tensor_int8 (torch.Tensor): 输入张量，形状为[N, K//2]，类型为torch.int8。

    Returns:
        torch.Tensor: 输出张量，形状为[N, K]，类型为torch.int32。
    """
    if tensor_int8.dtype != torch.int8:
        raise ValueError("Input tensor must be of type torch.int8")

    N, K_half = tensor_int8.shape
    tensor_uint8 = tensor_int8.to(torch.uint8)
    high4 = tensor_uint8 & 0x0F
    low4 = (tensor_uint8 >> 4) & 0x0F
    unpacked = torch.empty((N, K_half * 2), dtype=torch.int32, device=tensor_int8.device)
    unpacked[:, 0::2] = low4.to(torch.int32)
    unpacked[:, 1::2] = high4.to(torch.int32)

    return unpacked

def get_weight_perms(interleave: bool=True):
    perm = []
    for i in range(64):

        for col in range(4):
            cur_col = (i % 16) * 4 + col
            for row in range(8):
                cur_row = (i // 16) * 8 + row
                cur_idx =  cur_row * 64 + cur_col
                perm.append(cur_idx)

    perm = np.array(perm)
    if interleave:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()

    perm = torch.from_numpy(perm)

    return perm

def marlin_weights(q_w,weight_perm,k_tile=32,n_tile=64,pack_factor=8):
    size_k, size_n = q_w.shape
    q_w = q_w.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // k_tile, size_n * k_tile))
    q_w = q_w.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(q_w.shape)

    orig_device = q_w.device

    q_w = q_w.contiguous().to(torch.int32)
    M, N = q_w.shape
    assert N % pack_factor == 0, f"size_n ({N}) must be divisible by pack_factor ({pack_factor})"
    q_packed = torch.zeros((M, N // pack_factor), dtype=torch.int32, device=orig_device)
    for i in range(pack_factor):
        q_packed += q_w[:, i::pack_factor] << (4 * i)

    return q_packed

def w4a8_2_marlin_weight(w4a8_w):
    full_w4a8_w = unpack_int8_to_int4(w4a8_w)
    full_w4a8_w = full_w4a8_w.T
    weight_perm = get_weight_perms()
    marlin_q_w = marlin_weights(full_w4a8_w, weight_perm, k_tile=32, n_tile=64, pack_factor=8)
    return marlin_q_w

def w4a8_weight_repack_impl(input):
    if use_lightop:
        size_batch = input.shape[0]
        size_n = input.shape[1]
        size_k = input.shape[2] * 2
        output = torch.zeros((size_batch, size_k // 32, size_n * 4), device=input.device, dtype=torch.int32)
        awq_marlin_repack_w4a8(input, output, size_batch, size_k, size_n)
    else:
        w_marlin_list = []
        for e in range(input.shape[0]):
            w_marlin_in = w4a8_2_marlin_weight(input[e])
            w_marlin_list.append(w_marlin_in)
        output = torch.stack(w_marlin_list, dim=0)

    return output