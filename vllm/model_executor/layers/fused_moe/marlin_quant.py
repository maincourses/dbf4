import torch
import numpy as np

# 从 [32, 64] int32的size中，重排后 每行相邻的8个uint4数据 混排后 pack成uint32数据

#原本是32 * 16算一次mmac，因为npack组成32 * 64大小
#现在是16 * 16算一次mmac，因为npack组成16 * 32大小
#这里是在对32 * 64 进行数据的重排 

def get_weight_perms(interleave: bool=False):
    # ================== 4条mmac 指令进行拼接的结果 ============
    perm = []
    for i in range(64): # 遍历64个线程，因为是针对一个warp内的

        for col in range(2): # 遍历列方向2次， 代表2次mmac指令 具体是行还是列还不知道

            cur_col = (i % 16) * 2 + col  #计算当前线程在哪个列 这里是占据4列

            for row in range(4): # 每个线程在 每个mmac中需要取8个uint4数据 占据8行
                cur_row = (i // 16) * 4 + row
                # 计算在整个 [32, 64]范围内的实际偏移
                cur_idx =  cur_row * 32 + cur_col
                perm.append(cur_idx)

    perm = np.array(perm)
    if interleave:
        # =================  加入混排策略 =================
        # # interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # # interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        # QQQ 类似的 pack混排策略
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # 按照 interleave 重排后展成 一维数组
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
    
    perm = torch.from_numpy(perm)

    return perm

#npack重排 //512大小
def marlin_weights_npack2(
                    q_w,
                    weight_perm,
                    k_tile=16,
                    n_tile=32):
    # 2048, 768
    size_k, size_n = q_w.shape

    # [7168, 512] ==> [128, 16, 24，32]
    q_w = q_w.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
    # [128, 16, 24，32] ==> [128, 24, 16，32]
    q_w = q_w.permute((0, 2, 1, 3))
    # [128, 24, 16，32] ==> [128, 12288]
    q_w = q_w.reshape((size_k // k_tile, size_n * k_tile))
    # 按照指定的 perm进行重排
    q_w = q_w.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(q_w.shape)

    # orig_device = q_w.device

    # q_w = q_w.cpu().numpy()
    # q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    # for i in range(pack_factor):
    #     q_packed |= q_w[:, i::pack_factor] << 4 * i
    # q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_w

def w16a16_marlin_weight(full_w16a16_w # [size_n, size_k]
                        ):
    # import pdb
    # pdb.set_trace()
    # [size_n, size_k] == > [size_k, size_n] 此时已经是默认NN的 k * n 基于这个进行重排
    full_w16a16_w = full_w16a16_w.T
    # 获取 [16, 32]的权重数据块中，需要重排的顺序
    weight_perm = get_weight_perms()
    # 按照索引进行重排
    marlin_q_w = marlin_weights_npack2(full_w16a16_w, weight_perm, k_tile=16, n_tile=32)
    return marlin_q_w