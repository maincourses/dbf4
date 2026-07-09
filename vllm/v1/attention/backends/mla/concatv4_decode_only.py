import triton
import triton.language as tl
import torch
from functools import reduce
import pytest
import torch
import math

@pytest.mark.parametrize("shape_pair,dim", [

    (((4, 8, 512), (4, 8, 64)), 2),
    (((8, 8, 512), (8, 8, 64)), 2),
    (((16, 8, 512), (16, 8, 64)), 2),
    (((32, 8, 512), (32, 8, 64)), 2),
    (((64, 8, 512), (64, 8, 64)), 2),
    (((128, 8, 512), (128, 8, 64)), 2),
    (((256, 8, 512), (256, 8, 64)), 2),
    (((512, 8, 512), (512, 8, 64)), 2),
    (((672, 8, 512), (672, 8, 64)), 2),    
    (((768, 8, 512), (768, 8, 64)), 2),
    (((896, 8, 512), (896, 8, 64)), 2),
    (((1024, 8, 512), (1024, 8, 64)), 2),

    (((4, 16, 512), (4, 16, 64)), 2),
    (((8, 16, 512), (8, 16, 64)), 2),
    (((16, 16, 512), (16, 16, 64)), 2),
    (((32, 16, 512), (32, 16, 64)), 2),
    (((64, 16, 512), (64, 16, 64)), 2),
    (((128, 16, 512), (128, 16, 64)), 2),
    (((256, 16, 512), (256, 16, 64)), 2),
    (((512, 16, 512), (512, 16, 64)), 2),
    (((672, 16, 512), (672, 16, 64)), 2),    
    (((768, 16, 512), (768, 16, 64)), 2),
    (((896, 16, 512), (896, 16, 64)), 2),
    (((1024, 16, 512), (1024, 16, 64)), 2),


    (((4, 32, 512), (4, 32, 64)), 2),
    (((8, 32, 512), (8, 32, 64)), 2),
    (((16, 32, 512), (16, 32, 64)), 2),
    (((32, 32, 512), (32, 32, 64)), 2),
    (((64, 32, 512), (64, 32, 64)), 2),
    (((128, 32, 512), (128, 32, 64)), 2),
    (((256, 32, 512), (256, 32, 64)), 2),
    (((512, 32, 512), (512, 32, 64)), 2),
    (((672, 32, 512), (672, 32, 64)), 2),    
    (((768, 32, 512), (768, 32, 64)), 2),
    (((896, 32, 512), (896, 32, 64)), 2),
    (((1024, 32, 512), (1024, 32, 64)), 2),




    
])
def test_concat_Acc(shape_pair, dim):

    torch.manual_seed(1)
    shape1, shape2 = shape_pair
    M = shape1[0]
    N = shape1[1]
    x_sizes = [M, N, 512]
    x_strides = [512, 512*M, 1]
    x_max_index = M * N * 512
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)

    # print("形状:", x.shape)      # [4, 8, 512]
    # print("步幅:", x.stride())   # (1536, 192, 1)
    
    y_sizes = [M, N, 64]
    y_strides = [1536*(N//8), 192, 1]
    y_max_index = 1536*(N//8) * M
    y_required_length = y_max_index
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)
 
    
    
    expected = torch.cat([x,y], dim=dim)
    result = concat_helper(x, y, dim=dim)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), "Mismatch"





@triton.jit
def concat_kernel(
    A_ptr, B_ptr, C_ptr,
    A_section_numel, B_section_numel, C_section_numel,
    Per_block, 
    section_num,
    M,
    N,
    Astride_0,
    Astride_1,
    Astride_2,
    Bstride_0,
    Bstride_1,
    Bstride_2,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    for sub_section_index in range(Per_block):
        sub_offset = block_idx * Per_block + sub_section_index
        M_idx = sub_offset // N
        N_idx = sub_offset % N
        if sub_offset <= section_num-1:
            C_ptr_block_start = C_ptr + sub_offset * C_section_numel
            A_ptr_block_start = A_ptr + M_idx * Astride_0 + N_idx * Astride_1 
            B_ptr_block_start = B_ptr + M_idx * Bstride_0 + N_idx * Bstride_1 
            
            for offset in range(0, A_section_numel, BLOCK_SIZE):
                offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                mask = offset_idx < A_section_numel
                val_from_A = tl.load(A_ptr_block_start + offset_idx, mask=mask)
                tl.store(C_ptr_block_start + offset_idx, val_from_A, mask=mask)

            for offset in range(0, B_section_numel, BLOCK_SIZE):
                offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                mask = offset_idx < B_section_numel
                val_from_B = tl.load(B_ptr_block_start + offset_idx, mask=mask)
                tl.store(C_ptr_block_start + A_section_numel + offset_idx, val_from_B, mask=mask)                
    
def concat_helper(A:torch.Tensor, B:torch.Tensor, dim:int):

    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    
    if dim!=0 :
        block_num = reduce(lambda x, y: x * y, output_shape[:dim])
        Per_block = 1 
        unit_offset_A, unit_offset_B, unit_offset_C = A.shape[dim],B.shape[dim],C.shape[dim]
        if (A.shape[1]==8 and A.shape[0] > 512) or ( A.shape[1]==16 and A.shape[0] > 256):
            Per_block = 2
        if ( A.shape[1]==32 and A.shape[2] == 512 and A.shape[0] > 256):
            Per_block = 8
        num_blocks = math.ceil(block_num/Per_block)
        concat_kernel[(num_blocks,)](
                A, B, C,
                unit_offset_A, unit_offset_B, unit_offset_C,
                Per_block, 
                block_num,
                output_shape[0],
                output_shape[1],
                A.stride(0),
                A.stride(1),
                A.stride(2),
                B.stride(0),
                B.stride(1),
                B.stride(2),                 
                BLOCK_SIZE=1024)
        return C

    assert False, "not support"


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=['M','N'],
        x_vals=[(4,8),(8,8),(16,8),(32,8),(64,8),(96,8),(128,8),(256,8),(512,8),(768,8),(1024,8), \
                (4,16),(8,16),(16,16),(32,16),(64,16),(96,16),(128,16),(256,16),(512,16),(768,16),(1024,16), \
                (4,32),(8,32),(16,32),(32,32),(64,32),(96,32),(128,32),(256,32),(512,32),(768,32),(1024,32)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='s',
        plot_name='concat-dim2',
        args={"dim":2},
    ),
)

@triton.testing.perf_report(configs)
def benchmark(M, N, provider, dim):

    x_sizes = [M, N, 512]
    x_strides = [512, 512*M, 1]
    x_max_index = M * N * 512
    x_required_length = x_max_index
    x_data = torch.arange(x_required_length,device='cuda').bfloat16()  
    x = torch.as_strided(x_data, size=x_sizes, stride=x_strides)

    # print("形状:", x.shape)      # [M, 8, 512]
    # print("步幅:", x.stride())   # (512, 512*M, 1)
    
    y_sizes = [M, N, 64]
    y_strides = [1536*(N//8), 192, 1]
    y_max_index = 1536*(N//8) * M
    y_required_length = y_max_index 
    y_data = torch.arange(y_required_length,device='cuda').bfloat16()  
    y = torch.as_strided(y_data, size=y_sizes, stride=y_strides)

    # print("形状:", y.shape)      # [M, 8, 64]
    # print("步幅:", y.stride())   # (1536, 192, 1)
  
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_helper(x, y,dim=dim), quantiles=quantiles)
    return (ms*1000), (max_ms*1000), (min_ms*1000)

# @triton.testing.perf_report(configs)
# def benchmark_16(size, provider, dim):
#     x = torch.rand([size,16,512], device='cuda', dtype=torch.bfloat16)
#     y = torch.rand([size,16,64], device='cuda', dtype=torch.bfloat16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == 'torch':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_helper(x, y,dim=dim), quantiles=quantiles)
#     return (ms*1000), (max_ms*1000), (min_ms*1000)

# @triton.testing.perf_report(configs)
# def benchmark_32(size, provider, dim):
#     x = torch.rand([size,32,512], device='cuda', dtype=torch.bfloat16)
#     y = torch.rand([size,32,64], device='cuda', dtype=torch.bfloat16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == 'torch':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_helper(x, y,dim=dim), quantiles=quantiles)
#     return (ms*1000), (max_ms*1000), (min_ms*1000)

# @triton.testing.perf_report(configs)
# def benchmark_prefill(size, provider, dim):
#     x = torch.rand([size,32,128], device='cuda', dtype=torch.bfloat16)
#     y = torch.rand([size,32,64], device='cuda', dtype=torch.bfloat16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == 'torch':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cat([x,y],dim=dim), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: concat_helper(x, y,dim=dim), quantiles=quantiles)
#     return (ms*1000), (max_ms*1000), (min_ms*1000)

if __name__ == '__main__':
    benchmark.run(save_path="./triton_test",print_data=True)
    # benchmark_16.run(save_path="./triton_test_16",print_data=True)
    # benchmark_32.run(save_path="./triton_test_32",print_data=True)
    # benchmark_prefill.run(save_path="./triton_test_prefill",print_data=True)