# <div align="center"><strong>DCU vLLM</strong></div>

## vLLM_dcu简介
vLLM 是一个快速易用的 LLM 推理和服务库。可用于大型语言模型和多模态模型的高性能服务框架，旨在在从单个GPU到大型分布式集群的各种设置中提供低延迟和高吞吐量的推理，我们基于开源社区做了DCU平台的适配和针对性的优化。
其核心功能包括：快速运行时：通过PagedAttention提供高效的服务，用于前缀缓存、零开销CPU调度器、预填充解码分解、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行性、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）和多LoRA批处理。
广泛的模型支持：支持各种语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等）、嵌入模型（E5-Mistral、GTE、ColBERT）和奖励模型（Qwen-Math），易于扩展以添加新模型。与大多数Hugging Face模型和OpenAI API兼容。
强化学习和训练后主干：vLLM是一个经过验证的全球推广后端，具有原生强化学习集成，并被知名训练后框架采用。

## 支持模型结构列表

| 结构                                | 模型 | FP16/BF16 | AWQ | GPTQ | 支持版本 | 是否优化 |
| :---------------------------------: | :------: | :------: | :------: |:------: | :------: |:------: |
| LlamaForCausalLM                    | Llama 3.2, Llama 3.1,Llama 3,Llama 2,Llama,Yi,Codellama,DeepSeek-R1-Distill-Llama     | Yes | Yes | Yes | v0.5.0，Llama 3.2>=v0.6.2 | Yes |  
| Llama4ForConditionalGeneration      | Llama 4                                                                               | No/Yes | -  | - | v0.8.5.post1  | No |
| QWenLMHeadModel                     | QWen,Qwen-VL                                                                          | Yes | Yes | Yes | v0.5.0，Qwen-VL>=v0.6.2 | Yes |
| Qwen2ForCausalLM                    | QWen2,QWen1.5,CodeQwen1.5,DeepSeek-R1-Distill-Qwen,gte_Qwen2-1.5B-instruct            | Yes | Yes | Yes | v0.5.0，gte>=v0.7.2   | Yes |
| Qwen3ForCausalLM                    | QWen3,Qwen3-Embedding,Qwen3-Reranker                                                  | Yes | - | - | v0.8.4   | Yes |
| Qwen3MoeForCausalLM                 | QWen3MoE                                      | Yes    | -   | -   | v0.8.4       | Yes |
| Qwen3NextForCausalLM                | QWen3-Next                                    | Yes    | -   | -   | v0.11.0      | Yes |
| ChatGLMModel                        | glm-4v-9b,chatglm3,chatglm2                   | Yes    | No  | Yes | v0.5.0       | Yes |
| Glm4ForCausalLM                     | GLM-4-0414                                    | No/Yes | -   | -   | v0.8.5.post1 | Yes |
| Glm4MoeForCausalLM                  | GLM-4.5,GLM-4.6,GLM-4.7,GLM-4.5-Air           | Yes    | -   | -   | v0.9.2       | Yes |
| Glm4vMoeForConditionalGeneration    | GLM-4.5V                                      | Yes    | -   | -   | v0.11.0      | Yes |
| DeepseekForCausalLM                 | Deepseek                                      | Yes    | No  | -   | v0.5.0       | Yes |
| DeepseekV2ForCausalLM               | DeepSeek-V2                                   | Yes    | No  | -   | v0.6.2       | Yes |
| DeepseekVLV2ForCausalLM             | DeepSeek-VL2                                  | Yes    | No  | -   | v0.7.2       | Yes |
| DeepseekV3ForCausalLM               | DeepSeek-V3                                   | Yes    | Yes | -   | v0.7.2       | Yes |
| DeepseekV32ForCausalLM              | DeepSeek-V3.2                                 | Yes    | Yes | -   | v0.11.0      | No  |
| GptOssForCausalLM                   | gpt-oss                                       | Yes    | -   | -   | v0.11.0      | Yes |
| BaiChuanForCausalLM                 | Baichuan2,Baichuan                            | Yes    | No  | No  | v0.11.0      | Yes |
| BloomForCausalLM                    | BLOOM                                         | Yes    | No  | Yes | v0.5.0       | Yes |
| InternLMForCausalLM                 | InternLM                                      | Yes    | No  | -   | v0.5.0       | Yes |
| InternLM2ForCausalLM                | InternLM2                                     | Yes    | No  | -   | v0.5.0       | Yes |
| FalconForCausalLM                   | falcon                                        | Yes    | No  | Yes | v0.5.0       | Yes |
| TeleChat2ForCausalLM                | TeleChat2                                     | Yes    | No  | -   | v0.7.2       | Yes |
| MiniCPMForCausalLM                  | MiniCPM                                       | Yes    | No  | -   | v0.5.0       | Yes |
| MiniCPM3ForCausalLM                 | MiniCPM3                                      | Yes    | No  | -   | v0.6.2       | Yes |
| MixtralForCausalLM                  | Mixtral-8x7B,Mixtral-8x7B-Instruct            | Yes    | No  | -   | v0.5.0       | Yes |
| Qwen2MoeForCausalLM                 | Qwen2-57B-A14B,Qwen2-57B-A14B-Instruct        | Yes    | No  | -   | v0.5.0       | No  |
| LlavaForConditionalGeneration       | LLaMA,LLaMA-2,LLaMA-3                         | Yes    | No  | -   | v0.6.2       | No  |
| Qwen2VLForConditionalGeneration     | Qwen2-VL                                      | Yes    | No  | Yes | v0.6.2       | No  |
| Qwen2_5_VLForConditionalGeneration  | Qwen2.5-VL                                    | Yes    | No  | Yes | v0.7.2       | No  |
| Qwen3VLForConditionalGeneration     | Qwen3-VL                                      | Yes    | No  | Yes | v0.11.0      | No  |
| Mistral3ForConditionalGeneration    | Mistral3                                      | Yes    | No  | -   | v0.8.5.post1 | No  |
| Gemma3ForConditionalGeneration      | Gemma 3                                       | Yes    | -   | -   | v0.8.5.post1 | No  |
| MiniCPMV                            | MiniCPM-V                                     | Yes    | No  | -   | v0.6.2       | No  |
| Phi3VForCausalLM                    | Phi-3.5-vision                                | Yes    | No  | -   | v0.6.2       | No  |
| BertModel                           | bge-large-zh-v1.5                             | Yes    | No  | -   | v0.7.2       | No  |
| XLMRobertaModel                     | bge-m3                                        | Yes    | No  | -   | v0.7.2       | No  |
| XLMRobertaForSequenceClassification | bge-reranker-v2-m3                            | Yes    | No  | -   | v0.7.2       | No  |

## 使用源码编译方式安装

提供2种环境准备方式:

1. 基于光源pytorch2.9.0基础镜像环境:根据pytorch2.9.0、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境:安装pytorch2.9.0,pytorch whl包下载目录:https://cancon.hpccube.com:65024/4/main/pytorch,根据python、dtk版本,下载对应pytorch2.5.1的whl包。安装命令如下:

```shell
pip install torch* (下载的torch的whl包)
pip install setuptools wheel
```

### 源码编译安装
```shell
git clone http://10.16.6.30/dcutoolkit/deeplearing/vllm.git # 根据需要的分支进行切换
```
安装依赖:
```shell
pip install -r requirements/rocm.txt
```
- 提供2种源码编译方式(进入vllm目录):
```
1. 编译whl包并安装
python setup.py bdist_wheel 
cd dist
pip install vllm*

2. 源码编译安装
python3 setup.py install （若调试，可使用python3 setup.py develop）
```
若需要添加git号，设置环境变量: export ADD_GIT_VERSION=1


### 运行基础环境准备
1、使用上面基于光源pytorch2.9.0基础镜像环境

2、根据pytorch2.9.0、python、dtk及系统下载对应的依赖包:
- triton:[https://cancon.hpccube.com:65024/4/main/triton](https://cancon.hpccube.com:65024/4/main/triton/)
- flash_attn: [https://cancon.hpccube.com:65024/4/main/flash_attn](https://cancon.hpccube.com:65024/4/main/flash_attn)
- flash_mla: [https://cancon.hpccube.com:65024/4/main/flash_mla](https://cancon.hpccube.com:65024/4/main/flash_mla)
- lightop: [https://cancon.hpccube.com:65024/4/main/lightop](https://cancon.hpccube.com:65024/4/main/lightop)
- lmslim: [https://cancon.hpccube.com:65024/4/main/lmslim](https://cancon.hpccube.com:65024/4/main/lmslim)

### 注意事项
+ 若使用 pip install 下载安装过慢,可添加源: -i https://pypi.tuna.tsinghua.edu.cn/simple/

## 验证
- python -c "import vllm; print(vllm.\_\_version__)",版本号与官方版本同步,查询该软件的版本号,例如0.15.1;

## PD 分离
#### 注释：enable_multiple_machines:true：是否是跨机的这里P和D的服务都要设置，只要有一个跨机，就要设置true；enable_asymmetric_p2p：是否是非对称切分；remote_tp_size：D的tpsize；remote_pp_size：D的ppsize （这里的非对成切分支持mla的模型）
### 环境变量

```bash
export NCCL_NCHANNELS_PER_PEER=2
export IP_CONFIG_FILE=/data/ip_config.txt ## 第一个ip为D的第一个节点，第二个ip为D的第二个节点
export NCCL_IB_HCA=,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export VLLM_HOST_IP=10.16.1.76  #ip地址 不同的节点这个需要对应修改
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
```
## P、D单实例单机的任意切分方式（满足D的tp>=P的tp)使用。
### 代理
```bash
在P的节点，例子里是75节点：
cd vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd
python3 disagg_proxy_p2p_nccl_xpyd.py
特别注意，这里如果服务重启，代理也需要重启
```
### P的运行指令：
```bash
 vllm serve /module/DeepSeek-R1-W4A8-V2/   --port 20011 --trust-remote-code  --dtype bfloat16 --max-model-len 49152  --max-num-batched-tokens 8192  -tp 1 -pp 8  --gpu-memory-utilization 0.9 --max-num-seqs 256  --disable-log-requests  --block-size 64 --enforce-eager -q slimquant_w4a8_marlin --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'      --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"enable_asymmetric_p2p":true,"remote_tp_size":2,"remote_pp_size":4,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20011","send_type":"PUT_ASYNC"}}'  --kv-cache-dtype fp8_e5m2
```
### D的运行指令：
```bash
vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 2 -pp 4  --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2     --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'
```
## P：PP2 TP8 D：TP8
### 代理
```bash
在P的节点，例子里是75节点：
cd vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd
python3 disagg_proxy_p2p_nccl_xpyd_mult_mac.py  # 最新版本执行，老版本没有这个文件，就执行disagg_proxy_p2p_nccl_xpyd.py
```
### P的运行指令：
```bash
在75节点运行：ray start --head --node-ip-address=10.16.1.75 --port=8244 --num-gpus=8 --num-cpus=32
在76节点运行：ray start --address='10.16.1.75:8244' --num-gpus=8 --num-cpus=32
在75节点启动服务：vllm serve  /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20005  --trust-remote-code --distributed-executor-backend ray --dtype bfloat16 --max-model-len 32768  -tp 8 -pp 2  --gpu-memory-utilization 0.90 --max-num-seqs 256 --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' --disable-log-requests --block-size 64 --enable-chunked-prefill --max-num-batched-tokens 6144 --no-enable-prefix-caching  --enforce-eager --kv-cache-dtype fp8_e5m2 -q slimquant_marlin --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"enable_multiple_machines":true,"enable_asymmetric_p2p":false,"remote_tp_size":8,"remote_pp_size":1,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","mem_pool_size_gb":64}}'

```
### D的运行指令：
```bash
vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 8 --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2 --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"enable_multiple_machines":true，"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'

```
## P：PP2 TP8 D：PP2 TP8
### 代理
```bash
在P的节点，例子里是75节点：
cd /data/vllm092_dev_xiabo/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd
python3 disagg_proxy_p2p_nccl_xpyd_mult_mac.py  # 最新版本执行，老版本没有这个文件，就执行disagg_proxy_p2p_nccl_xpyd.py
```
### P的运行指令：
```bash
在75节点运行：ray start --head --node-ip-address=10.16.1.75 --port=8244 --num-gpus=8 --num-cpus=32
在76节点运行：ray start --address='10.16.1.75:8244' --num-gpus=8 --num-cpus=32
在75节点启动服务：vllm serve  /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20005  --trust-remote-code --distributed-executor-backend ray --dtype bfloat16 --max-model-len 32768  -tp 8 -pp 2  --gpu-memory-utilization 0.90 --max-num-seqs 256 --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' --disable-log-requests --block-size 64 --enable-chunked-prefill --max-num-batched-tokens 6144 --no-enable-prefix-caching  --enforce-eager --kv-cache-dtype fp8_e5m2 -q slimquant_marlin --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"enable_multiple_machines":true,"enable_asymmetric_p2p":false,"remote_tp_size":8,"remote_pp_size":1,"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","mem_pool_size_gb":64}}'
```
### D的运行指令：
```bash
在77节点运行：ray start --head --node-ip-address=10.16.1.77 --port=9244 --num-gpus=8 --num-cpus=32
在26节点运行：ray start --address='10.16.1.77:9244' --num-gpus=8 --num-cpus=32
在77节点启动服务：vllm serve /module/DeepSeek-R1-W4A8-V2/ --host 0.0.0.0   --port 20009 --trust-remote-code --dtype bfloat16 -q slimquant_w4a8_marlin --max-model-len 16484 -tp 8 -pp 2 --gpu-memory-utilization 0.90 --max-num-seqs 100 --block-size 64 --disable-log-requests  --max-num-batched-tokens 16484  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}'  --kv-cache-dtype fp8_e5m2 --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e8","kv_port":"22001","kv_connector_extra_config":{"enable_multiple_machines":true，"proxy_ip":"10.16.1.75","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","mem_pool_size_gb":128}}'
```
## low_latency （使用deepep）

```bash
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_LIGHTOP=1

# deep_ep
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/work/topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256

```
topo.config

```YAML
0000:9f:00.0 mlx5_2 2
0000:57:00.0 mlx5_3 3
0000:5e:00.0 mlx5_4 4
0000:05:00.0 mlx5_5 5
0000:e5:00.0 mlx5_6 6
0000:c1:00.0 mlx5_7 7
0000:cc:00.0 mlx5_8 8
0000:b1:00.0 mlx5_9 9
```
单机ep8dp8部署示例
```bash
vllm serve /models/GLM-5-W8A8  \
  --disable-log-requests \
  -q slimquant_marlin \
  --trust-remote-code \
  -dp 8 \
  -tp 1 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 50000 \
   --max-num-batched-tokens 128 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}' \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3, "quantization": "slimquant_marlin"}' 
```
双机ep16dp16部署示例
```bash
#node1 作为主节点
vllm serve /models/GLM-5-W8A8 \
  --disable-log-requests \
  -q slimquant_marlin \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 72000 \
   --max-num-batched-tokens 128 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.88 \
  --data-parallel-size-local 8 \
  --data-parallel-address ${node1_ip} \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 0 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}' \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3, "quantization": "slimquant_marlin"}'  

#node2
vllm serve /models/GLM-5-W8A8 \
  --disable-log-requests \
  -q slimquant_marlin \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 72000 \
   --max-num-batched-tokens 128 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.88 \
  --data-parallel-size-local 8 \
  --data-parallel-address ${node1_ip} \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 8 \
  --kv-cache-dtype fp8_ds_mla \
  --headless \
  -cc '{"inductor_compile_config":{"combo_kernels": false}}'  \
  --speculative_config '{"method":"mtp","num_speculative_tokens":3, "quantization": "slimquant_marlin"}'
```

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)


