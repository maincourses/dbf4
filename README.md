# <div align="center"><strong>deprecated vLLM</strong></div>
## 简介
vLLM是一个快速且易于使用的LLM推理和服务库,使用PageAttention高效管理kv内存,Continuous batching传入请求,支持很多Hugging Face模型,如LLaMA & LLaMA-2、Qwen、Chatglm2 & Chatglm3等。


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

### 编译环境准备
提供2种环境准备方式:

1. 基于光源pytorch2.9.0基础镜像环境:根据pytorch2.9.0、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境:安装pytorch2.9.0,pytorch whl包下载目录:[https://cancon.hpccube.com:65024/4/main/pytorch](https://cancon.hpccube.com:65024/4/main/pytorch),根据python、dtk版本,下载对应pytorch2.5.1的whl包。安装命令如下:
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

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
