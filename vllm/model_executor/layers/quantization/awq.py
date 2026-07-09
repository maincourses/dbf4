# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Union

import os
import json
import torch
from vllm import envs

from vllm.platforms import current_platform
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper
 
from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton
triton_configs_dict={}

def is_layer_skipped_awq(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)

def get_triton_cache(file_path):
        #会将所报错的json文件以字典的形式return出来
     
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            cachedata = json.load(file)
                
    #把所有的cache解析成key:config的形式：[M_N_K]:[config]
    for key, value in cachedata.items():
        for sub_key, sub_value in value.items():
            configs_key= f"{sub_key}_{key}"
            configs_value={
                'SPLIT_K': int(sub_value["SPLIT_K"]),
                'BLOCK_SIZE_M': int(sub_value["BLOCK_SIZE_M"]),
                'BLOCK_SIZE_N': int(sub_value["BLOCK_SIZE_N"]),
                'BLOCK_SIZE_K': int(sub_value["BLOCK_SIZE_K"]),
                'GROUP_SIZE_M': int(sub_value["GROUP_SIZE_M"]),
                'num_stages':int(sub_value['num_stages']),
                'num_warps':int(sub_value['num_warps'])
            }
            if 'num_ldmatrixes' in sub_value:
                configs_value["num_ldmatrixes"] = int(sub_value['num_ldmatrixes'])
            triton_configs_dict[configs_key]=configs_value
    logger.info("%s have loaded!", file_path)

def default_execution(k,n):
    configs_key= f"1_{n}_{k}"
    if configs_key in triton_configs_dict:
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_json_file=f"{script_dir}/configs/awq/"
    device_name = current_platform.get_device_name().replace(" ", "_")
    filename = f"AWQ_{n}_{k}_{device_name}.json"
    file_full_path = os.path.join(cache_json_file, filename)

    if os.path.isfile(file_full_path) and file_full_path.endswith(".json"):
        # 如果是文件，则添加到列表
        get_triton_cache(file_full_path)
    return


def getspec_config(M,N,K):
    m_config = M
    if M > 16:
        # 直接计算 2 的幂
        m_config = 1
        while m_config < M:
            m_config *= 2
    if f"{m_config}_{N}_{K}" in triton_configs_dict:
        return triton_configs_dict[f"{m_config}_{N}_{K}"]
    else:
        return None  


class AWQShareWorkSpace:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AWQShareWorkSpace, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.awqworkshapcesize = ops.GetAWQShareWorkspaceSize()
        self.awqworkshapce = ops.GetAWQShareWorkspace()


logger = init_logger(__name__)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_name(self) -> "QuantizationMethods":
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # Lazy import to avoid circular import.
            from .awq_marlin import AWQMarlinConfig
            from .moe_wna16 import MoeWNA16Config
            from .utils.marlin_utils import check_moe_marlin_supports_layer

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                config = {
                    "quant_method": "awq",
                    "bits": self.weight_bits,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "lm_head": False,
                    "modules_to_not_convert": self.modules_to_not_convert,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )
            marlin_compatible_config_dict = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            awq_marlin_config = AWQMarlinConfig.from_config(
                marlin_compatible_config_dict
            )
            return awq_marlin_config.get_quant_method(layer, prefix)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config
        self.awqsingleton= AWQShareWorkSpace()
        self.use_awq_pad = os.environ.get('AWQ_PAD') == '1'

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        zeros_and_scales = GroupQuantScaleParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)
        layer.register_parameter("zeros_and_scales", zeros_and_scales)
        # 加载triton_config
        if envs.VLLM_USE_TRITON_AWQ:
            default_execution(input_size_per_partition,output_size_per_partition)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not envs.VLLM_USE_TRITON_AWQ:
            
            group_size= self.quant_config.group_size 
            pad_group=2 
            dim_n = layer.scales.data.shape[1]
            dim_k = layer.qweight.data.shape[0]
            _qw, _sz=ops.convert_s4(layer.qweight,layer.qzeros,layer.scales.to(torch.float16),int(group_size)) 
            sz = ops.sz_permute(_sz).reshape(-1,dim_n)  
            sz = sz.reshape(dim_n,-1)
            _qw = _qw.reshape(dim_n,-1)
            
            if dim_k % 4096==0 and self.use_awq_pad:
                zeros_and_scalse_pad = torch.zeros(dim_n,pad_group,dtype=torch.int32).cuda()
                sz = torch.cat((sz,zeros_and_scalse_pad),dim=1).contiguous()
                qweight_pad = torch.zeros(dim_n,int(group_size//4),dtype=torch.int32).cuda()
                _qw=torch.cat((_qw,qweight_pad),dim=1).contiguous()
                        
            layer.qweight = torch.nn.Parameter(_qw, requires_grad=False)
            layer.zeros_and_scales = torch.nn.Parameter(sz, requires_grad=False)
            layer.qzeros = None
            layer.scales = None
        else:

            layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
            layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
            layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        zeros_and_scales = layer.zeros_and_scales
        qzeros = layer.qzeros
        scales = layer.scales
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[0] * 1, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        
        m = reshaped_x.shape[0]
        k = reshaped_x.shape[-1]
        n = qweight.shape[0]
        
        if self.use_awq_pad:
            if k % 4096 == 0:
                padding_group=2
            else:
                padding_group=0
        else:
            padding_group=0

        if envs.VLLM_USE_TRITON_AWQ:
            best_config=getspec_config(m,n,k)
            out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, pack_factor, best_config)    
            out_shape = (x.shape[:-1] + (qweight.shape[1] * 8, ))
        else:
            # out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
            out = torch.ops.vllm.awq_gemm(reshaped_x,
                            qweight,
                            zeros_and_scales,
                            m,
                            n,
                            k,
                            self.quant_config.group_size,
                            padding_group,
                            self.awqsingleton.awqworkshapce,
                            self.awqsingleton.awqworkshapcesize)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)