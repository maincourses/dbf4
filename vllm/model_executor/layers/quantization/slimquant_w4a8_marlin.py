from typing import Any, Callable, Dict, List, Optional
import os
import torch
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import get_tensor_model_parallel_world_size
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.linear import (LinearBase,LinearMethodBase)
from vllm.model_executor.layers.quantization.base_config import (QuantizationConfig,
                                                                 QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.w4a8_utils import w4a8_weight_repack_impl
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod, SlimQuantW4A8Int8AiterMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_cache
try:
    from lmslim.layers.fused_moe.fuse_moe_w4a8_marlin import fused_experts_impl_w4a8_marlin
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n")

class MarlinMoeWorkspace:
    """
    Singleton manager for device-specific workspace buffers used by w4a8 Marlin-MoE.
    global_reduce_buffer will take 1.5MB * cus (about 120MB for BW200) memoery in each device
    """
    _instances = {}
    def __new__(cls, device):
        if device not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[device] = instance
        return cls._instances[device]

    def __init__(self, device):
        if self._initialized:
            return
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(
            500, dtype=torch.int, device=device, requires_grad=False
        )
        self.global_reduce_buffer = torch.zeros(
            sms * 6 * 128 * 512, dtype=torch.int, device=device, requires_grad=False
        )
        self._initialized = True

    def get_buffers(self):
        return self.workspace, self.global_reduce_buffer

def baseline_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    scales= scale_a* scale_b.T
    gemmout= torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales *gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


class SlimQuantW4A8Int8MarlinConfig(QuantizationConfig):
    """Config class for W4A8 Int8 Quantization.
    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8_marlin"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlimQuantW4A8Int8MarlinConfig":
        return cls()
    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" \
                and user_quant in ("slimquant_w4a8_marlin", "slimquant_marlin"):
            return cls.get_name()
        return None
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            return SlimQuantW4A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if envs.VLLM_ROCM_USE_AITER_MOE:
                return SlimQuantW4A8Int8AiterMoEMethod(self, layer.moe_config)
            else:
                return SlimQuantW4A8Int8MarlinMoEMethod(self, layer.moe_config)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SlimQuantW4A8Int8MarlinMoEMethod:
    """MoE method for W4A8INT8 Marlin.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config, moe):
        self.moe = moe
        self.quant_config = quant_config
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.moe_mk: Optional[FusedMoEModularKernel] = None
        
    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) :
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size//2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size//2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

        layer.w13_weight = Parameter(w4a8_weight_repack_impl(layer.w13_weight), requires_grad=False)
        layer.w2_weight = Parameter(w4a8_weight_repack_impl(layer.w2_weight), requires_grad=False)

    def apply(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            use_nn_moe: bool | None = False,
            use_fused_gate: bool | None = False,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
            shared_output: torch.Tensor | None = None,
            routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        return fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            activation=layer.activation,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            moe_cache_getter=get_moe_cache if envs.VLLM_USE_GLOBAL_CACHE13 else None,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor,
        )
