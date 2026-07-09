# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum
from typing import Callable, Optional
import torch
from compressed_tensors.quantization import (QuantizationStrategy)

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from torch.nn.parameter import Parameter
from vllm.distributed import get_ep_group, get_dp_group
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEMethodBase,
    FusedMoeWeightScaleSupported, FusedMoEConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.utils.w8a8_utils import(
    get_w8a8_int8_marlin_weights, w8a8_nt_kpack2_marlin_weight, weight8bit_nt_kpack2_marlin1)
from vllm.model_executor.layers.fused_moe.config import (FusedMoEQuantConfig, int8_w8a8_moe_quant_config, fp8_w8a8_moe_quant_config)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported,
)
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.ops.shuffle import moe_layout_shuffle_gemm1, moe_layout_shuffle_gemm2
from aiter.fused_moe import fused_topk, torch_moe
from aiter import dtypes, ActivationType
from aiter.moe import get_aiter_moe_config, aiter_moe, MoeSolutionType, MoeQuantType
try:
    from lmslim.layers.fused_moe.fuse_moe_int8_marlin import fused_experts_impl_int8_marlin
    from lmslim.layers.fused_moe.fuse_moe_fp8_marlin import fused_experts_impl_fp8_marlin
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n")

logger = init_logger(__name__)

__all__ = [
    "CompressedTensorsW8A8Int8MarlinMoEMethod",
    "CompressedTensorsW8A8FP8MarlinMoEMethod",
]


def fp32_to_fp8_e4m3fn(t: torch.Tensor) -> torch.Tensor:
    """更合理的FP32到Float8_e4m3fn转换，使用最近值而不是简单舍弃尾数"""
    # torch.float8_e4m3fn的数值范围约[-448, 448]
    fp8_min, fp8_max = -448.0, 448.0
    t_clamped = t.clamp(min=fp8_min, max=fp8_max)
    # 保证不会下溢到0
    # 转换前到float16再转fp8可能提升精度（float8实现本身通常通过float16做rounding）
    t_fp16 = t_clamped.to(torch.float16)
    return t_fp16.to(torch.float8_e4m3fn)


def w8a8_fp8_nt_kpack2_marlin_weight(w8a8_w,  # [size_n, size_k// 2 ]
                                     k_tile=16,
                                     n_tile=16, ):
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    w8a8_w = w8a8_w.reshape((size_n // n_tile, n_tile, size_k // k_tile, k_tile))
    w8a8_w = w8a8_w.permute((0, 2, 1, 3)).contiguous()
    w8a8_w = w8a8_w.reshape((size_n // k_tile, size_k * k_tile))
    return w8a8_w

class CompressedTensorsMarlinMoEMethod(FusedMoEMethodBase):
    def __init_(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
        quant_config: "SlimQuantCompressedTensorsMarlinConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMarlinMoEMethod":

        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        if quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8FP8MarlinMoEMethod(quant_config, layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MarlinMoEMethod(quant_config, layer.moe_config)
        else:
            raise RuntimeError(
                f"Slimquant_marlin does not support the FusedMoe scheme: {weight_quant}, {input_quant}")


class CompressedTensorsW8A8FP8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
            self,
            quant_config: "CompressedTensorsMarlinConfig",  # type: ignore # noqa E501
            moe: FusedMoEConfig
    ):
        self.quant_config = quant_config
        super().__init__(moe)
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
                self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For FP8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For FP8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")
        self.fused_experts = self.fused_moe_forward

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dp_size = get_dp_group().world_size
        self.use_deepep = self.dp_size > 1 and parallel_config.enable_expert_parallel and \
            (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput" or \
             envs.VLLM_ALL2ALL_BACKEND == "deepep_low_latency" or \
             envs.VLLM_ALL2ALL_BACKEND == "deepep_auto")
        
        if self.use_deepep:
            all2all_manager = get_ep_group().device_communicator.all2all_manager
            assert all2all_manager is not None
            self.num_dispatchers = all2all_manager.world_size

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=[256, 256] if self.use_deepep else None,
        )

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        
        if self.use_deepep:
            self.N = 2 * intermediate_size_per_partition
            self.K = hidden_size
        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def shuffle_w8a8_gemm1(self, weight_data):
        w_fp8 = weight_data.to(torch.float8_e4m3fn)
        shuffled = moe_layout_shuffle_gemm1(w_fp8)
        return shuffled.view(torch.int8)

    def shuffle_w8a8_gemm2(self, weight_data):
        w_fp8 = weight_data.to(torch.float8_e4m3fn)
        shuffled = moe_layout_shuffle_gemm2(w_fp8)
        return shuffled.view(torch.int8)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if envs.VLLM_USE_AITER_MOE_W8A8==True:
            layer.w13_weight_scale = Parameter(layer.w13_weight_scale.data, requires_grad=False)
            layer.w2_weight_scale = Parameter(layer.w2_weight_scale.data, requires_grad=False)   

            shuffled_w13 = self.shuffle_w8a8_gemm1(layer.w13_weight)
            w13_data = shuffled_w13.view(*layer.w13_weight.shape).view(torch.int8)
            layer.w13_weight = Parameter(w13_data, requires_grad=False)

            shuffled_w2 = self.shuffle_w8a8_gemm2(layer.w2_weight)
            w2_data = shuffled_w2.view(*layer.w2_weight.shape).view(torch.int8)
            layer.w2_weight = Parameter(w2_data, requires_grad=False)
        else:
            w1_marlin_list = []
            for ii in range(layer.w13_weight.shape[0]):
                if not self.use_deepep:
                    w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
                else:
                    w1_marlin_in = weight8bit_nt_kpack2_marlin1(layer.w13_weight[ii])
                w1_marlin_list.append(w1_marlin_in.float() if w1_marlin_in.dtype == torch.float8_e4m3fn else w1_marlin_in)
            w1_marlin = torch.stack(w1_marlin_list, dim=0)
            w1_marlin = fp32_to_fp8_e4m3fn(w1_marlin)

            del w1_marlin_list
            w2_marlin_list = []
            for ii in range(layer.w2_weight.shape[0]):
                if not self.use_deepep:
                    w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
                else:
                    w2_marlin_in = weight8bit_nt_kpack2_marlin1(layer.w2_weight[ii])
                w2_marlin_list.append(w2_marlin_in.float() if w2_marlin_in.dtype == torch.float8_e4m3fn else w2_marlin_in)
            w2_marlin = torch.stack(w2_marlin_list, dim=0)
            w2_marlin = fp32_to_fp8_e4m3fn(w2_marlin)
            layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
            layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    def fused_moe_forward(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            routed_scaling_factor: Optional[float] = None,
            shared_output: Optional[torch.Tensor] = None,
    ):
        if envs.VLLM_USE_AITER_MOE_W8A8==True:
            m_flat = x.view(-1, x.shape[-1])
            M = m_flat.shape[0]
            E = layer.w13_weight.size(0)
            K = x.size(-1)
            N1 = layer.w13_weight.size(1)
            topk = topk_ids.size(1)
            w1_input = layer.w13_weight.view(E, N1, K)
            w2_input = layer.w2_weight.view(E, K, N1 // 2)

            _, moe_cfg = get_aiter_moe_config(
                M=M,
                E=E,
                N1=N1,
                N2=N1 // 2,
                K=K,
                top_k=topk,
                block_size=0,
                dtype=x.dtype,
                quant_type=MoeQuantType.W8A8,
            )
            output = aiter_moe(
                hidden_states=x,
                w1=w1_input,
                w2=w2_input,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=moe_cfg,
                inplace=False,
                activation=getattr(layer, "activation", "silu"),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
                global_num_experts=E,
                expert_map=getattr(layer, "expert_map", None),
                routed_scaling_factor=routed_scaling_factor,
            )
            return output
        else:
            return fused_experts_impl_fp8_marlin(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=True,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                use_nn_moe=False,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            enable_eplb: bool = False,
            use_nn_moe: Optional[bool] = False,
            routed_scaling_factor: Optional[float] = None,
            use_fused_gate: Optional[bool] = False,
            expert_load_view: Optional[torch.Tensor] = None,
            logical_to_physical_map: Optional[torch.Tensor] = None,
            logical_replica_count: Optional[torch.Tensor] = None,
            shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for "
                "`CompressedTensorsW8A8Int8MoEMethod` yet.")

        return self.fused_experts(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output, )
    
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
            DeepGemmExperts,
        )

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            logger.debug("BatchedDeepGemmExperts(%s)", self.__class__.__name__)
            return BatchedDeepGemmExperts(
                moe_config=self.moe,
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
                N=self.N,
                K=self.K
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(moe_config=self.moe,
                                   quant_config=self.moe_quant_config,
                                   N=self.N,
                                   K=self.K)
    
class CompressedTensorsW8A8Int8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
            self,
            quant_config: "CompressedTensorsMarlinConfig",  # type: ignore # noqa E501
            moe: FusedMoEConfig
    ):
        self.quant_config = quant_config
        super().__init__(moe)
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")
        
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dp_size = get_dp_group().world_size
        self.use_deepep = self.dp_size > 1 and parallel_config.enable_expert_parallel and \
            (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput" or \
             envs.VLLM_ALL2ALL_BACKEND == "deepep_low_latency" or \
             envs.VLLM_ALL2ALL_BACKEND == "deepep_auto")
        
        if self.use_deepep:
            all2all_manager = get_ep_group().device_communicator.all2all_manager
            assert all2all_manager is not None
            self.num_dispatchers = all2all_manager.world_size
    
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            block_shape=[256, 256] if self.use_deepep else None,
        )
        
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        
        if self.use_deepep:
            self.N = 2 * intermediate_size_per_partition
            self.K = hidden_size

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def shuffle_w8a8_gemm1(self, weight_data):
        w_i8 = weight_data.to(torch.int8)
        return moe_layout_shuffle_gemm1(w_i8)

    def shuffle_w8a8_gemm2(self, weight_data):
        w_i8 = weight_data.to(torch.int8)
        return moe_layout_shuffle_gemm2(w_i8)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if envs.VLLM_USE_AITER_MOE_W8A8==True:
            layer.w13_weight_scale = Parameter(layer.w13_weight_scale.data, requires_grad=False)
            layer.w2_weight_scale = Parameter(layer.w2_weight_scale.data, requires_grad=False)     
            shuffled_w13 = self.shuffle_w8a8_gemm1(layer.w13_weight)
            layer.w13_weight = Parameter(shuffled_w13.view(*layer.w13_weight.shape),  requires_grad=False)
            shuffled_w2 = self.shuffle_w8a8_gemm2(layer.w2_weight)
            layer.w2_weight = Parameter(shuffled_w2.view(*layer.w2_weight.shape), requires_grad=False)       
        else:
            w1_marlin_list = []
            for ii in range(layer.w13_weight.shape[0]):
                if not self.use_deepep:
                    w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
                else:
                    w1_marlin_in = w8a8_nt_kpack2_marlin_weight(layer.w13_weight[ii])
                w1_marlin_list.append(w1_marlin_in)
            w1_marlin = torch.stack(w1_marlin_list, dim=0)

            del w1_marlin_list
            w2_marlin_list = []
            for ii in range(layer.w2_weight.shape[0]):
                if not self.use_deepep:
                    w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
                else:
                    w2_marlin_in = w8a8_nt_kpack2_marlin_weight(layer.w2_weight[ii])
                w2_marlin_list.append(w2_marlin_in)
            w2_marlin = torch.stack(w2_marlin_list, dim=0)

            layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
            layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_nn_moe: bool | None = False,
        i_q: torch.Tensor | None = None,
        i_s: torch.Tensor | None = None,
        shared_output: Optional[torch.Tensor] = None,
        routed_scaling_factor: Optional[float] = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        if envs.VLLM_USE_AITER_MOE_W8A8==True:
            m_flat = x.view(-1, x.shape[-1])
            M = m_flat.shape[0]
            E = layer.w13_weight.size(0)
            K = x.size(-1)
            N1 = layer.w13_weight.size(1)
            topk = topk_ids.size(1)
            w1_input = layer.w13_weight.view(E, N1, K)
            w2_input = layer.w2_weight.view(E, K, N1 // 2)

            _, moe_cfg = get_aiter_moe_config(
                M=M,
                E=E,
                N1=N1,
                N2=N1 // 2,
                K=K,
                top_k=topk,
                block_size=0,
                dtype=x.dtype,
                quant_type=MoeQuantType.W8A8,
            )
            output = aiter_moe(
                hidden_states=x,
                w1=w1_input,
                w2=w2_input,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=moe_cfg,
                inplace=False,
                activation=getattr(layer, "activation", "silu"),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
                global_num_experts=E,
                expert_map=getattr(layer, "expert_map", None),
                routed_scaling_factor=routed_scaling_factor,
            )
            return output
        else:
            return fused_experts_impl_int8_marlin(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                use_int8_w8a8=True,
                per_channel_quant=True,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                use_nn_moe=False,
                i_q=i_q,
                i_s=i_s,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
            )
    
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
            DeepGemmExperts,
        )

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            logger.debug("BatchedDeepGemmExperts(%s)", self.__class__.__name__)
            return BatchedDeepGemmExperts(
                moe_config=self.moe,
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
                N=self.N,
                K=self.K
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(moe_config=self.moe,
                                   quant_config=self.moe_quant_config,
                                   N=self.N,
                                   K=self.K)
