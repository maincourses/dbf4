# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts
else:
    TritonExperts = None  # type: ignore


logger = init_logger(__name__)


# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend = select_unquantized_moe_backend(
            use_ep=self.moe.moe_parallel_config.use_ep,
            use_dp=self.moe.moe_parallel_config.dp_size > 1,
        )

        # AITER only supports gated activations (silu/gelu), so disable it
        # for non-gated MoE (is_act_and_mul=False)
        self.rocm_aiter_moe_enabled = (
            rocm_aiter_ops.is_fused_moe_enabled() and moe.is_act_and_mul
        )
        self.kernel: mk.FusedMoEModularKernel | None = None
        self._is_monolithic = current_platform.is_cpu() or current_platform.is_xpu()

    @property
    def is_monolithic(self) -> bool:
        return self._is_monolithic

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        if self.unquantized_backend == UnquantizedMoeBackend.AITER:
            return None
        else:
            return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        use_nn_moe: bool,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        if not use_nn_moe:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_up_dim,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    w13_up_dim,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        if not use_nn_moe:
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    intermediate_size_per_partition,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()

        return weight

    def _setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2 = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        # Setup Modular Kernel for TP Case
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        self.kernel, self.use_inplace = make_unquantized_moe_kernel(
            backend=self.unquantized_backend,
            quant_config=self.moe_quant_config,
            moe_config=self.moe,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # If Marlin W16A16 MoE is supported, pre-pack weights once during the
        # post-load hook and replace parameters with the packed layout.
        #
        # This avoids first-run packing peaks during KV cache profiling and
        # keeps only one copy of weights resident on GPU in steady state.
        if (
            getattr(layer, "_marlin_w16a16_moe_enabled", False)
            and current_platform.is_cuda_alike()
            and not getattr(layer, "use_nn_moe", False)
            and not getattr(layer, "_marlin_w16a16_moe_packed", False)
        ):
            w1 = layer.w13_weight
            w2 = layer.w2_weight
            if (
                w1.is_cuda
                and w2.is_cuda
                and w1.dtype in (torch.float16, torch.bfloat16)
                and w2.dtype in (torch.float16, torch.bfloat16)
            ):
                try:
                    if w1.dim() != 3 or w2.dim() != 3 or w1.size(0) != w2.size(0):
                        raise RuntimeError("Unexpected MoE weight shapes")

                    twoN, K = w1.size(1), w1.size(2)
                    if w2.size(1) != K:
                        raise RuntimeError("Unexpected MoE w2 layout")

                    N = w2.size(2)
                    if twoN != 2 * N:
                        raise RuntimeError("Unexpected MoE hidden dims")

                    if K % 32 != 0 or N % 16 != 0:
                        raise RuntimeError("Marlin packing requires alignment")

                    from torch.nn.parameter import Parameter

                    from vllm.model_executor.layers.fused_moe.marlin_quant import (
                        w16a16_marlin_weight,
                    )

                    def _pack_per_expert(weight: torch.Tensor) -> torch.Tensor:
                        num_experts = weight.shape[0]
                        packed0 = w16a16_marlin_weight(weight[0]).contiguous()
                        packed = packed0.new_empty((num_experts,) + packed0.shape)
                        packed[0].copy_(packed0)
                        del packed0

                        for i in range(1, num_experts):
                            tmp = w16a16_marlin_weight(weight[i]).contiguous()
                            packed[i].copy_(tmp)
                            del tmp
                        return packed

                    with torch.no_grad():
                        w1_packed = _pack_per_expert(w1)
                        w2_packed = _pack_per_expert(w2)

                        new_w1 = Parameter(w1_packed, requires_grad=False)
                        new_w2 = Parameter(w2_packed, requires_grad=False)

                        # Preserve any custom weight attributes (e.g. loaders).
                        if hasattr(w1, "__dict__"):
                            for k, v in w1.__dict__.items():
                                setattr(new_w1, k, v)
                        if hasattr(w2, "__dict__"):
                            for k, v in w2.__dict__.items():
                                setattr(new_w2, k, v)

                        setattr(new_w1, "marlin_w16a16_packed", True)
                        setattr(new_w2, "marlin_w16a16_packed", True)

                        layer.w13_weight = new_w1
                        layer.w2_weight = new_w2
                        layer._marlin_w16a16_moe_packed = True

                    return
                except Exception:
                    # If packing dependencies are unavailable, fall back to the
                    # standard (non-Marlin) layouts.
                    pass

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        if self.unquantized_backend == UnquantizedMoeBackend.XPU:
            import intel_extension_for_pytorch as ipex

            ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
            self.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
                experts_start_id=ep_rank_start,
            )
        elif self.unquantized_backend == UnquantizedMoeBackend.CPU:
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (
                    envs.VLLM_CPU_SGL_KERNEL
                    and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                    and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)
                ):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight
                    )
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight
                    )
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    self.cpu_fused_moe: Callable = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
            else:
                self.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
        elif current_platform.is_cuda_alike():
            self._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )

    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_nn_moe: bool | None = False,
        shared_output: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        **_,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_nn_moe=use_nn_moe,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_nn_moe: bool | None = False,
        use_fused_gate: bool | None = False,
        shared_output: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            getattr(layer, "_marlin_w16a16_moe_enabled", False)
            and getattr(layer, "_marlin_w16a16_moe_packed", False)
        ):
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=self.allow_inplace,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=self.get_fused_moe_quant_config(layer),
                use_nn_moe=use_nn_moe,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
            )

        assert self.kernel is not None
        return self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            use_nn_moe=use_nn_moe,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor,
        )

    def forward_monolithic_cpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.cpu_fused_moe(
            layer,
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            layer.global_num_experts,
            layer.expert_map,
            layer.custom_routing_function,
            layer.scoring_func,
            layer.routed_scaling_factor,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.activation,
        )

    def forward_monolithic_xpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
        )

    if current_platform.is_cpu():
        forward_native: Callable = forward_monolithic_cpu
        apply_monolithic = forward_monolithic_cpu
    elif current_platform.is_xpu():
        forward_native = forward_monolithic_xpu
        apply_monolithic = forward_monolithic_xpu
    else:
        forward_native = forward_cuda
