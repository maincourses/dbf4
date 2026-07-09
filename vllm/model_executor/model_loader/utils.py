# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import os
import torch
from torch import nn
from typing_extensions import assert_never

from vllm.attention.layer import Attention, MLAAttention
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.utils.platform_utils import is_pin_memory_available
from vllm import envs

logger = init_logger(__name__)


def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = (
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        return model_class(**kwargs)


def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    if getattr(model, "process_weights_after_loading_already_called", False):
        # In case `process_weights_after_loading` is called multiple times
        # we'll skip it at later times
        logger.debug_once(
            "process_weights_after_loading already called for model %s", model
        )
        return

    # to avoid circular dependency
    from vllm.model_executor.model_loader.online_quantization import (
        maybe_save_metadata_and_attributes_for_weight_reloading,
    )

    maybe_save_metadata_and_attributes_for_weight_reloading(model, model_config)

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    architectures = getattr(model_config.hf_config, "architectures", [])
    visions = getattr(model_config.hf_config, "visual", []) or getattr(model_config.hf_config, "vision_config", [])
    # TODO: 'Qwen2_5_VLForConditionalGeneration', 
    support_nn_architectures = ['LlamaForCausalLM', 'QWenLMHeadModel', 'Qwen2ForCausalLM', 'Qwen2VLForConditionalGeneration', 'Qwen2MoeForCausalLM', 'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM',
                                'ChatGLMModel', 'Glm4ForCausalLM', 'ChatGLMForConditionalGeneration', 'BaichuanForCausalLM', 'BloomForCausalLM', 'TeleChat2ForCausalLM', 'MixtralForCausalLM', 'FalconForCausalLM',
                                'MedusaModel', 'MLPSpeculatorPreTrainedModel', 'DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'DeepSeekMTPModel']  
    if any(arch in architectures for arch in support_nn_architectures): 
        if not envs.VLLM_USE_NN:
            if os.getenv('LLAMA_NN') != '0': 
                if (architectures == ['QWenLMHeadModel'] or architectures == ['ChatGLMModel'] ) and visions != []:
                    os.environ['LLAMA_NN'] = '0'
                else:
                    os.environ['LLAMA_NN'] = '1'
            if (architectures == ['BloomForCausalLM'] or architectures == ['FalconForCausalLM']) or os.getenv('LM_NN') == '0':
                os.environ['LM_NN'] = '0'
            else:
                os.environ['LM_NN'] = '1'
                
            if architectures in [['DeepseekV3ForCausalLM'], ['DeepSeekMTPModel']]:
                if not envs.is_set("VLLM_USE_LIGHTOP"):
                    os.environ['VLLM_USE_LIGHTOP'] = '1'
                if not envs.is_set("VLLM_USE_OPT_CAT"):
                    os.environ['VLLM_USE_OPT_CAT'] = '1'
                if not envs.is_set("VLLM_USE_FUSED_FILL_RMS_CAT"):
                    os.environ['VLLM_USE_FUSED_FILL_RMS_CAT'] = '1'
                if model_config.quantization in {"slimquant_w4a8", "slimquant_w4a8_marlin", "slimquant_compressed_tensors_marlin", "compressed-tensors"}:
                    if not envs.is_set("VLLM_USE_LIGHTOP_FILL_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_FILL_MOE_ALIGN'] = '1'
                # if model_config.quantization in {"slimquant_w4a8", "slimquant_w4a8_marlin", "slimquant_compressed_tensors_marlin", "compressed-tensors"}:
                #     if not envs.is_set("USE_FUSED_RMS_QUANT"):
                #         os.environ['USE_FUSED_RMS_QUANT'] = '1'
                #     if not envs.is_set("USE_FUSED_SILU_MUL_QUANT"):
                #         os.environ['USE_FUSED_SILU_MUL_QUANT'] = '1'
            else:
                if not envs.is_set("VLLM_USE_PD_SPLIT"):
                    os.environ['VLLM_USE_PD_SPLIT'] = '1'
                if architectures in [['Qwen3MoeForCausalLM']]:
                    if not envs.is_set("VLLM_USE_LIGHTOP_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_MOE_ALIGN'] = '1'
                    if not envs.is_set("VLLM_USE_LIGHTOP_FILL_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_FILL_MOE_ALIGN'] = '1'
                    if not envs.is_set("VLLM_USE_LIGHTOP_MOE_SUM"):
                        os.environ['VLLM_USE_LIGHTOP_MOE_SUM'] = '1'    
                    if not envs.is_set("VLLM_USE_FUSE_SILU_AND_MUL"):
                        os.environ['VLLM_USE_FUSE_SILU_AND_MUL'] = '1'
                    if not envs.is_set("VLLM_USE_OPT_RESHAPE_AND_CACHE"):
                        os.environ['VLLM_USE_OPT_RESHAPE_AND_CACHE'] = '1'
                # if architectures in [['Qwen3ForCausalLM']]:
                #     if not envs.is_set("VLLM_USE_OPT_RESHAPE_AND_CACHE"):
                #         os.environ['VLLM_USE_OPT_RESHAPE_AND_CACHE'] = '0'

            if architectures in [['DeepseekV32ForCausalLM']]:
                if not envs.is_set("VLLM_USE_V32_ENCODE"):
                    os.environ['VLLM_USE_V32_ENCODE'] = '1'
                    
            if os.getenv('GEMM_PAD') != '1': 
                os.environ['GEMM_PAD'] = '0'
            if os.getenv('FA_PAD') != '1': 
                os.environ['FA_PAD'] = '0'
        else:
            if architectures in [['DeepseekV3ForCausalLM'], ['DeepSeekMTPModel']]:
                if not envs.is_set("VLLM_USE_LIGHTOP"):
                    os.environ['VLLM_USE_LIGHTOP'] = '1'
                if not envs.is_set("VLLM_USE_OPT_CAT"):
                    os.environ['VLLM_USE_OPT_CAT'] = '1'
                if not envs.is_set("VLLM_USE_FUSED_FILL_RMS_CAT"):
                    os.environ['VLLM_USE_FUSED_FILL_RMS_CAT'] = '1'
                if model_config.quantization in {"slimquant_w4a8", "slimquant_w4a8_marlin", "slimquant_compressed_tensors_marlin", "compressed-tensors"}:
                    if not envs.is_set("VLLM_USE_LIGHTOP_FILL_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_FILL_MOE_ALIGN'] = '1'
                # if model_config.quantization in {"slimquant_w4a8", "slimquant_w4a8_marlin", "slimquant_compressed_tensors_marlin", "compressed-tensors"}:
                #     if not envs.is_set("USE_FUSED_RMS_QUANT"):
                #         os.environ['USE_FUSED_RMS_QUANT'] = '1'
                #     if not envs.is_set("USE_FUSED_SILU_MUL_QUANT"):
                #         os.environ['USE_FUSED_SILU_MUL_QUANT'] = '1'
            else:
                if not envs.is_set("VLLM_USE_PD_SPLIT"):
                    os.environ['VLLM_USE_PD_SPLIT'] = '1'
                if architectures in [['Qwen3MoeForCausalLM']]:
                    if not envs.is_set("VLLM_USE_LIGHTOP_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_MOE_ALIGN'] = '1'
                    if not envs.is_set("VLLM_USE_LIGHTOP_FILL_MOE_ALIGN"):
                        os.environ['VLLM_USE_LIGHTOP_FILL_MOE_ALIGN'] = '1'
                    if not envs.is_set("VLLM_USE_LIGHTOP_MOE_SUM"):
                        os.environ['VLLM_USE_LIGHTOP_MOE_SUM'] = '1'    
                    if not envs.is_set("VLLM_USE_FUSE_SILU_AND_MUL"):
                        os.environ['VLLM_USE_FUSE_SILU_AND_MUL'] = '1'
                    if not envs.is_set("VLLM_USE_OPT_RESHAPE_AND_CACHE"):
                        os.environ['VLLM_USE_OPT_RESHAPE_AND_CACHE'] = '1'
            
            if architectures in [['DeepseekV32ForCausalLM']]:
                if not envs.is_set("VLLM_USE_V32_ENCODE"):
                    os.environ['VLLM_USE_V32_ENCODE'] = '1'
                    
        # awq相关配置
        try:
            if os.getenv('AWQ_PAD') == None and (torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count == 120):
                os.environ['AWQ_PAD'] = '1'
        except Exception as e:
            if os.getenv('AWQ_PAD') != '0': 
                os.environ['AWQ_PAD'] = '1'
            else:
                os.environ['AWQ_PAD'] = '0'
    else:
        os.environ['LLAMA_NN'] = '0'
        os.environ['LM_NN'] = '0'
        os.environ['GEMM_PAD'] = '0'
        os.environ['FA_PAD'] = '0'
        os.environ['AWQ_PAD'] = '0'

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "vllm"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )

    convert_type = model_config.convert_type
    if convert_type == "none":
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            model_config.quantization,
            tuple(getattr(model_config.hf_config, "architectures", [])),
        )
    )
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    model_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    """

    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
