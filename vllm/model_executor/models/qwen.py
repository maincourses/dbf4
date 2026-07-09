# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""

import json
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

import os
import re

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm import _custom_ops as ops
from vllm.model_executor.utils import pad_weight, gemm_bank_conf


class QWenMLP(nn.Module):
    """MLP for the language component of the Qwen model, which contains a
    MergedColumnParallelLinear merging 2 outputs via silu activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_parameters: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.quant_method = None
        if quant_config is not None:
            self.quant_method=quant_config.get_name()
            self.quant_config=quant_config

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        # if os.environ.get('FA_PAD') == '1' and self.quant_method is None:
        #     qkv = qkv[...,:-32]
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class QWenModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: QWenBlock(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class QWenBaseModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[QWenModel] = QWenModel,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.transformer = transformer_type(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )
        self.quant_method = None
        if quant_config is not None:
            self.quant_method=quant_config.get_name()
            self.quant_config=quant_config
              
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'
        self.use_gemm_pad = os.environ.get('GEMM_PAD') == '1'
        self.use_fa_pad = os.environ.get('FA_PAD') == '1'
        self.use_awq_pad = os.environ.get('AWQ_PAD') == '1'

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if self.use_llama_nn and self.quant_method is None :
            lay_key_words = [
                "attn.c_attn.weight",
                "attn.c_proj.weight",
                "mlp.gate_up_proj.weight",
                "mlp.c_proj.weight",
                "lm_head.weight"
            ]
            combined_words = "|".join(lay_key_words)
            
            # lay_qkv_words = ["attn.c_attn.weight"]   
            # qkv_words = "|".join(lay_qkv_words)  
            
            # lay_qkv_bias_words = ["attn.c_attn.bias"]   
            # qkv_bias_words = "|".join(lay_qkv_bias_words) 
                      
            for layername in loaded_params:
                weight = params_dict[layername]
                # if self.use_fa_pad and (re.findall(qkv_bias_words, layername)):
                #     weight.data = pad_weight(weight.data, 32)
                
                matches = re.findall(combined_words, layername)
                if matches:         
                    # if self.use_gemm_pad and gemm_bank_conf(weight.data.shape[0]):
                    #     weight.data = pad_weight(weight.data, 32)  
                        
                    # if self.use_fa_pad and (re.findall(qkv_words, layername)):
                    #     if not gemm_bank_conf(weight.data.shape[0]):
                    #         weight.data = pad_weight(weight.data, 32)
                        
                    _weight = torch.zeros_like(weight.data)
                    ori_shape =_weight.shape
                    
                    ops.trans_w16_gemm(_weight, weight.data, _weight.shape[0], _weight.shape[1])
                    weight.data.copy_(_weight)
                    
                    weight.data=weight.data.reshape(ori_shape[1],-1)
            
        return loaded_params


class QWenLMHeadModel(QWenBaseModel, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "c_attn": ["c_attn"],
        "gate_up_proj": [
            "w2",
            "w1",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if hasattr(config, "visual"):
            hf_overrides = {"architectures": ["QwenVLForConditionalGeneration"]}
            raise RuntimeError(
                "The configuration of this model indicates that it supports "
                "vision inputs, but you instantiated the text-only version "
                "of this model. Please use the vision model by setting "
                f"`--hf-overrides '{json.dumps(hf_overrides)}'`"
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states
