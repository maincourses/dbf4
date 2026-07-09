# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re

import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.forward_context import get_forward_context
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm.distributed import (tensor_model_parallel_all_gather, 
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from .utils import maybe_prefix
from .interfaces import SupportsPP
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.blockwise_int8 import BlockInt8Config
import vllm.envs as envs
from vllm.utils import direct_register_custom_op
logger = init_logger(__name__)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.device = current_platform.device_type

        #添加判断，默认开启DSA
        force_disable_dsa = envs.VLLM_DISABLE_DSA
        self.is_v32 = hasattr(config, "index_topk") and not force_disable_dsa
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
        )
    def fuse_fill_rms_x2_concat(hidden_states_fuse: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor,
                                    previous_hidden_states: torch.Tensor, weight_inputs_embeds: torch.Tensor, 
                                    weight_previous_hidden_states: torch.Tensor, epsilon: float) -> None:
        from lightop import fuse_fill_rms_x2_concat
        fuse_fill_rms_x2_concat(hidden_states_fuse, positions, inputs_embeds, previous_hidden_states, weight_inputs_embeds, weight_previous_hidden_states, epsilon)

    def fuse_fill_rms_x2_concat_fake(hidden_states_fuse: torch.Tensor, positions: torch.Tensor, inputs_embeds: torch.Tensor,
                                    previous_hidden_states: torch.Tensor, weight_inputs_embeds: torch.Tensor, 
                                    weight_previous_hidden_states: torch.Tensor, epsilon: float) -> None:
        pass

    direct_register_custom_op(
        op_name="fuse_fill_rms_x2_concat",
        op_func=fuse_fill_rms_x2_concat,
        mutates_args=["hidden_states_fuse", "inputs_embeds"], 
        fake_impl=fuse_fill_rms_x2_concat_fake,
    )


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        if envs.VLLM_USE_FUSED_FILL_RMS_CAT:
            hidden_states_fuse = torch.empty(inputs_embeds.shape[0], inputs_embeds.shape[1]*2, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            torch.ops.vllm.fuse_fill_rms_x2_concat(hidden_states_fuse, positions, inputs_embeds, previous_hidden_states, self.enorm.weight, self.hnorm.weight, self.enorm.variance_epsilon)
            hidden_states = self.eh_proj(hidden_states_fuse)
        else:
            inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
            inputs_embeds = self.enorm(inputs_embeds)
            previous_hidden_states = self.hnorm(previous_hidden_states)

            hidden_states = self.eh_proj(
                torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
            )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp:
            scatter_indexes_tensor = get_forward_context().scatter_indexes_tensor
            if scatter_indexes_tensor is None:
                inputs_embeds_per_rank = torch.chunk(inputs_embeds, chunks=self.tp_size, dim=0)
                inputs_embeds = inputs_embeds_per_rank[self.tp_rank].contiguous()

                previous_hidden_states_per_rank = torch.chunk(previous_hidden_states, chunks=self.tp_size, dim=0)
                previous_hidden_states = previous_hidden_states_per_rank[self.tp_rank].contiguous()

                if positions is not None:
                    positions_per_rank = torch.chunk(positions, chunks=self.tp_size, dim=0)
                    positions = positions_per_rank[self.tp_rank].contiguous()
            else:
                scatter_indexes_tensor = torch.where(scatter_indexes_tensor == -1, 0, scatter_indexes_tensor)
                inputs_embeds = torch.index_select(inputs_embeds, 0, scatter_indexes_tensor)

                previous_hidden_states = torch.index_select(previous_hidden_states, 0, scatter_indexes_tensor)
                if positions is not None:
                    positions = torch.index_select(positions, 0, scatter_indexes_tensor)

        hidden_states = self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

        if enable_lightly_cp:
            hidden_states = tensor_model_parallel_all_gather(hidden_states.contiguous(), dim=0)
            gather_indexes_tensor = get_forward_context().gather_indexes_tensor
            if gather_indexes_tensor is not None:
                hidden_states = torch.index_select(hidden_states, 0, gather_indexes_tensor)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile
class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Set MoE hyperparameters
        self.set_moe_parameters()
        quant_config = vllm_config.quant_config

        self.quant_method = None
        if quant_config is not None:
            self.quant_method = quant_config.get_name()
            os.environ['LLAMA_NN'] = '0'
            os.environ['LM_NN'] = '0'
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'
                

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # 判断是否加载"indexer"权重
        model_has_indexer = any("indexer" in param_name for param_name in params_dict.keys())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            #  跳过加载"indexer"权重
            if "indexer" in name and not model_has_indexer:
                logger.info(f"Skipping indexer weight (DSA disabled): {name}")
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[
                                j * chunk_size : (j + 1) * chunk_size, :
                            ]
                        else:
                            weight_to_load = loaded_weight[
                                :, j * chunk_size : (j + 1) * chunk_size
                            ]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        # According to DeepSeek-V3 Technical Report, MTP modules
                        # shares embedding layer. We only load the first weights.
                        if (
                            spec_layer != self.model.mtp_start_layer_idx
                            and ".layers" not in name
                        ):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)
        
        if self.use_llama_nn and self.quant_method is None:
            lay_key_words = [
                "self_attn.eh_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
                "mlp.gate.weight",
                "shared_experts.gate_up_proj.weight",
                "shared_experts.down_proj.weight",
                "shared_head.head.weight",
            ]

            combined_words = "|".join(lay_key_words)
            
            for layername in loaded_params:
                weight = params_dict[layername]
                matches = re.findall(combined_words, layername)
                if matches:
                    _weight = torch.zeros_like(weight.data)
                    ori_shape =_weight.shape
                    
                    ops.trans_w16_gemm(_weight, weight.data, _weight.shape[0], _weight.shape[1])
                    weight.data.copy_(_weight)
                    
                    weight.data=weight.data.reshape(ori_shape[1],-1)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
