# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any, Literal

import os
import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION
# from ..utils import models_path_prefix

models_path_prefix = os.getenv('VLLM_OPTEST_MODELS_PATH') or os.getenv("OPTEST_MODELS_PATH")

from vllm.config.model import ModelDType, TokenizerMode


@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str
    """The default model to use for testing this architecture."""

    extras: Mapping[str, str] = field(default_factory=dict)
    """Extra models to use for testing this architecture."""

    tokenizer: str | None = None
    """Set the tokenizer to load for this architecture."""

    tokenizer_mode: TokenizerMode | str = "auto"
    """Set the tokenizer type for this architecture."""

    speculative_model: str | None = None
    """
    The default model to use for testing this architecture, which is only used
    for speculative decoding.
    """

    speculative_method: str | None = None
    """
    The method to use for speculative decoding.
    """

    min_transformers_version: str | None = None
    """
    The minimum version of HF Transformers that is required to run this model.
    """

    max_transformers_version: str | None = None
    """
    The maximum version of HF Transformers that this model runs on.
    """

    transformers_version_reason: dict[Literal["vllm", "hf"], str] | None = None
    """
    The type and reason to skip test for the minimum/maximum version requirement.
    vllm: skip all vLLM tests if the version requirement is not met.
    hf: only skip tests that uses HF runner if the version requirement is not met.
    """

    require_embed_inputs: bool = False
    """
    If `True`, enables prompt and multi-modal embedding inputs while
    disabling tokenization.
    """

    dtype: ModelDType = "auto"
    """
    The data type for the model weights and activations.
    """

    enforce_eager: bool = False
    """
    Whether to enforce eager execution. If True, we will
    disable CUDA graph and always execute the model in eager mode.
    If False, we will use CUDA graph and eager execution in hybrid.
    """

    is_available_online: bool = True
    """
    Set this to `False` if the name of this architecture no longer exists on
    the HF repo. To maintain backwards compatibility, we have not removed them
    from the main model registry, so without this flag the registry tests will
    fail.
    """

    trust_remote_code: bool = False
    """The `trust_remote_code` level required to load the model."""

    hf_overrides: dict[str, Any] = field(default_factory=dict)
    """The `hf_overrides` required to load the model."""

    max_model_len: int | None = None
    """
    The maximum model length to use for this model. Some models default to a
    length that is too large to fit into memory in CI.
    """

    max_num_batched_tokens: int | None = None
    """
    The maximum number of tokens to be processed in a single batch.
    """

    revision: str | None = None
    """
    The specific revision (commit hash, tag, or branch) to use for the model.
    If not specified, the default revision will be used.
    """

    max_num_seqs: int | None = None
    """Maximum number of sequences to be processed in a single iteration."""

    use_original_num_layers: bool = False
    """
    If True, use the original number of layers from the model config
    instead of minimal layers for testing.
    """

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"],
        check_version_reason: Literal["vllm", "hf"] = "hf",
        check_min_version: bool = True,
        check_max_version: bool = True,
    ) -> str | None:
        """
        If the installed transformers version does not meet the requirements,
        perform the given action.
        """
        if (
            self.min_transformers_version is None
            and self.max_transformers_version is None
        ):
            return None

        current_version = TRANSFORMERS_VERSION
        cur_base_version = Version(current_version).base_version
        min_version = self.min_transformers_version
        max_version = self.max_transformers_version
        msg = f"`transformers=={current_version}` installed, but `transformers"
        # Only check the base version for the min/max version, otherwise preview
        # models cannot be run because `x.yy.0.dev0`<`x.yy.0`
        if min_version and Version(cur_base_version) < Version(min_version):
            is_version_valid = not check_min_version
            msg += f">={min_version}` is required to run this model."
        elif max_version and Version(cur_base_version) > Version(max_version):
            is_version_valid = not check_max_version
            msg += f"<={max_version}` is required to run this model."
        else:
            is_version_valid = True

        # check if Transformers version breaks the corresponding model runner,
        # skip test when model runner not compatible
        is_reason_valid = not (
            check_version_reason
            and self.transformers_version_reason
            and check_version_reason in self.transformers_version_reason
        )
        is_transformers_valid = is_version_valid and is_reason_valid
        if is_transformers_valid:
            return None
        elif self.transformers_version_reason:
            for reason_type, reason in self.transformers_version_reason.items():
                msg += f" Reason({reason_type}): {reason}"

        if on_fail == "error":
            raise RuntimeError(msg)
        elif on_fail == "skip":
            pytest.skip(msg)

        return msg

    def check_available_online(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        """
        If the model is not available online, perform the given action.
        """
        if not self.is_available_online:
            msg = "Model is not available online"

            if on_fail == "error":
                raise RuntimeError(msg)
            else:
                pytest.skip(msg)


_TEXT_GENERATION_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AfmoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "arcee-ai/Trinity-Nano-Preview")),
    "ApertusForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "swiss-ai/Apertus-8B-Instruct-2509")),
    "AquilaModel": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/AquilaChat-7B"), trust_remote_code=True),
    "AquilaForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/AquilaChat2-7B"), trust_remote_code=True),
    "ArceeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "arcee-ai/AFM-4.5B-Base")),
    "ArcticForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Snowflake/snowflake-arctic-instruct"), trust_remote_code=True
    ),
    "BaiChuanForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "baichuan-inc/Baichuan-7B"), trust_remote_code=True
    ),
    "BaichuanForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "baichuan-inc/Baichuan2-7B-chat"), trust_remote_code=True
    ),
    "BailingMoeForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "inclusionAI/Ling-lite-1.5"), trust_remote_code=True
    ),
    "BailingMoeV2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "inclusionAI/Ling-mini-2.0"), trust_remote_code=True
    ),
    "BambaForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-ai-platform/Bamba-9B-v1"),
        extras={"tiny": os.path.join(models_path_prefix, "hmellor/tiny-random-BambaForCausalLM")},
    ),
    "BloomForCausalLM": _HfExamplesInfo(
        "bigscience/bloom-560m", {"1b": os.path.join(models_path_prefix, "bigscience/bloomz-1b1")}
    ),
    "ChatGLMModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/chatglm3-6b"), trust_remote_code=True, max_transformers_version="4.48"
    ),
    "ChatGLMForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "thu-coai/ShieldLM-6B-chatglm3"),
        trust_remote_code=True,
    ),
    "CohereForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "CohereLabs/c4ai-command-r-v01"), trust_remote_code=True
    ),
    "Cohere2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "CohereLabs/c4ai-command-r7b-12-2024"),
        trust_remote_code=True,
    ),
    "CwmForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "facebook/cwm"), min_transformers_version="4.58"),
    # FIXME: databricks/dbrx-instruct has been deleted
    "DbrxForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "databricks/dbrx-instruct"), is_available_online=False
    ),
    "DeciLMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/Llama-3_3-Nemotron-Super-49B-v1"),
        trust_remote_code=True,
    ),
    "DeepseekForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "deepseek-ai/deepseek-moe-16b-base"),
        trust_remote_code=True,
    ),
    "DeepseekV2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "deepseek-ai/DeepSeek-V2-Lite-Chat"),
        trust_remote_code=True,
    ),
    "DeepseekV3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "deepseek-ai/DeepSeek-V3"),
        trust_remote_code=True,
    ),
    "DeepseekV32ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "deepseek-ai/DeepSeek-V3.2-Exp")),
    "Ernie4_5ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "baidu/ERNIE-4.5-0.3B-PT")),
    "Ernie4_5_MoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "baidu/ERNIE-4.5-21B-A3B-PT")),
    "ExaoneForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"), trust_remote_code=True
    ),
    "Exaone4ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "LGAI-EXAONE/EXAONE-4.0-32B")),
    "ExaoneMoEForCausalLM": _HfExamplesInfo(
        "LGAI-EXAONE/K-EXAONE-236B-A23B", min_transformers_version="5.1.0"
    ),
    "Fairseq2LlamaForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "mgleize/fairseq2-dummy-Llama-3.2-1B")),
    "FalconForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "tiiuae/falcon-7b")),
    "FalconH1ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "tiiuae/Falcon-H1-0.5B-Base")),
    "FlexOlmoForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "allenai/Flex-reddit-2x7B-1T")),
    "GemmaForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-1.1-2b-it")),
    "Gemma2ForCausalLM": _HfExamplesInfo(
        "google/gemma-2-9b", extras={"tiny": os.path.join(models_path_prefix, "google/gemma-2-2b-it")}
    ),
    "Gemma3ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-3-1b-it")),
    "Gemma3nForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-3n-E2B-it")),
    "GlmForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "zai-org/glm-4-9b-chat-hf")),
    "Glm4ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "zai-org/GLM-4-9B-0414")),
    "Glm4MoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "zai-org/GLM-4.5")),
    "Glm4MoeLiteForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/GLM-4.7-Flash"),
        min_transformers_version="5.0.0",
    ),
    "GlmMoeDsaForCausalLM": _HfExamplesInfo(
        "zai-org/GLM-5", min_transformers_version="5.0.1", is_available_online=False
    ),
    "GPT2LMHeadModel": _HfExamplesInfo(os.path.join(models_path_prefix, "openai-community/gpt2"), {"alias": os.path.join(models_path_prefix, "gpt2")}),
    "GPTBigCodeForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "bigcode/starcoder"),
        extras={
            "tiny": os.path.join(models_path_prefix, "bigcode/tiny_starcoder_py"),
            "santacoder": os.path.join(models_path_prefix, "bigcode/gpt_bigcode-santacoder"),
        },
    ),
    "GPTJForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Milos/slovak-gpt-j-405M"), {"6b": os.path.join(models_path_prefix, "EleutherAI/gpt-j-6b")}
    ),
    "GPTNeoXForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "EleutherAI/pythia-70m"), {"1b": os.path.join(models_path_prefix, "EleutherAI/pythia-1.4b")}
    ),
    "GptOssForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "lmsys/gpt-oss-20b-bf16")),
    "GraniteForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "ibm/PowerLM-3b")),
    "GraniteMoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "ibm/PowerMoE-3b")),
    "GraniteMoeHybridForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-granite/granite-4.0-tiny-preview")
    ),
    "GraniteMoeSharedForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-research/moe-7b-1b-active-shared-experts")
    ),
    "Grok1ModelForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "hpcai-tech/grok-1"), trust_remote_code=True
    ),
    "Grok1ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "xai-org/grok-2"), trust_remote_code=True),
    "HunYuanDenseV1ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "tencent/Hunyuan-7B-Instruct")),
    "HunYuanMoEV1ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "tencent/Hunyuan-A13B-Instruct"), trust_remote_code=True
    ),
    "InternLMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "internlm/internlm-chat-7b"), trust_remote_code=True
    ),
    "InternLM2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "internlm/internlm2-chat-7b"), trust_remote_code=True
    ),
    "InternLM2VEForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "OpenGVLab/Mono-InternVL-2B"), trust_remote_code=True
    ),
    "InternLM3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "internlm/internlm3-8b-instruct"), trust_remote_code=True
    ),
    "JAISLMHeadModel": _HfExamplesInfo(os.path.join(models_path_prefix, "inceptionai/jais-13b-chat")),
    "Jais2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "inceptionai/Jais-2-8B-Chat"), min_transformers_version="4.58"
    ),
    "IQuestCoderForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "IQuestLab/IQuest-Coder-V1-40B-Instruct"), trust_remote_code=True
    ),
    "IQuestLoopCoderForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct"), trust_remote_code=True
    ),
    "JAISLMHeadModel": _HfExamplesInfo(os.path.join(models_path_prefix, "inceptionai/jais-13b-chat")),
    "Jais2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "inceptionai/Jais-2-8B-Chat"), min_transformers_version="4.58"
    ),
    "JambaForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ai21labs/AI21-Jamba-1.5-Mini"),
        extras={
            "tiny": os.path.join(models_path_prefix, "ai21labs/Jamba-tiny-dev"),
            "random": os.path.join(models_path_prefix, "ai21labs/Jamba-tiny-random"),
        },
    ),
    "KimiLinearForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "moonshotai/Kimi-Linear-48B-A3B-Instruct"), trust_remote_code=True
    ),
    "Lfm2ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "LiquidAI/LFM2-1.2B")),
    "Lfm2MoeForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "LiquidAI/LFM2-8B-A1B"), min_transformers_version="4.58"
    ),
    "LlamaForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "meta-llama/Llama-3.2-1B-Instruct"),
        extras={
            "guard": os.path.join(models_path_prefix, "meta-llama/Llama-Guard-3-1B"),
            "hermes": os.path.join(models_path_prefix, "NousResearch/Hermes-3-Llama-3.1-8B"),
            "fp8": os.path.join(models_path_prefix, "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"),
            "tiny": os.path.join(models_path_prefix, "hmellor/tiny-random-LlamaForCausalLM"),
        },
    ),
    "LLaMAForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "decapoda-research/llama-7b-hf"), is_available_online=False
    ),
    "Llama4ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    ),
    "LongcatFlashForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "meituan-longcat/LongCat-Flash-Chat"), trust_remote_code=True
    ),
    "MambaForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "state-spaces/mamba-130m-hf")),
    "Mamba2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Mamba-Codestral-7B-v0.1"),
        extras={
            "random": os.path.join(models_path_prefix, "yujiepan/mamba2-codestral-v0.1-tiny-random"),
        },
    ),
    "FalconMambaForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "tiiuae/falcon-mamba-7b-instruct")),
    "MiniCPMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openbmb/MiniCPM-2B-sft-bf16"), trust_remote_code=True
    ),
    "MiniCPM3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openbmb/MiniCPM3-4B"), trust_remote_code=True
    ),
    "MiniCPM4ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openbmb/MiniCPM4.1-8B"), trust_remote_code=True
    ),
    "MiniMaxForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "MiniMaxAI/MiniMax-Text-01-hf")),
    "MiniMaxText01ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "MiniMaxAI/MiniMax-Text-01"),
        trust_remote_code=True,
        revision="a59aa9cbc53b9fb8742ca4e9e1531b9802b6fdc3",
    ),
    "MiniMaxM1ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "MiniMaxAI/MiniMax-M1-40k"), trust_remote_code=True
    ),
    "MiniMaxM2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "MiniMaxAI/MiniMax-M2"),
        trust_remote_code=True,
    ),
    "MistralForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "mistralai/Mistral-7B-Instruct-v0.1")),
    "MistralLarge3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4")
    ),
    "MixtralForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        {"tiny": os.path.join(models_path_prefix, "TitanML/tiny-mixtral")},
    ),
    "MptForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "mpt"), is_available_online=False),
    # FIXME: mosaicml/mpt-7b has been deleted
    "MPTForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "mosaicml/mpt-7b"), is_available_online=False),
    "NemotronForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "nvidia/Minitron-8B-Base")),
    "NemotronHForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/Nemotron-H-8B-Base-8K"), trust_remote_code=True
    ),
    "OlmoForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "allenai/OLMo-1B-hf")),
    "Olmo2ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "allenai/OLMo-2-0425-1B")),
    "Olmo3ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "allenai/Olmo-3-7B-Instruct")),
    "OlmoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "allenai/OLMoE-1B-7B-0924-Instruct")),
    "OpenPanguMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1"),
        trust_remote_code=True,
        is_available_online=False,
    ),
    "OPTForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "facebook/opt-125m"), {"1b": os.path.join(models_path_prefix, "facebook/opt-iml-max-1.3b")}
    ),
    "OrionForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "OrionStarAI/Orion-14B-Chat"), trust_remote_code=True
    ),
    "OuroForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "ByteDance/Ouro-1.4B"), trust_remote_code=True),
    "PanguEmbeddedForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "FreedomIntelligence/openPangu-Embedded-7B-V1.1"), trust_remote_code=True
    ),
    "PanguProMoEV2ForCausalLM": _HfExamplesInfo(
        "",
        trust_remote_code=True,
        is_available_online=False,
    ),
    "PanguUltraMoEForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1"),
        trust_remote_code=True,
        is_available_online=False,
    ),
    "PersimmonForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "adept/persimmon-8b-chat")),
    "PhiForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "microsoft/phi-2")),
    "Phi3ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "microsoft/Phi-3-mini-4k-instruct")),
    "PhiMoEForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "microsoft/Phi-3.5-MoE-instruct"), trust_remote_code=True
    ),
    "Plamo2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "pfnet/plamo-2-1b"),
        trust_remote_code=True,
    ),
    "Plamo3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "pfnet/plamo-3-nict-2b-base"),
        trust_remote_code=True,
    ),
    "QWenLMHeadModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen-7B-Chat"),
        max_transformers_version="4.53",
        transformers_version_reason={
            "hf": "HF model uses remote code that is not compatible with latest Transformers"  # noqa: E501
        },
        trust_remote_code=True,
    ),
    "Qwen2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2-0.5B-Instruct"),
        extras={
            "2.5": os.path.join(models_path_prefix, "Qwen/Qwen2.5-0.5B-Instruct"),
            "2.5-1.5B": os.path.join(models_path_prefix, "Qwen/Qwen2.5-1.5B-Instruct"),
        },
    ),
    "Qwen2MoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen1.5-MoE-A2.7B-Chat")),
    "Qwen3ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen3-8B")),
    "Qwen3MoeForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen3-30B-A3B")),
    "Qwen3NextForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        extras={"tiny-random": os.path.join(models_path_prefix, "tiny-random/qwen3-next-moe")},
        min_transformers_version="4.56.3",
    ),
    "RWForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "tiiuae/falcon-40b")),
    "SeedOssForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ByteDance-Seed/Seed-OSS-36B-Instruct"),
        trust_remote_code=True,
    ),
    "Step1ForCausalLM": _HfExamplesInfo(
        "stepfun-ai/Step-Audio-EditX", trust_remote_code=True
    ),
    "Step3p5ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "stepfun-ai/step-3.5-flash"), is_available_online=False
    ),
    "SmolLM3ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "HuggingFaceTB/SmolLM3-3B")),
    "StableLMEpochForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "stabilityai/stablelm-zephyr-3b")),
    "StableLmForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "stabilityai/stablelm-3b-4e1t")),
    "Starcoder2ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "bigcode/starcoder2-3b")),
    "Step3TextForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "stepfun-ai/step3"), trust_remote_code=True),
    "SolarForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "upstage/solar-pro-preview-instruct"), trust_remote_code=True
    ),
    "TeleChatForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "chuhac/TeleChat2-35B"), trust_remote_code=True
    ),
    "TeleChat2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Tele-AI/TeleChat2-3B"), trust_remote_code=True
    ),
    "TeleFLMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "CofeAI/FLM-2-52B-Instruct-2407"), trust_remote_code=True
    ),
    "XverseForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "xverse/XVERSE-7B-Chat"),
        tokenizer=os.path.join(models_path_prefix, "meta-llama/Llama-2-7b"),
        trust_remote_code=True,
    ),
    "Zamba2ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "Zyphra/Zamba2-7B-instruct")),
    "MiMoForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "XiaomiMiMo/MiMo-7B-RL"), trust_remote_code=True),
    "MiMoV2FlashForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "XiaomiMiMo/MiMo-V2-Flash"), trust_remote_code=True
    ),
    "Dots1ForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "rednote-hilab/dots.llm1.inst")),
}

_EMBEDDING_EXAMPLE_MODELS = {
    # [Text-only]
    "BertModel": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/bge-base-en-v1.5")),
    "BgeM3EmbeddingModel": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/bge-m3")),
    "Gemma2Model": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/bge-multilingual-gemma2")),
    "Gemma3TextModel": _HfExamplesInfo(os.path.join(models_path_prefix, "google/embeddinggemma-300m")),
    "GritLM": _HfExamplesInfo(os.path.join(models_path_prefix, "parasail-ai/GritLM-7B-vllm")),
    "GteModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Snowflake/snowflake-arctic-embed-m-v2.0"), trust_remote_code=True
    ),
    "GteNewModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Alibaba-NLP/gte-base-en-v1.5"),
        trust_remote_code=True,
        hf_overrides={"architectures": ["GteNewModel"]},
    ),
    "InternLM2ForRewardModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "internlm/internlm2-1_8b-reward"), trust_remote_code=True
    ),
    "JambaForSequenceClassification": _HfExamplesInfo(os.path.join(models_path_prefix, "ai21labs/Jamba-tiny-reward-dev")),
    "LlamaModel": _HfExamplesInfo("llama", is_available_online=False),
    "LlamaBidirectionalModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/llama-nemotron-embed-1b-v2"), trust_remote_code=True
    ),
    "MistralModel": _HfExamplesInfo(os.path.join(models_path_prefix, "intfloat/e5-mistral-7b-instruct")),
    "ModernBertModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Alibaba-NLP/gte-modernbert-base"), trust_remote_code=True
    ),
    "NomicBertModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nomic-ai/nomic-embed-text-v2-moe"), trust_remote_code=True
    ),
    "Qwen2Model": _HfExamplesInfo(os.path.join(models_path_prefix, "ssmits/Qwen2-7B-Instruct-embed-base")),
    "Qwen2ForRewardModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2.5-Math-RM-72B"),
        max_transformers_version="4.53",
        transformers_version_reason={
            "hf": "HF model uses remote code that is not compatible with latest Transformers"  # noqa: E501
        },
    ),
    "Qwen2ForProcessRewardModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2.5-Math-PRM-7B"),
        max_transformers_version="4.53",
        transformers_version_reason={
            "hf": "HF model uses remote code that is not compatible with latest Transformers"  # noqa: E501
        },
    ),
    "RobertaModel": _HfExamplesInfo(os.path.join(models_path_prefix, "sentence-transformers/stsb-roberta-base-v2")),
    "RobertaForMaskedLM": _HfExamplesInfo(os.path.join(models_path_prefix, "sentence-transformers/all-roberta-large-v1")),
    "XLMRobertaModel": _HfExamplesInfo(os.path.join(models_path_prefix, "intfloat/multilingual-e5-small")),
    "BertSpladeSparseEmbeddingModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "naver/splade-v3"),
        hf_overrides={"architectures": ["BertSpladeSparseEmbeddingModel"]},
    ),
    # [Multimodal]
    "CLIPModel": _HfExamplesInfo(os.path.join(models_path_prefix, "openai/clip-vit-base-patch32")),
    "LlavaNextForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "royokong/e5-v")),
    "Phi3VForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "TIGER-Lab/VLM2Vec-Full"), trust_remote_code=True
    ),
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "MrLight/dse-qwen2-2b-mrl-v1")),
    "SiglipModel": _HfExamplesInfo(os.path.join(models_path_prefix, "google/siglip-base-patch16-224")),
    "PrithviGeoSpatialMAE": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"),
        dtype="float16",
        enforce_eager=True,
        require_embed_inputs=True,
        # This is to avoid the model going OOM in CI
        max_num_seqs=32,
    ),
    "Terratorch": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"),
        dtype="float16",
        enforce_eager=True,
        require_embed_inputs=True,
        # This is to avoid the model going OOM in CI
        max_num_seqs=32,
    ),
}

_SEQUENCE_CLASSIFICATION_EXAMPLE_MODELS = {
    # [Decoder-only]
    "GPT2ForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nie3e/sentiment-polish-gpt2-small")
    ),
    # [Cross-encoder]
    "BertForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "cross-encoder/ms-marco-MiniLM-L-6-v2")
    ),
    "BertForTokenClassification": _HfExamplesInfo(os.path.join(models_path_prefix, "boltuix/NeuroBERT-NER")),
    "GteNewForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Alibaba-NLP/gte-multilingual-reranker-base"),
        trust_remote_code=True,
        hf_overrides={"architectures": ["GteNewForSequenceClassification"]},
    ),
    "LlamaBidirectionalForSequenceClassification": _HfExamplesInfo(
        "nvidia/llama-nemotron-rerank-1b-v2", trust_remote_code=True
    ),
    "ModernBertForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Alibaba-NLP/gte-reranker-modernbert-base")
    ),
    "ModernBertForTokenClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "disham993/electrical-ner-ModernBERT-base")
    ),
    "RobertaForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "cross-encoder/quora-roberta-base")
    ),
    "XLMRobertaForSequenceClassification": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/bge-reranker-v2-m3")),
}

_AUTOMATIC_CONVERTED_MODELS = {
    # Use as_seq_cls_model for automatic conversion
    "GemmaForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "BAAI/bge-reranker-v2-gemma"),
        hf_overrides={
            "architectures": ["GemmaForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        },
    ),
    "LlamaForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Skywork/Skywork-Reward-V2-Llama-3.2-1B")
    ),
    "Qwen2ForSequenceClassification": _HfExamplesInfo(os.path.join(models_path_prefix, "jason9693/Qwen2.5-1.5B-apeach")),
    "Qwen3ForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
    ),
    "Qwen3ForTokenClassification": _HfExamplesInfo("bd2lcco/Qwen3-0.6B-finetuned"),
    "Qwen3VLForSequenceClassification": _HfExamplesInfo(
        "Qwen/Qwen3-VL-Reranker-2B",
        is_available_online=False,
        hf_overrides={
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    ),
}

_MULTIMODAL_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "rhymes-ai/Aria")),
    "AudioFlamingo3ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/audio-flamingo-3-hf"), min_transformers_version="5.0.0"
    ),
    "AyaVisionForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "CohereLabs/aya-vision-8b")),
    "BagelForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "ByteDance-Seed/BAGEL-7B-MoT")),
    "BeeForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Open-Bee/Bee-8B-RL"),
        trust_remote_code=True,
    ),
    "Blip2ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Salesforce/blip2-opt-2.7b"),
        extras={"6b": os.path.join(models_path_prefix, "Salesforce/blip2-opt-6.7b")},
    ),
    "ChameleonForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "facebook/chameleon-7b")),
    "Cohere2VisionForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "CohereLabs/command-a-vision-07-2025")
    ),
    "DeepseekVLV2ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "deepseek-ai/deepseek-vl2-tiny"),
        extras={"fork": os.path.join(models_path_prefix, "Isotr0py/deepseek-vl2-tiny")},
        max_transformers_version="4.48",
        transformers_version_reason={"hf": "HF model is not compatible."},
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
    ),
    "DeepseekOCRForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "deepseek-ai/DeepSeek-OCR"),
    ),
    "DotsOCRForCausalLM": _HfExamplesInfo(
       os.path.join(models_path_prefix,  "rednote-hilab/dots.ocr"), trust_remote_code=True
    ),
    "Eagle2_5_VLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/Eagle2.5-8B"), trust_remote_code=True, is_available_online=False
    ),
    "Emu3ForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/Emu3-Chat-hf")),
    "Ernie4_5_VLMoeForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "baidu/ERNIE-4.5-VL-28B-A3B-PT"),
        trust_remote_code=True,
    ),
    "FuyuForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "adept/fuyu-8b")),
    "Gemma3ForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-3-4b-it")),
    "Gemma3nForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-3n-E2B-it")),
    "GlmAsrForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/GLM-ASR-Nano-2512"),
        trust_remote_code=True,
        min_transformers_version="5.0.0",
    ),
    "GraniteVision": _HfExamplesInfo("ibm-granite/granite-vision-3.3-2b"),
    "GraniteSpeechForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "ibm-granite/granite-speech-3.3-2b")
    ),
    "GLM4VForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/glm-4v-9b"),
        trust_remote_code=True,
        hf_overrides={"architectures": ["GLM4VForCausalLM"]},
    ),
    "Glm4vForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "zai-org/GLM-4.1V-9B-Thinking")),
    "Glm4vMoeForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "zai-org/GLM-4.5V")),
    "GlmOcrForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/GLM-OCR"),
        is_available_online=False,
        min_transformers_version="5.1.0",
    ),
    "H2OVLChatModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "h2oai/h2ovl-mississippi-800m"),
        trust_remote_code=True,
        extras={"2b": os.path.join(models_path_prefix, "h2oai/h2ovl-mississippi-2b")},
        max_transformers_version="4.48",
        transformers_version_reason={"hf": "HF model is not compatible."},
    ),
    "HCXVisionForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"),
        trust_remote_code=True,
    ),
    "HunYuanVLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "tencent/HunyuanOCR"),
        hf_overrides={"num_experts": 0},
    ),
    "Idefics3ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "HuggingFaceM4/Idefics3-8B-Llama3"),
        extras={"tiny": os.path.join(models_path_prefix, "HuggingFaceTB/SmolVLM-256M-Instruct")},
    ),
    "IsaacForConditionalGeneration": _HfExamplesInfo(
        "PerceptronAI/Isaac-0.1",
        trust_remote_code=True,
        extras={"0.2-2B-Preview": "PerceptronAI/Isaac-0.2-2B-Preview"},
    ),
    "InternS1ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "internlm/Intern-S1"), trust_remote_code=True
    ),
    "InternVLChatModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "OpenGVLab/InternVL2-1B"),
        extras={
            "2B": os.path.join(models_path_prefix, "OpenGVLab/InternVL2-2B"),
            "3.0": os.path.join(models_path_prefix, "OpenGVLab/InternVL3-1B"),
            "3.5-qwen3": os.path.join(models_path_prefix, "OpenGVLab/InternVL3_5-1B"),
            "3.5-qwen3moe": os.path.join(models_path_prefix, "OpenGVLab/InternVL3_5-30B-A3B"),
            "3.5-gptoss": os.path.join(models_path_prefix, "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview"),
        },
        trust_remote_code=True,
    ),
    "InternVLForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "OpenGVLab/InternVL3-1B-hf")),
    "KananaVForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "kakaocorp/kanana-1.5-v-3b-instruct"),
        trust_remote_code=True,
    ),
    "KeyeForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Kwai-Keye/Keye-VL-8B-Preview"),
        trust_remote_code=True,
    ),
    "KeyeVL1_5ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Kwai-Keye/Keye-VL-1_5-8B"),
        trust_remote_code=True,
    ),
    "KimiVLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "moonshotai/Kimi-VL-A3B-Instruct"),
        extras={"thinking": os.path.join(models_path_prefix, "moonshotai/Kimi-VL-A3B-Thinking")},
        trust_remote_code=True,
        max_transformers_version="4.53.3",
        transformers_version_reason={
            "hf": (
                "HF model uses deprecated transformers API "
                "(PytorchGELUTanh, DynamicCache.seen_tokens, and more). See: "
                "https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/discussions/31"
            )
        },
    ),
    "KimiK25ForConditionalGeneration": _HfExamplesInfo(
        "moonshotai/Kimi-K2.5",
        trust_remote_code=True,
        is_available_online=False,
    ),
    "LightOnOCRForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "lightonai/LightOnOCR-1B-1025")
    ),
    "Lfm2VlForConditionalGeneration": _HfExamplesInfo(
        "LiquidAI/LFM2-VL-450M",
        min_transformers_version="5.0.0",
    ),
    "Llama4ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        max_model_len=10240,
        extras={"llama-guard-4": os.path.join(models_path_prefix, "meta-llama/Llama-Guard-4-12B")},
    ),
    "LlavaForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "llava-hf/llava-1.5-7b-hf"),
        extras={
            "mistral": os.path.join(models_path_prefix, "mistral-community/pixtral-12b"),
            "mistral-fp8": os.path.join(models_path_prefix, "nm-testing/pixtral-12b-FP8-dynamic"),
        },
    ),
    "LlavaNextForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "llava-hf/llava-v1.6-mistral-7b-hf")
    ),
    "LlavaNextVideoForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "llava-hf/LLaVA-NeXT-Video-7B-hf")
    ),
    "LlavaOnevisionForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    ),
    "MantisForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "TIGER-Lab/Mantis-8B-siglip-llama3"),
        max_transformers_version="4.48",
        transformers_version_reason={"hf": "HF model is not compatible."},
        hf_overrides={"architectures": ["MantisForConditionalGeneration"]},
    ),
    "MiDashengLMModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mispeech/midashenglm-7b"), trust_remote_code=True
    ),
    "MiniCPMO": _HfExamplesInfo(os.path.join(models_path_prefix, "openbmb/MiniCPM-o-2_6"), trust_remote_code=True),
    "MiniCPMV": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openbmb/MiniCPM-Llama3-V-2_5"),
        extras={
            "2.6": os.path.join(models_path_prefix, "openbmb/MiniCPM-V-2_6"),
            "4.0": os.path.join(models_path_prefix, "openbmb/MiniCPM-V-4"),
            "4.5": os.path.join(models_path_prefix, "openbmb/MiniCPM-V-4_5"),
        },
        trust_remote_code=True,
    ),
    "MiniMaxVL01ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "MiniMaxAI/MiniMax-VL-01"),
        trust_remote_code=True,
    ),
    "Mistral3ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
        extras={"fp8": os.path.join(models_path_prefix, "nm-testing/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic")},
    ),
    "MolmoForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "allenai/Molmo-7B-D-0924"),
        max_transformers_version="4.48",
        transformers_version_reason={
            "vllm": "Incorrectly-detected `tensorflow` import from processor."
        },
        extras={"olmo": os.path.join(models_path_prefix, "allenai/Molmo-7B-O-0924")},
        trust_remote_code=True,
    ),
    "Molmo2ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "allenai/Molmo2-8B"),
        extras={"olmo": "allenai/Molmo2-O-7B"},
        min_transformers_version="4.51",
        trust_remote_code=True,
        # required by current PrefixLM implementation
        max_num_batched_tokens=31872,
    ),
    "NVLM_D": _HfExamplesInfo(os.path.join(models_path_prefix, "nvidia/NVLM-D-72B"), trust_remote_code=True),
    "Llama_Nemotron_Nano_VL": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"),
        trust_remote_code=True,
    ),
    "NemotronH_Nano_VL_V2": _HfExamplesInfo(
        os.path.join(models_path_prefix, "nano_vl_dummy"), is_available_online=False, trust_remote_code=True
    ),
    "OpenCUAForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "xlangai/OpenCUA-7B"), trust_remote_code=True
    ),
    "Ovis": _HfExamplesInfo(
        os.path.join(models_path_prefix, "AIDC-AI/Ovis2-1B"),
        trust_remote_code=True,
        max_transformers_version="4.53",
        transformers_version_reason={"hf": "HF model is not compatible"},
        extras={
            "1.6-llama": os.path.join(models_path_prefix, "AIDC-AI/Ovis1.6-Llama3.2-3B"),
            "1.6-gemma": os.path.join(models_path_prefix, "AIDC-AI/Ovis1.6-Gemma2-9B"),
        },
    ),
    "Ovis2_5": _HfExamplesInfo(os.path.join(models_path_prefix, "AIDC-AI/Ovis2.5-2B"), trust_remote_code=True),
    "PaddleOCRVLForConditionalGeneration": _HfExamplesInfo(
       os.path.join(models_path_prefix,  "PaddlePaddle/PaddleOCR-VL"),
        trust_remote_code=True,
    ),
    "PaliGemmaForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "google/paligemma-3b-mix-224"),
        extras={"v2": os.path.join(models_path_prefix, "google/paligemma2-3b-ft-docci-448")},
    ),
    "Phi3VForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "microsoft/Phi-3-vision-128k-instruct"),
        trust_remote_code=True,
        max_transformers_version="4.48",
        transformers_version_reason={
            "hf": "HF model use deprecated imports which have been removed."
        },  # noqa: E501
        extras={"phi3.5": os.path.join(models_path_prefix, "microsoft/Phi-3.5-vision-instruct")},
    ),
    "Phi4MMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "microsoft/Phi-4-multimodal-instruct"), trust_remote_code=True
    ),
    "PixtralForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Pixtral-12B-2409"),
        extras={
            "mistral-large-3": os.path.join(models_path_prefix, "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4"),
            "ministral-3": os.path.join(models_path_prefix, "mistralai/Ministral-3-3B-Instruct-2512"),
        },
        tokenizer_mode="mistral",
    ),
    "QwenVLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen-VL"),
        extras={"chat": os.path.join(models_path_prefix, "Qwen/Qwen-VL-Chat")},
        trust_remote_code=True,
        max_transformers_version="4.53.3",
        transformers_version_reason={
            "hf": "HF model uses deprecated imports which have been removed."
        },  # noqa: E501
        hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]},
    ),
    "Qwen2AudioForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2-Audio-7B-Instruct")
    ),
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen2-VL-2B-Instruct")),
    "Qwen2_5_VLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2.5-VL-3B-Instruct"),
        max_model_len=4096,
    ),
    "Qwen2_5OmniModel": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen2.5-Omni-3B")),
    "Qwen2_5OmniForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "Qwen/Qwen2.5-Omni-7B-AWQ")),
    "Qwen3VLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-VL-4B-Instruct"),
        max_model_len=4096,
        min_transformers_version="4.57",
    ),
    "Qwen3VLMoeForConditionalGeneration": _HfExamplesInfo(
       os.path.join(models_path_prefix,  "Qwen/Qwen3-VL-30B-A3B-Instruct"),
        max_model_len=4096,
        min_transformers_version="4.57",
    ),
    "Qwen3_5ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3.5-9B-Instruct"),
        max_model_len=4096,
        min_transformers_version="5.1.0",
    ),
    "Qwen3_5MoeForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3.5-35B-A3B-Instruct"),
        max_model_len=4096,
        min_transformers_version="5.1.0",
    ),
    "Qwen3_5MTP": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3.5-9B-Instruct"),
        speculative_model="Qwen/Qwen3.5-9B-Instruct",
        min_transformers_version="5.1.0",
    ),
    "Qwen3_5MoeMTP": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3.5-35B-A3B-Instruct"),
        speculative_model="Qwen/Qwen3.5-35B-A3B-Instruct",
        min_transformers_version="5.1.0",
    ),
    "Qwen3OmniMoeForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        max_model_len=4096,
        min_transformers_version="4.57",
    ),
    "RForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "YannQi/R-4B"), trust_remote_code=True),
    "SkyworkR1VChatModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Skywork/Skywork-R1V-38B"), trust_remote_code=True
    ),
    "SmolVLMForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ),
    "Step3VLForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "stepfun-ai/step3"), trust_remote_code=True
    ),
    "StepVLForConditionalGeneration": _HfExamplesInfo(
        "stepfun-ai/Step3-VL-10B", trust_remote_code=True
    ),
    "UltravoxModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "fixie-ai/ultravox-v0_5-llama-3_2-1b"),
        trust_remote_code=True,
    ),
    "TarsierForConditionalGeneration": _HfExamplesInfo(os.path.join(models_path_prefix, "omni-research/Tarsier-7b")),
    "Tarsier2ForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "omni-research/Tarsier2-Recap-7b"),
        hf_overrides={
            "architectures": [os.path.join(models_path_prefix, "Tarsier2ForConditionalGeneration")],
            "model_type": "tarsier2",
        },
    ),
    "VoxtralForConditionalGeneration": _HfExamplesInfo(
        "mistralai/Voxtral-Mini-3B-2507",
        # disable this temporarily until we support HF format
        is_available_online=False,
    ),
    "VoxtralStreamingGeneration": _HfExamplesInfo(
        "<place-holder>",
        # disable this temporarily until we support HF format
        is_available_online=False,
    ),
    # [Encoder-decoder]
    "NemotronParseForConditionalGeneration": _HfExamplesInfo(
        "nvidia/NVIDIA-Nemotron-Parse-v1.1", trust_remote_code=True
    ),
    "WhisperForConditionalGeneration": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openai/whisper-large-v3-turbo"),
        extras={"v3": os.path.join(models_path_prefix, "openai/whisper-large-v3")},
    ),
    # [Cross-encoder]
    "JinaVLForRanking": _HfExamplesInfo(os.path.join(models_path_prefix, "jinaai/jina-reranker-m0")),
}


_SPECULATIVE_DECODING_EXAMPLE_MODELS = {
    "MedusaModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "JackFram/llama-68m"), speculative_model=os.path.join(models_path_prefix, "abhigoyal/vllm-medusa-llama-68m-random")
    ),
    # Temporarily disabled.
    # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.
    # "MLPSpeculatorPreTrainedModel": _HfExamplesInfo(
    #     "JackFram/llama-160m",
    #     speculative_model="ibm-ai-platform/llama-160m-accelerator"
    # ),
    "DeepSeekMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "luccafong/deepseek_mtp_main_random"),
        speculative_model=os.path.join(models_path_prefix, "luccafong/deepseek_mtp_draft_random"),
        trust_remote_code=True,
    ),
    "EagleDeepSeekMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "eagle618/deepseek-v3-random"),
        speculative_model=os.path.join(models_path_prefix, "eagle618/eagle-deepseek-v3-random"),
        trust_remote_code=True,
    ),
    "EagleLlamaForCausalLM": _HfExamplesInfo(
       os.path.join(models_path_prefix,  "meta-llama/Meta-Llama-3-8B-Instruct"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "yuhuili/EAGLE-LLaMA3-Instruct-8B"),
        tokenizer=os.path.join(models_path_prefix, "meta-llama/Meta-Llama-3-8B-Instruct"),
    ),
    "Eagle3LlamaForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "meta-llama/Llama-3.1-8B-Instruct"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
        tokenizer=os.path.join(models_path_prefix, "meta-llama/Llama-3.1-8B-Instruct"),
        use_original_num_layers=True,
        max_model_len=10240,
    ),
    "EagleMistralLarge3ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "mistralai/Mistral-Large-3-675B-Instruct-2512"),
        speculative_model=os.path.join(models_path_prefix, "mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle"),
        # TODO: revert once figuring out OOM in CI
        is_available_online=False,
    ),
    "LlamaForCausalLMEagle3": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-8B"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "AngelSlim/Qwen3-8B_eagle3"),
        tokenizer=os.path.join(models_path_prefix, "Qwen/Qwen3-8B"),
        use_original_num_layers=True,
    ),
    "EagleLlama4ForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct"),
        tokenizer=os.path.join(models_path_prefix, "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    ),
    "EagleMiniCPMForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "openbmb/MiniCPM-1B-sft-bf16"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "openbmb/MiniCPM-2B-sft-bf16"),
        speculative_method=os.path.join(models_path_prefix, "eagle"),
        tokenizer=os.path.join(models_path_prefix, "openbmb/MiniCPM-2B-sft-bf16"),
    ),
    "ErnieMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "baidu/ERNIE-4.5-21B-A3B-PT"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "baidu/ERNIE-4.5-21B-A3B-PT"),
    ),
    "ExaoneMoeMTP": _HfExamplesInfo(
        "LGAI-EXAONE/K-EXAONE-236B-A23B",
        speculative_model="LGAI-EXAONE/K-EXAONE-236B-A23B",
        min_transformers_version="5.1.0",
    ),
    "Glm4MoeMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "zai-org/GLM-4.5"),
        speculative_model="zai-org/GLM-4.5",
    ),
    "Glm4MoeLiteMTPModel": _HfExamplesInfo(
        "zai-org/GLM-4.7-Flash",
        speculative_model="zai-org/GLM-4.7-Flash",
        min_transformers_version="5.0.0",
    ),
    "GlmOcrMTPModel": _HfExamplesInfo(
        "zai-org/GLM-OCR",
        speculative_model="zai-org/GLM-OCR",
        is_available_online=False,
        min_transformers_version="5.1.0",
    ),
    "LongCatFlashMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, os.path.join(models_path_prefix, "meituan-longcat/LongCat-Flash-Chat")),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "meituan-longcat/LongCat-Flash-Chat"),
    ),
    "MiMoMTPModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "XiaomiMiMo/MiMo-7B-RL"),
        trust_remote_code=True,
        speculative_model=os.path.join(models_path_prefix, "XiaomiMiMo/MiMo-7B-RL"),
    ),
    "Eagle3Qwen2_5vlForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen2.5-VL-7B-Instruct"),
        speculative_model=os.path.join(models_path_prefix, "Rayzl/qwen2.5-vl-7b-eagle3-sgl"),
    ),
    "Eagle3Qwen3vlForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-VL-8B-Instruct"),
        speculative_model=os.path.join(models_path_prefix, "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"),
    ),
    "Qwen3NextMTP": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-Next-80B-A3B-Instruct"), min_transformers_version="4.56.3"
    ),
    "Step3p5MTP": _HfExamplesInfo(
        "stepfun-ai/Step-3.5-Flash",
        trust_remote_code=True,
        speculative_model="stepfun-ai/Step-3.5-Flash",
        is_available_online=False,
    ),
}

_TRANSFORMERS_BACKEND_MODELS = {
    "TransformersEmbeddingModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "BAAI/bge-base-en-v1.5"), min_transformers_version="5.0.0"
    ),
    "TransformersForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "papluca/xlm-roberta-base-language-detection"),
        min_transformers_version="5.0.0",
    ),
    "TransformersForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "hmellor/Ilama-3.2-1B"), trust_remote_code=True
    ),
    "TransformersMultiModalForCausalLM": _HfExamplesInfo(os.path.join(models_path_prefix, "BAAI/Emu3-Chat-hf")),
    "TransformersMoEForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "allenai/OLMoE-1B-7B-0924"), min_transformers_version="5.0.0"
    ),
    "TransformersMultiModalMoEForCausalLM": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-VL-30B-A3B-Instruct"), min_transformers_version="5.0.0"
    ),
    "TransformersMoEEmbeddingModel": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-30B-A3B"), min_transformers_version="5.0.0"
    ),
    "TransformersMoEForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "Qwen/Qwen3-30B-A3B"), min_transformers_version="5.0.0"
    ),
    "TransformersMultiModalEmbeddingModel": _HfExamplesInfo(os.path.join(models_path_prefix, "google/gemma-3-4b-it")),
    "TransformersMultiModalForSequenceClassification": _HfExamplesInfo(
        os.path.join(models_path_prefix, "google/gemma-3-4b-it")
    ),
}

_EXAMPLE_MODELS = {
    **_TEXT_GENERATION_EXAMPLE_MODELS,
    **_EMBEDDING_EXAMPLE_MODELS,
    **_SEQUENCE_CLASSIFICATION_EXAMPLE_MODELS,
    **_MULTIMODAL_EXAMPLE_MODELS,
    **_SPECULATIVE_DECODING_EXAMPLE_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> Set[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        try:
            return self.hf_models[model_arch]
        except KeyError:
            raise ValueError(
                f"No example model defined for {model_arch}; please update this file."
            ) from None

    def find_hf_info(self, model_id: str) -> _HfExamplesInfo:
        for info in self.hf_models.values():
            if info.default == model_id:
                return info

        # Fallback to extras
        for info in self.hf_models.values():
            if any(extra == model_id for extra in info.extras.values()):
                return info

        raise ValueError(
            f"No example model defined for {model_id}; please update this file."
        )


HF_EXAMPLE_MODELS = HfExampleModels(_EXAMPLE_MODELS)
AUTO_EXAMPLE_MODELS = HfExampleModels(_AUTOMATIC_CONVERTED_MODELS)