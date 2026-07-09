# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import os
import torch

from vllm import LLM
from ...utils import models_path_prefix


def test_empty_prompt():
    llm = LLM(model=os.path.join(models_path_prefix, "openai-community/gpt2"), enforce_eager=True)
    with pytest.raises(ValueError, match="decoder prompt cannot be empty"):
        llm.generate([""])


def test_out_of_vocab_token():
    llm = LLM(model=os.path.join(models_path_prefix, "openai-community/gpt2"), enforce_eager=True)
    with pytest.raises(ValueError, match="out of vocabulary"):
        llm.generate({"prompt_token_ids": [999999]})


def test_require_mm_embeds():
    llm = LLM(
        model=os.path.join(models_path_prefix, "llava-hf/llava-1.5-7b-hf"),
        enforce_eager=True,
        enable_mm_embeds=False,
    )
    with pytest.raises(ValueError, match="--enable-mm-embeds"):
        llm.generate(
            {
                "prompt": "<image>",
                "multi_modal_data": {"image": torch.empty(1, 1, 1)},
            }
        )
