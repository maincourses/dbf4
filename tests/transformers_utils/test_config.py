# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`vllm.LLMEngine._get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config
from ..utils import models_path_prefix
import os
import pytest
from vllm.platforms import current_platform



@pytest.mark.skipif(current_platform.is_rocm(),
                    reason="Consistent with NV.")
def test_get_llama3_eos_token():
    model_name = os.path.join(models_path_prefix, "meta-llama/Meta-Llama-3-8B-Instruct")

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = os.path.join(models_path_prefix, "Salesforce/blip2-opt-2.7b")

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118
