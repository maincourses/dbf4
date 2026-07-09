# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from ..utils import models_path_prefix

from ..entrypoints.openai.test_oot_registration import run_and_test_dummy_opt_api_server


def test_distributed_oot(dummy_opt_path: str):
    dummy_opt_path = os.path.join(models_path_prefix, "facebook/opt-125m")
    run_and_test_dummy_opt_api_server(dummy_opt_path, tp=2)