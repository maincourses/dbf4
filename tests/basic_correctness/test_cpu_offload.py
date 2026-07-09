# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

from ..utils import compare_two_settings, models_path_prefix


def test_cpu_offload():
    compare_two_settings(
        os.path.join(models_path_prefix, "hmellor/tiny-random-LlamaForCausalLM"), [], ["--cpu-offload-gb", "1"]
    )
