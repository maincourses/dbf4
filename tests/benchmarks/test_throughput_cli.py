# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import os
import pytest

from ..utils import  models_path_prefix

MODEL_NAME = os.path.join(models_path_prefix, "meta-llama/Llama-3.2-1B-Instruct")


@pytest.mark.benchmark
def test_bench_throughput():
    command = [
        "vllm",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"