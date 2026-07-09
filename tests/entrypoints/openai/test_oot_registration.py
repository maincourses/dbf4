# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from ...utils import VLLM_PATH, RemoteOpenAIServer, models_path_prefix, envs

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


def run_and_test_dummy_opt_api_server(model, tp=1):
    # the model is registered through the plugin
    if envs.VLLM_USE_TRITON_FLASH_ATTN:
        server_args = [
            "--gpu-memory-utilization",
            "0.10",
            "--dtype",
            "float32",
            "--chat-template",
            str(chatml_jinja_path),
            "--load-format",
            "dummy",
            "-tp",
            f"{tp}",
        ]
    else:
        server_args = [
            "--gpu-memory-utilization",
            "0.10",
            "--dtype",
            "float16",
            "--chat-template",
            str(chatml_jinja_path),
            "--load-format",
            "dummy",
            "-tp",
            f"{tp}",
    ]
    with RemoteOpenAIServer(model, server_args) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            temperature=0,
        )
        generated_text = completion.choices[0].message.content
        assert generated_text is not None
        # make sure only the first token is generated
        rest = generated_text.replace("<s>", "")
        assert rest == ""


def test_oot_registration_for_api_server(dummy_opt_path: str):
    dummy_opt_path = os.path.join(models_path_prefix, "facebook/opt-125m")
    run_and_test_dummy_opt_api_server(dummy_opt_path)
