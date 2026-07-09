
'''
python offline_streaming_inference_chat_demo.py --model /models/llama2/Llama-2-7b-chat-hf  --dtype float16 --enforce-eager -tp 1 
'''
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
import asyncio
from transformers import AutoTokenizer
import logging
import argparse
import sys


if __name__ == '__main__':
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.WARNING)

    class FlexibleArgumentParser(argparse.ArgumentParser):
        """ArgumentParser that allows both underscore and dash in names."""

        def parse_args(self, args=None, namespace=None):
            if args is None:
                args = sys.argv[1:]

            # Convert underscores to dashes and vice versa in argument names
            processed_args = []
            for arg in args:
                if arg.startswith('--'):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = '--' + key[len('--'):].replace('_', '-')
                        processed_args.append(f'{key}={value}')
                    else:
                        processed_args.append('--' +
                                            arg[len('--'):].replace('_', '-'))
                else:
                    processed_args.append(arg)

            return super().parse_args(processed_args, namespace)
    
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # chat = [
    #   {"role": "user", "content": "Hello, how are you?"},
    #   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    #   {"role": "user", "content": "I'd like to show off how chat templating works!"},
    # ]

    tokenizer =  AutoTokenizer.from_pretrained(args.model)
    # try:
    #      f = open(args.template,'r')
    #      tokenizer.chat_template = f.read()
    # except Exception as e:
    #      print('except:',e)
    # finally:
    #      f.close()



    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)


    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] !=""  else args.model.split("/")[-2]
    print(f"欢迎使用{model_name}模型,输入内容即可进行对话,stop 终止程序")


    def build_prompt(history):
        prompt = ""
        for query, response in history:
            prompt += f"\n\n用户:{query}"
            prompt += f"\n\n{model_name}:{response}"
        return prompt


    history = []
    while True:
        query = input("\n用户:")
        if query.strip() == "stop":
            break 
        history.append({"role": "user", "content": query})
        new_query = tokenizer.apply_chat_template(history, tokenize=False)
        example_input = {
        "prompt": new_query,
        "stream": False, 
        "temperature": 0.0,
        "request_id": 0,
        }

        results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"], max_tokens=100),
        example_input["request_id"]
        )

        start = 0
        end = 0
        response = ""
        async def process_results():
            async for  output in results_generator: 
                global end 
                global start 
                global response
                print(output.outputs[0].text[start:], end="", flush=True)
                length = len(output.outputs[0].text)
                start = length
                response = output.outputs[0].text
        
        asyncio.run(process_results())
        history.append({"role": "assistant", "content": response})
    print()

