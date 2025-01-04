import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM,CodeGenForCausalLM
proxies = {
   'http': 'http://127.0.0.1:7890',
   'https': 'http://127.0.0.1:7890',
}


model_cards = [
    # 'huggyllama/llama-7b',
    # 'facebook/opt-6.7b' ,
    # 'mosaicml/mpt-7b',
    # 'EleutherAI/gpt-j-6b',
    # 'EleutherAI/gpt-neo-2.7B',
    # 'EleutherAI/pythia-6.9b',
    'Salesforce/codegen2-7B',
]


for model in model_cards:
    while True:
        # try:
            print(f'downloading {model}')
            # tokenizer = AutoTokenizer.from_pretrained(model, proxies=proxies)
            start_time = time.time()
            model = CodeGenForCausalLM.from_pretrained(model, proxies=proxies,torch_dtype=torch.bfloat16)
            print(f'loaded {model} in {time.time() - start_time}')

            print()

        # finally:
        #     break


