import math
from datetime import datetime

from transformers import AutoModel, LlamaTokenizer, LlamaConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, \
    AutoConfig, GenerationConfig
import torch
import json
from transformers import AutoTokenizer
import pandas as pd
from prompt_utils import prompt_model_related_pre_processing,decode_str_post_processing
from config import *
from pathlib import Path
from string import Template
from tqdm import tqdm
from example_recorders import ExampleRecorder
import fire
from utils import get_logger,set_gpu,load_model_and_tokenizer

test_set = json.load(open('../vul4c_dataset/test.json', mode='r'))

small_test_set = list(filter(lambda x: len(x['func'].splitlines()) < 30, test_set))

small_test_set_df = pd.DataFrame(small_test_set)
small_test_set_df[small_test_set_df['vul'] == 1].to_dict('records')

balanced_small_test_set = pd.concat(
    [small_test_set_df[small_test_set_df['vul'] == 0].sample(5, random_state=2020),
     small_test_set_df[small_test_set_df['vul'] == 1].sample(5, random_state=2021)]
).sample(frac=1, random_state=2022)
balanced_small_test_set = balanced_small_test_set.to_dict('records')

prompt_templates = [
    """
I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, answer "yes" if the function is vulnerable, answer "no" if there is no vulnerability, no other information needs to be output.

$func
    """
    ,
    """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, you should analyze its code for potential security vulnerabilities, such as input validation issues, buffer overflow vulnerabilities, SQL injection vulnerabilities, and other security risks.

$func

Is there any vulnerability in the above code? Answer "yes" or "no"
    """
    ,
    """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, you should analyze its code for potential security vulnerabilities, such as input validation issues, buffer overflow vulnerabilities, SQL injection vulnerabilities, and other security risks. Output "yes" if the function is vulnerable, output "no" if there is no vulnerability.

$func

The answer (Yes or No) is
    """
]





def main(gpus: list, split_idx: int = 0, total_split_cnt: int = 1):
    # python zero_shot.py -gpus="0,1,2" -split_idx=0 -total_split_cnt=3
    save_dir = Path(f"result/zero_shot/{model_name.replace('/', '-')}")
    logger = get_logger(str(save_dir / f'output_{split_idx}.log'))
    logger.debug(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    set_gpu(gpus,logger)

    global test_set
    every_split_cnt = math.ceil(len(test_set) / total_split_cnt)
    test_set_len = len(test_set)
    test_set = test_set[split_idx * every_split_cnt: (split_idx + 1) * every_split_cnt]
    logger.debug(f'running split [{split_idx}/{total_split_cnt - 1}] , test set size: {len(test_set)}/{test_set_len}')

    logger.debug(f'loading model {model_name}')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 proxies=proxies, use_auth_token=hf_token, cache_dir=cache_dir,
                                                 device_map='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_name, proxies=proxies, use_auth_token=hf_token,
                                               cache_dir=cache_dir, use_fast=True
                                               , padding_side='left')

    # add pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        do_sample=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # repetition_penalty=1.1
    )

    logger.debug('begin generate!!!!')
    for prompt_id, prompt_template in enumerate(prompt_templates):
        logger.debug(f'running prompt {prompt_id}')
        llm_result = []
        example_recorders = ExampleRecorder(save_dir, f'prompt_{prompt_id}_split_{split_idx}.txt', prompt_template)

        for batch_id in tqdm(range(0, len(test_set), batch_size)):
            batch = test_set[batch_id:batch_id + batch_size]
            funcs = [i['func'] for i in batch]
            labels = [i['vul'] for i in batch]
            prompts = [Template(prompt_template).substitute(func=func) for func in funcs]
            prompts = list(map(lambda x: prompt_model_related_pre_processing(model_name, x), prompts))
            x = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
            y = model.generate(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                generation_config=generation_config,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                # return_full_text = False
            )
            decode_results = tokenizer.batch_decode(y.sequences, skip_special_tokens=True)

            for i, decode_result in enumerate(decode_results):
                # post-processing stop words for some models
                decode_result = decode_str_post_processing(model_name, prompts[i], decode_result)
                llm_result.append({
                    'input': funcs[i],
                    'prompt': prompts[i],
                    'output': decode_result[len(prompts[i]):],
                    'label':labels[i]
                })

                example_recorders.record(prompts[i], decode_result)

        json.dump(llm_result, open(save_dir / f'prompt_{prompt_id}_result_split_{split_idx}.json', mode='w'))


if __name__ == '__main__':
    fire.Fire(main)
