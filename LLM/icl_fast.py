from typing import Callable
import math
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer, PreTrainedTokenizer
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import pandas as pd
import json
import torch
from enum import Enum
from config import *
from tqdm import tqdm
from transformers import GenerationConfig
from pathlib import Path
from example_recorders import ExampleRecorder
from functools import partial
from prompt_utils import prompt_model_related_pre_processing, decode_str_post_processing
from typing import Union
from utils import get_logger, set_gpu, load_model_and_tokenizer
import fire
from datetime import datetime
from vllm import LLM,LLMEngine,RequestOutput

# TSNE(n_components=2,random_state=2023).fit_transform(load_digits()['data'])
test_set = json.load(open('../vul4c_dataset/test.json', mode='r'))

small_test_set = list(filter(lambda x: len(x['func'].splitlines()) < 30, test_set))
small_test_set_df = pd.DataFrame(small_test_set)

balanced_small_test_set = pd.concat(
    [small_test_set_df[small_test_set_df['vul'] == 0].sample(5, random_state=2020),
     small_test_set_df[small_test_set_df['vul'] == 1].sample(5, random_state=2021)]
).sample(frac=1, random_state=2022).to_dict('records')

# random select 3 vulnerable  3 non-vulnerable function
train_set = pd.read_json('../vul4c_dataset/train.json')
filter_train_set = train_set[
    (train_set['func'].str.split('\n').str.len() < 30) & (train_set['func'].str.split('\n').str.len() > 10)]
print(f'train set {len(filter_train_set)}')
random_select_example = pd.concat(
    [filter_train_set[filter_train_set['vul'] == 1].sample(n=3, random_state=2023),
     filter_train_set[filter_train_set['vul'] == 0].sample(n=3, random_state=2023),
     ]
)
random_select_example = random_select_example.to_dict('records')
fix_select_example_len = None


# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
#                                              proxies=proxies, use_auth_token=hf_token, cache_dir=cache_dir,
#                                              device_map='auto')


def icl_example_prompt(example: dict, is_test: bool, function_prefix: Union[str, None], answer_prefix: str,
                       answer_map: dict,
                       test_prefix_need: bool = True) -> str:
    """
        ${function_prefix} function
        ${func}
        ${answer_prefix} ${answer}
    """
    print(example['vul'] , answer_map)
    answer = answer_map[example['vul']]
    func = f"{example['func']}" if function_prefix is None else f"{function_prefix}\n{example['func']}"
    if is_test:
        if test_prefix_need:
            return f"{func}\n{answer_prefix}\n"
        else:
            return f"{func}\n"
    else:
        return f"{func}\n{answer_prefix} {answer}\n"


icl_example_prompt_makers = {
    '#yes no': partial(icl_example_prompt, function_prefix='# function', answer_prefix='#',
                       answer_map={0: 'no', 1: 'yes'}, test_prefix_need=True),
    '//yes no': partial(icl_example_prompt, function_prefix='// function', answer_prefix='//',
                        answer_map={0: 'no', 1: 'yes'}, test_prefix_need=True),
    'Vulnerability:vulnerable not-vulnerable': partial(icl_example_prompt, function_prefix='# function',
                                                       answer_prefix='Vulnerability:',
                                                       answer_map={0: 'not vulnerable', 1: 'vulnerable'},
                                                       test_prefix_need=True),
    'Vulnerability:yes no': partial(icl_example_prompt, function_prefix='Function:', answer_prefix='Vulnerability:',
                                    answer_map={0: 'no', 1: 'yes'},
                                    test_prefix_need=True),
}

"""    
        [Prompt Format]
    instruction_templates[0]
    examples
    instruction_templates[1]
"""
prompt_templates = [
    # (
    #     'I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you several examples, each containing a function and an answer corresponding to whether there is a vulnerability. At the end I will give you a function that answers "yes" if the function has a vulnerability and "no" if there is no vulnerability, without outputting any other information.',
    #     'Is the last function vulnerable? Answer "yes" or "no"',
    #     icl_example_prompt_makers['#yes no']
    # ),
    (
        'I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you several examples, each containing a function and an answer corresponding to whether there is a vulnerability. At the end I will give you a function that answers "yes" if the function has a vulnerability and "no" if there is no vulnerability, without outputting any other information.',
        'Is the last function vulnerable? Answer "yes" or "no"',
        icl_example_prompt_makers['Vulnerability:yes no']
    )
    # ,
    # (
    #     'I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you several examples, each containing a function and an answer corresponding to whether there is a vulnerability. You need to check if the last function contains a vulnerability and answer "yes" if last function has a vulnerability, answer "no" if there is no vulnerability.',
    #     '',
    #     icl_example_prompt_makers['#yes no']
    # )
    ,
    # (
    #     'As a vulnerability detector, your objective is to detect if a function is vulnerable. Code may contain potential vulnerabilities, such as buffer overflow, XSS, SQL injection, CSRF, command injection. I\'ll give you a few functions and their corresponding labels for whether or not there are vulnerabilities. You need to learn from these functions and answer whether the last one contains a vulnerability, answer "vulnerable" if there is a vulnerability, answer "not vulnerable" if there is no vulnerability.',
    #     'Answer "vulnerable" or "not vulnerable"',
    #     icl_example_prompt_makers['Vulnerability:vulnerable not-vulnerable']
    # )
]


def make_icl_prompt(prompt_template: tuple, prompt_template_len: tuple,
                    test_example: dict, test_example_lens: list,
                    examples: list[dict],
                    tokenizer: PreTrainedTokenizer, generation_config: GenerationConfig,
                    model_max_length: int = 2048, ):
    def encode_length(s: str):
        return len(tokenizer.encode(s))

    example_prompt_maker: Callable[[dict, bool], str] = prompt_template[2]
    left_length = model_max_length - prompt_template_len[0] - prompt_template_len[1] \
                  - encode_length(test_example['func']) - generation_config.max_new_tokens

    # encode as many example prompts as possible
    example_str = ""
    for idx,e in enumerate(examples):
        example_prompt = example_prompt_maker(e, False)
        example_prompt_len = test_example_lens[idx]
        if left_length - example_prompt_len < 0:
            break
        left_length -= example_prompt_len
        example_str += example_prompt + '\n'

    # add test
    example_str += example_prompt_maker(test_example, True)

    return f"{prompt_template[0]}\n{example_str}{prompt_template[1]}"


class ExampleSelectStrategy(Enum):
    Random = "random"
    Diversity = "diversity"
    CosineSimilar = "cosine"


example_select_strategy = ExampleSelectStrategy.Diversity

# e.g. [1,103,2032,4444,5555,6890] 6 categories(6 center item)
train_closest_to_center_idxs = pd.read_pickle('../vul4c_dataset/train_closest_to_center_idxs.pkl').iloc[:,
                               0].values.tolist()

train_closest_to_center_examples = train_set.iloc[train_closest_to_center_idxs].to_dict('records')
train_closest_to_center_examples_len = None

test_similar_to_train = pd.read_pickle('../vul4c_dataset/test_similar_to_train.pkl').to_dict('records')


def select_examples(test_id: int, strategy: ExampleSelectStrategy, tokenizer: PreTrainedTokenizer):
    if strategy == ExampleSelectStrategy.Random:
        global fix_select_example_len
        if random_select_example_len is None:
            random_select_example_len = []
            for e in random_select_example:
                random_select_example_len.append(len(tokenizer.encode(e['func'])))
        return random_select_example, random_select_example_len
    elif strategy == ExampleSelectStrategy.Diversity:
        global train_closest_to_center_examples_len
        if train_closest_to_center_examples_len is None:
            train_closest_to_center_examples_len = []
            for e in train_closest_to_center_examples:
                train_closest_to_center_examples_len.append(len(tokenizer.encode(e['func'])))
        return train_closest_to_center_examples, train_closest_to_center_examples_len
    elif strategy == ExampleSelectStrategy.CosineSimilar:
        test_similar_items = test_similar_to_train[test_id]
        test_similar_items_idxs = test_similar_items['idxs'][:6]  # select top 6
        print('close item', test_similar_items_idxs)
        test_similar_items_scores = test_similar_items['similar_scores'][:6]  # select top 6
        examples = train_set.iloc[test_similar_items_idxs].to_dict('records')
        example_lens = []
        for e in examples:
            example_lens.append(len(tokenizer.encode(e['func'])))
        return examples, example_lens


def main(gpus: list, split_idx: int = 0, total_split_cnt: int = 1):
    # python icl.py -gpus="0,1,2" -split_idx=0 -total_split_cnt=3
    save_dir = Path(f"result/icl/{example_select_strategy.value}/{model_name.replace('/', '-')}")
    save_dir.mkdir(parents=True,exist_ok=True)
    logger = get_logger(str(save_dir / f'output_{split_idx}.log'))
    logger.debug(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    set_gpu(gpus, logger)

    global test_set
    every_split_cnt = math.ceil(len(test_set) / total_split_cnt)
    test_set_len = len(test_set)
    test_set = test_set[split_idx * every_split_cnt: (split_idx + 1) * every_split_cnt]
    logger.debug(f'running split [{split_idx}/{total_split_cnt - 1}] , test set size: {len(test_set)}/{test_set_len}')

    logger.debug(f'loading model {model_name}')
    model, tokenizer = load_model_and_tokenizer(model_name)

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
        example_recorder = ExampleRecorder(save_dir, f'prompt_{prompt_id}_split_{split_idx}.txt',
                                           f"{prompt_template[0]}\n${{examples}}\n{prompt_template[1]}")
        prompt_template_len = (len(tokenizer.encode(prompt_template[0])), len(tokenizer.encode(prompt_template[1])))

        for batch_id in tqdm(range(0, len(test_set), batch_size)):
            batch = test_set[batch_id:batch_id + batch_size]
            funcs = [i['func'] for i in batch]
            labels = [i['vul'] for i in batch]

            prompts = []
            for test_set_item in batch:
                examples, examples_lens = select_examples(test_set_item['id'], example_select_strategy, tokenizer)
                prompt = make_icl_prompt(prompt_template, prompt_template_len, test_set_item, examples, examples_lens,
                                         tokenizer, generation_config)
                prompt = prompt_model_related_pre_processing(model_name, prompt)
                prompts.append(prompt)

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
                    'label': labels[i]
                })

                example_recorder.record(prompts[i], decode_result)

        json.dump(llm_result, open(save_dir / f'prompt_{prompt_id}_result_split_{split_idx}.json', mode='w'))


if __name__ == '__main__':
    fire.Fire(main)
