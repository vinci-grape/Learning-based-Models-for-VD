import json

import fire
import openai
import os

import pandas as pd
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import proxies
from icl_util import ExampleSelectStrategy
from tqdm.contrib.concurrent import process_map
from chatgpt_data import gpt_cot_data, gpt_instruction_data, gpt_in_context_learning_data
from pathlib import Path
from chatgpt_data import OpenAIMessage, CotStrategy
from result_classifier import result_classifier
from functools import partial
from chatgpt_utils import chatgpt_response
from utils import get_logger
from tqdm.contrib.logging import  tqdm_logging_redirect

openai.api_key = 'your openai key'
logger = get_logger('result/chatgpt/chatgpt.log', mode='a')

####### MODE ########
mode = "zero_shot"
which_test_set = "top_cwe_25"
example_select_strategy = ExampleSelectStrategy.Random
print(ExampleSelectStrategy['Random'])
cot_strategy = CotStrategy.FewShotByDiversity
temperature = 0.0


#####################

def chatgpt_generate(item: dict, is_test_sample: bool = False):
    messages = None
    need_cwe_id = True if which_test_set == 'top_cwe_25' else False
    if mode == 'zero_shot':
        messages = gpt_instruction_data(item,need_cwe_id)
    elif mode == 'icl':
        messages = gpt_in_context_learning_data(item, example_select_strategy,need_cwe_id)
    elif mode == 'cot':
        messages = gpt_cot_data(item, cot_strategy, need_cwe_id , None)
        cot = chatgpt_response(messages.get())
        messages = gpt_cot_data(item, cot_strategy, need_cwe_id ,cot)

    assert type(messages) is OpenAIMessage

    response: str = chatgpt_response(messages.get())
    response = response.lower()

    if is_test_sample:
        logger.info('\n\n')
        logger.info('****************************  Sample  ****************************')
        logger.info('*** Input ***')
        logger.info(messages.pretty())
        logger.info('*** Output ***')
        logger.info(response)
        logger.info('******************************************************************\n\n')

    return {'id': item['id'], 'input': messages.raw(), 'output': response, 'vul': item['vul'],
            'cve' : item['cve'],
            'cwe_list': item['cwe_list']}


def main(
        p_which_test_set='raw',  # ['raw', 'top_cwe_25']
        p_temperature=0.0,
        p_mode='icl',
        p_icl_strategy='SameRepo',        # [Fix , Random , Diversity , CosineSimilar , SameRepo]
        p_cot_strategy='FewShotByDiversity', # [FewShotByHand, FewShotByDiversity , ZeroShot ]
):
    global mode, which_test_set, example_select_strategy, cot_strategy, temperature
    mode = p_mode
    which_test_set = p_which_test_set
    temperature = p_temperature
    example_select_strategy = ExampleSelectStrategy[p_icl_strategy]
    cot_strategy = CotStrategy[p_cot_strategy]

    assert which_test_set in ['raw', 'top_cwe_25']
    assert mode in ['zero_shot', 'icl', 'cot']
    strategy = ""
    if mode == "icl":
        strategy = f"{example_select_strategy.value}"
    elif mode == "cot":
        strategy = f"{cot_strategy.value}"

    result_path = f"./result/chatgpt/{mode}/{which_test_set}_{temperature}.json"
    if mode == 'icl':
        result_path = f"./result/chatgpt/{mode}/{example_select_strategy.value}/{which_test_set}_{temperature}.json"
    elif mode == 'cot':
        result_path = f"./result/chatgpt/{mode}/{cot_strategy.value}/{which_test_set}_{temperature}.json"
    Path(result_path).parent.mkdir(exist_ok=True, parents=True)

    test_set: list[dict]
    if which_test_set == 'raw':
        test_set = pd.read_json('../vul4c_rename_identifier_dataset/test.json')
        logger.info(test_set['vul'].value_counts())
        test_set = test_set.sample(n=1039, random_state=2023)
        logger.info('**** after ****')
        logger.info(test_set['vul'].value_counts())
        test_set = test_set.to_dict('records')

    elif which_test_set == "top_cwe_25":
        test_set = pd.read_json('../vul4c_dataset/test_top_cwe_25.json').to_dict('records')

    logger.info(f'test set size:{len(test_set)}')

    setting_info = f"test_set:[{which_test_set}] temperature:[{temperature}] mode:[{mode}] strategy:[{strategy}]"
    logger.info(f'[Running Setting] {setting_info}')

    global chatgpt_response
    chatgpt_response = partial(chatgpt_response, temperature=temperature)

    chatgpt_generate(test_set[0], True)
    
    result = process_map(chatgpt_generate, test_set, max_workers=10 )
    json.dump(result, open(result_path, mode='w'), indent=4)

    logger.info('****************************  Result  ****************************')
    logger.info(setting_info)
    result_metrics = result_classifier(result_path)
    logger.info(result_metrics)
    tqdm.tqdm()

if __name__ == '__main__':
    fire.Fire(main)

