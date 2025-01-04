from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
import json
from icl_util import select_examples, ExampleSelectStrategy
import tiktoken
from enum import Enum
from chatgpt_utils import chatgpt_response
import random

class OpenAIMessage:
    def __init__(self, ):
        self.messages = []

    def system(self, s: str):
        self.messages.append(
            {"role": "system", "content": s}
        )

    def user(self, s: str):
        self.messages.append(
            {"role": "user", "content": s}
        )

    def assistant(self, s: str):
        self.messages.append(
            {"role": "assistant", "content": s}
        )

    def get(self):
        return self.messages

    def raw(self) -> str:
        s = ""
        for i, v in enumerate(self.messages):
            s += v['content']
            s += '\n'
        return s[:-1]

    def pretty(self):
        s = ""
        for  v in self.messages:
            s += f"[{v['role']}]:\n"
            for line in v['content'].split('\n'):
                s+=f'\t\t\t{line}\n'
            s += '\n'
        return s[:-1]


def openai_tokenize(message, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(message)


def openai_tokenize_len(message, model="gpt-3.5-turbo-0613"):
    return len(openai_tokenize(message, model))


# instruction


def gpt_instruction_data(item: dict, need_cwe_id: bool):
    message = OpenAIMessage()
    if need_cwe_id:
        message.system(
            """As a vulnerability detector, your objective is to analyze the code of a given function for potential security vulnerabilities. You should specifically look for issues related to input validation, buffer overflow, SQL injection, and other security risks. Your output should be in JSON format, stating whether the function is vulnerable or not, along with the CWE ID if applicable. for example,If the function is vulnerable and the vulnerability type is Out-of-bounds Write, the output should be:
            {
                "vulnerable" : "yes",
                "cwe" : "CWE-787"
            }
            Please note that you should only output JSON format and no other information.
            """
        )
        message.user(item['func'])
    else:
        message.system(
            """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, you should analyze its code for potential security vulnerabilities, such as input validation issues, buffer overflow vulnerabilities, SQL injection vulnerabilities, and other security risks. Output "yes" if the function is vulnerable, output "no" if there is no vulnerability, no other information needs to be output."""
        )
        message.user(item['func'])
        message.user("""Is there any vulnerability in the above code? Answer "yes" or "no" """)

    return message


def icl_example_prompt(example: dict, is_test: bool, function_prefix: Union[str, None], answer_prefix: str,
                       answer_map: dict,
                       test_prefix_need: bool = True) -> str:
    """
        ${function_prefix} function
        ${func}
        ${answer_prefix} ${answer}
    """
    answer = answer_map[example['vul']]
    func = f"{example['func']}" if function_prefix is None else f"{function_prefix}\n{example['func']}"
    if is_test:
        if test_prefix_need:
            return f"{func}\n{answer_prefix}\n"
        else:
            return f"{func}\n"
    else:
        return f"{func}\n{answer_prefix} {answer}\n"


def gpt_in_context_learning_data(item: dict, strategy: ExampleSelectStrategy, need_cwe_id: bool):
    def encode_length(s: str):
        return len(openai_tokenize(s))

    examples, examples_len = select_examples(item['id'], strategy, openai_tokenize)
    model_max_length: int = 3900
    if need_cwe_id:
        system_message = """As a vulnerability detector, your objective is to analyze the code of a given function for potential security vulnerabilities. You should specifically look for issues related to input validation, buffer overflow, SQL injection, and other security risks. Your output should be in JSON format, stating whether the function is vulnerable or not, along with the CWE ID if applicable. for example,If the function is vulnerable and the vulnerability type is Out-of-bounds Write, the output should be:
{
    "vulnerable" : "yes",
    "cwe" : "CWE-787"
}
Please note that you should only output JSON format and no other information.
                    """
    else:
        system_message = """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you several examples, each containing a function and an answer corresponding to whether there is a vulnerability. At the end I will give you a function,you should analyze its code for potential security vulnerabilities, such as input validation issues, buffer overflow vulnerabilities, SOL injection vulnerabilities, andother security risks, Answers "yes" if the last function has a vulnerability and "no" if there is no vulnerability."""
    message = OpenAIMessage()
    message.system(system_message)
    left_length = model_max_length - encode_length(system_message) - encode_length(item['func'])

    for idx, e in enumerate(examples):
        input = ""
        output = ""
        if need_cwe_id:
            input = f'\n# function\n{e["func"]}\n# output'
            if e['vul'] == 1:
                output = f"""{{
    "vulnerable" : "yes"
    "cwe" : "{e['cwe_list'][0]}"                
}}"""
            else:
                output = f"""{{
    "vulnerable" : "no"
}}"""
        else:
            input  = f'# function\n{e["func"]}\n# vulnerable:'
            output = 'yes' if e['vul'] == 1 else 'no'
        assert len(input) != 0 and len(output) != 0
        example_prompt_len = openai_tokenize_len(f"{input}{output}")
        if left_length - example_prompt_len < 0:
            break
        message.user(input)
        message.assistant(output)
        left_length -= example_prompt_len

    # add test
    if need_cwe_id:
        message.user(f'\n# function\n{item["func"]}\n# output')
    else:
        message.user(f'\n# function\n{item["func"]}\n# vulnerable:')
    return message


cot_few_shot_info = [
    (11299, """Let's think step by step
1. The function decode_data takes two parameters: a pointer to a structure called struct sixpack named sp, and an unsigned char named inbyte.
2. then it declares a pointer variable buf of type unsigned char.
3. It checks if the rx_count member variable of the sp structure is not equal to 3. If it is not equal to 3, it means there is still space in the raw_buf array to store data.
4. Then it decoded 3 bytes and store the decode result in cooked_buf.
5. Here's a vulnerability that causes out-of-bounds memory writes, the function should check cooked_buf array size"""),
    (88703, """Let's think step by step
1. The function addPeer is a member function of the Logger class.
2. It takes three parameters: ip (of type QString) representing the IP address of the peer, blocked (of type bool) indicating if the peer is blocked, and reason (of type QString) providing the reason for blocking.
3. The ip parameter is passed in by the webUI without escaping many values, which could potentially lead to XSS."""),
    (17002, """Let's think step by step
1. The function dissector_get_string_handle takes two parameters: sub_dissectors: A dissector_table_t variable representing a table of sub-dissectors. string: A const gchar* representing a string.
2. Inside the function callsfind_string_dtbl_entry with the parameters sub_dissectors and string to search for a matching entry in the sub_dissectors table based on the provided string.
3. The result of the find_string_dtbl_entry function is stored in the dtbl_entry variable.
4. if dtbl_entry return the value of dtbl_entry->current else return null
5. There is a vulnerability where table searches find_string_dtbl_entry for empty strings are not handled properly, which allows remote attackers to cause a denial of service."""),
    (228194, """Let's think step by step
1. The function free_wininfo is defined, which takes a pointer to a structure wininfo_T called wip as its parameter.
2. The function first checks if the wi_optset member of the wininfo_T structure pointed to by wip is true using the if statement.
3. If wi_optset is true, the function calls the clear_winopt function, passing the wi_opt member of the wininfo_T structure pointed to by wip as an argument. This function is responsible for clearing the window options.
4. Next, there is an #ifdef directive that checks if the FEAT_FOLDING feature is defined. If it is defined, the code inside the #ifdef block is compiled.
5. Inside the #ifdef FEAT_FOLDING block, the function calls the deleteFoldRecurse function, passing the wi_folds member of the wininfo_T structure pointed to by wip as an argument. This function is responsible for deleting any existing folding information associated with the window.
6. After the #ifdef block, the function calls vim_free to deallocate the memory pointed to by wip.
7. Clearly, this function is not vulnerable."""),
    (92053, """Let's think step by step
1. The function napi_watchdog is defined, which takes a pointer to a struct hrtimer as an argument and returns an enum value of type hrtimer_restart.
2. The line napi = container_of(timer, struct napi_struct, timer); uses the container_of macro to obtain a pointer to the parent structure struct napi_struct using the timer pointer. It assumes that the timer is embedded within a larger structure struct napi_struct.
3. The next block of code checks certain conditions before scheduling the NAPI (New API) processing.
4. Finally, the function returns HRTIMER_NORESTART, indicating that the high-resolution timer should not be restarted.
5. In general this function is not vulnerable"""),
    (123609, """Let's think step by step
1. The function atl2_watchdog takes an unsigned long data as a parameter.
2. It checks if the __ATL2_DOWN bit is not set in the adapter flags using the test_bit function.
3. If the bit is not set, it proceeds with the following steps:
4. It acquires a spin lock using spin_lock_irqsave to protect the adapter stats.
5. It reads the value from the register REG_STS_RXD_OV using ATL2_READ_REG macro and assigns it to drop_rxd.
6. It releases the spin lock using spin_unlock_irqrestore.
7. It increments the rx_over_errors counter in the statistics of the adapter's associated network device (netdev) by adding the values of drop_rxd and drop_rxs.
8. Resets the watchdog timer for the adapter by modifying the expiration time using mod_timer
9. In the function, no obvious vulnerabilities are found."""),
]

cot_few_shot_hand_examples = []
cot_few_shot_hand_examples_with_cwe = []
train_set = pd.read_json('../vul4c_dataset/train.json')


def add_cot_few_shot_str(item):
    global cot_few_shot_hand_examples,cot_few_shot_hand_examples_with_cwe
    idx, info = item
    example = train_set[train_set['id'] == idx].iloc[0]
    vul = example['vul']
    func = example['func']


    intput = f"# function\n{func}\nLet's think step by step first"
    cot = info
    output = f'# vulnerable:{"yes" if vul == 1 else "no"}'
    cot_few_shot_hand_examples.append((intput, cot, output))

    if vul == 1:
        output = f"""# output
{{
    "vulnerable" : "yes"
    "cwe" : "{example['cwe_list'][0]}"                
}}"""
    else:
        output = f"""# output
{{
    "vulnerable" : "no"
}}"""
    cot_few_shot_hand_examples_with_cwe.append((intput, cot, output))

random.Random(123).shuffle(cot_few_shot_info)   # shuffle this array
for i in cot_few_shot_info:
    add_cot_few_shot_str(i)

class CotStrategy(Enum):
    ZeroShot = "zero_shot"
    FewShotByHand = "few_shot_by_hand" 
    FewShotByDiversity = "few_shot_by_diversity"  


def gpt_cot_data(item: dict, cot_strategy: CotStrategy, need_cwe_id: bool, cot_think_prompt: Union[str, None]):
    message = OpenAIMessage()
    left_length = 2500 - openai_tokenize_len(f"# function\n{item['func']}\nLet's think step by step first")

    if need_cwe_id:
        message.system("""As a vulnerability detector, your objective is to analyze the code of a given function for potential security vulnerabilities. You should specifically look for issues related to input validation, buffer overflow, SQL injection, and other security risks.""")
    else:
        if cot_strategy == CotStrategy.ZeroShot:
            message.system(
                """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, you should analyze its code for potential security vulnerabilities. Output "yes" if the function is vulnerable, output "no" if there is no vulnerability."""
            )
        elif cot_strategy in [CotStrategy.FewShotByHand, CotStrategy.FewShotByDiversity]:
            message.system(
                """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you several examples, each containing a function and an answer corresponding to whether there is a vulnerability. At the end I will give you a function,you should analyze its code for potential security vulnerabilities, such as input validation issues, buffer overflow vulnerabilities, SOL injection vulnerabilities, andother security risks, Answers "yes" if the last function has a vulnerability and "no" if there is no vulnerability."""
            )

    if cot_strategy == CotStrategy.FewShotByHand:
        for e in cot_few_shot_hand_examples_with_cwe if need_cwe_id else cot_few_shot_hand_examples:
            intput,cot,output = e
            item_len = openai_tokenize_len(intput + cot + output)
            if left_length - item_len < 0:
                break
            left_length -= item_len

            message.user(intput)
            message.assistant(f"{cot}\n{output}")

        message.user(f"# function\n{item['func']}\nLet's think step by step first")
    elif cot_strategy == CotStrategy.FewShotByDiversity:
        cache_file = Path(f"./result/chatgpt/cot/{CotStrategy.FewShotByDiversity.value}/diversity_cot_cache.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if not cache_file.exists():
            # cache CoT
            print('create cot-diversity cache file')
            cot_result = []
            train_closest_to_center_idxs = pd.read_pickle('../vul4c_dataset/train_closest_to_center_idxs.pkl').iloc[:,
                                           0].values.tolist()
            train_closest_to_center_examples = train_set.iloc[train_closest_to_center_idxs].to_dict('records')

            for example in train_closest_to_center_examples:
                message = OpenAIMessage()
                message.system(
                    """I want you to act as a vulnerability detector, your objective is to detect if a function is vulnerable. I will give you a function, you should analyze its code for security vulnerabilities."""
                )
                cot_think_prompt = "This function is vulnerable, Let's think step by step first about why it is vulnerable" if example[
                                                                                                         'vul'] == 1 else "This function is not vulnerable, Let's think step by step first."
                message.user(f'{example["func"]}\n{cot_think_prompt}')
                response = chatgpt_response(message.get(), temperature=0.0)
                cot_result.append(
                    {'id': example['id'], 'func': example['func'], 'cot': response, 'vul': example['vul'] , 'cve' : example['cve'] , 'cwe_list' : example['cwe_list']    })

            json.dump(cot_result, cache_file.open(mode='w'), indent=4)

        cot_result = json.load(cache_file.open(mode='r'))
        for e in cot_result: 
            input = f'# function\n{e["func"]}\nLet\'s think step by step first\n'
            cot = e['cot']
            output = f'# vulnerable:{"yes" if e["vul"] == 1 else "no"}'
            if need_cwe_id:
                if e["vul"] == 1:
                    output = f"""# output
{{
    "vulnerable" : "yes"
    "cwe" : "{e['cwe_list'][0]}"                
}}"""
                else:
                    output = f"""# output
{{
    "vulnerable" : "no"
}}"""
            item_str = f'{input}\n{cot}\n{output}'
            item_len = openai_tokenize_len(item_str)
            if left_length - item_len < 0:
                break
            left_length -= item_len
            message.user(input)
            message.assistant(f'{cot}\n{output}')
        message.user(f"# function\n{item['func']}\nLet's think step by step first")

    elif cot_strategy == CotStrategy.ZeroShot:
        message.user(f"{item['func']}\nLet's think step by step first")

    if cot_think_prompt is None:
        return message

    message.assistant(cot_think_prompt)

    if need_cwe_id:
        message.user("""Your output should be in JSON format, stating whether the function is vulnerable or not ("yes" or "no"), along with the CWE ID if applicable. for example,If the function is vulnerable and the vulnerability type is Out-of-bounds Write, the output should be:
{
    "vulnerable" : "yes",
    "cwe" : "CWE-787"
}
Please note that you should only output JSON format result and no other information.""")

    else:
        if cot_strategy == CotStrategy.ZeroShot:
            message.user(
                """answer "yes" if the function is vulnerable, answer "no" if there is no vulnerability, no other information needs to be output.""")
        else:  # few shot
            message.user(
                """answer "yes" if the last function is vulnerable, answer "no" if there is no vulnerability, no other information needs to be output.""")

    return message

