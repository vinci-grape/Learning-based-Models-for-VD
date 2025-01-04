import math
import re
from pathlib import Path
import json
from typing import Callable
from tqdm import tqdm
from treesitter_utils import traverse_source_code, get_function_body_pos,CodeAbstracter
import numpy as np
import random
random.seed(2023)

project_dir = Path(__file__).parent.parent.parent
raw_dataset_dir = project_dir / "vul4c_dataset"

splits = ['train', 'valid', 'test']

def transform(new_dataset_name:str,transform_func:Callable[[str],str],dump=True):
    save_dir = project_dir / new_dataset_name
    save_dir.mkdir(exist_ok=True)
    for split in splits:
        split_data = json.load((raw_dataset_dir / f"{split}.json").open(mode='r'))
        (save_dir / split).mkdir(exist_ok=True,parents=True)
        for item in split_data:
            lang = item['lang'].lower()
            id = item['id']
            item['func'] = transform_func(item['func'])
            item['func_after'] = transform_func(item['func_after'])

            # for generate joern
            if dump:
                with (save_dir / split / f'{id}.{lang}').open(mode='w') as f:
                    f.write(item['func'])

        if dump:
            json.dump(split_data,(save_dir / f'{split}.json').open(mode='w'))

def remove_comments(func:str) -> str:

    def replace_comments_with_whitespace(csrc):
        regex = r'(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)'
        old_len = len(csrc)
        def replace_block_comment(x:re.Match) -> str:
            if x.group(2) is not None:
                match = x.group()
                replace_str = list(' ' * len(match))
                for idx, c in enumerate(match):
                    if c == '\n':
                        replace_str[idx] = '\n'
                replace_str = ''.join(replace_str)
                return replace_str
            else:
                return x.group(1)

        csrc = re.sub(regex, replace_block_comment,csrc, flags=re.MULTILINE|re.DOTALL)

        assert old_len == len(csrc)

        return csrc

    rm_comments_func = replace_comments_with_whitespace(func)

    return rm_comments_func


random_generated_comments = [
"/* Initializing variables before the main loop. */",
"/* Function to display a welcome message. */",
"/* Check if the file exists before proceeding. */",
"/* Placeholder code for future feature expansion. */",
"/* Loop to iterate through the list of items. */",
"/* This function calculates the area of a rectangle. */",
"/* Temporary fix for a known issue in the code. */",
"/* Displaying an error message to the user. */",
"/* Function to validate user credentials. */",
"/* A block of code to handle database connection. */",
"/* Commenting out unused code for reference. */",
"/* Implementing a custom sorting algorithm. */",
"/* Section for including necessary header files. */",
"/* Code to print the contents of a linked list. */",
"/* Placeholder function for future optimization. */",
"/* This part of the code logs user activity. */",
"/* Performing boundary check on array indices. */",
"/* Function to calculate the average of values. */",
"/* A block to define global configuration settings. */",
"/* Section for handling command-line arguments. */",
"/* Code snippet to generate a random number. */",
"/* Conditional statement to handle special cases. */",
"/* Placeholder comment for error handling code. */",
"/* Function to validate and sanitize input data. */",
"/* This loop iterates through a matrix. */",
"/* Implementing a basic stack data structure. */",
"/* Section for initializing hardware peripherals. */",
"/* Temporary variable for storing intermediate result. */",
"/* Code to handle file read and write operations. */",
"/* Displaying a success message to the user. */",
"/* A block to calculate statistical measures. */",
"/* Placeholder code to simulate a time delay. */",
"/* Section for setting up interrupt handlers. */",
"/* This function converts data from one format to another. */",
"/* Commenting out code for debugging purposes. */",
"/* A loop to process characters in a string. */",
"/* Code to handle memory allocation and deallocation. */",
"/* Placeholder code for multi-threading support. */",
"/* A block to define macros for constant values. */",
"/* Section for handling network communication. */",
"/* Temporary condition for testing edge cases. */",
"/* Function to reverse the order of elements. */",
"/* Comment explaining the purpose of a specific struct. */",
"/* Code to handle error reporting and logging. */",
"/* Displaying the menu options to the user. */",
"/* Placeholder code for handling signal processing. */",
"/* Section for defining custom data types. */",
"/* A loop to iterate over elements in an array. */",
"/* Code to handle binary file read and write operations. */",
"/* Placeholder comment for implementing encryption. */",
]


def insert_comment(func:str):
    start_point , end_point = get_function_body_pos(func)
    if start_point is None or end_point is None:
        return func
    start_insert_row , end_insert_row = start_point[0] + 1, end_point[0]
    insert_ratio = 0.1
    insert_pos = range(start_insert_row,end_insert_row)
    insert_pos = random.sample(insert_pos, k = math.ceil(len(insert_pos) * insert_ratio),)   # insert position should be sorted

    func_rows = func.splitlines(keepends=True)
    acc = 0
    for pos in insert_pos:
        func_rows.insert( pos + acc ,  random.choice(random_generated_comments)  + '\n')

    func = ''.join(func_rows)

    return func

irrelevant_code = [
    "int var1=0;\nint var2=1;\nint var3=var1+var2;\n",
    "while(0)\n{\n}\n",
    "int var1=0;\nfor(; var1 != 0 ; var1++){\n}\n",
    "int varx=5;\nwhile(varx < 0)\n { \n }\n",
    "int var_num = 0;\nwhile(var_num < -100 && var_num > 100)\n{ \n}\n ",
    "int var1=0;\nint var2=0;\nwhile(var1!=var2)\n{ \n}\n",
    "do\n{\n}\nwhile(0);\n",
    "if(0)\n { \n }\n",
    "if(1)\n{\n}\nelse\n{\n}\n",
    "int var1=0;\nif(var1==0){\n}\n",
]

def insert_unexecuted_code(func:str):
    start_point , end_point = get_function_body_pos(func)
    if start_point is None or end_point is None:
        return func
    start_insert_row , end_insert_row = start_point[0] + 1, end_point[0]
    func_rows = func.splitlines(keepends=True)

    # first check empty line
    candidate_insert_pos = []
    for i,func_line in enumerate(func_rows):
        if i < start_insert_row or i >= end_insert_row:
            continue
        if len(func_line.strip()) == 0 : # candidate line
            candidate_insert_pos.append(i)

    if len(candidate_insert_pos) == 0 and (end_insert_row - start_insert_row) > 5 :
        # random select
        candidate_insert_pos.append(random.choice(range(start_insert_row,end_insert_row)))

    if len(candidate_insert_pos) == 0:
        return func

    candidate_insert_pos = random.choice(candidate_insert_pos)
    func_rows.insert(candidate_insert_pos, random.choice(irrelevant_code))
    new_func = ''.join(func_rows)

    # print('======== raw ===========')
    # print(func)
    # print('======== new ===========')
    # print(new_func)

    return new_func


code_abstracter = CodeAbstracter()
def rename_identifier(func:str):
    # print('======== raw ===========')
    # print(func)
    new_func , _ = code_abstracter.abstract_func(func)
    new_func = ''.join(new_func)
    # print('======== new ===========')
    # print(new_func)
    return new_func



transform('vul4c_rm_comments_dataset',remove_comments,dump=True)
transform('vul4c_insert_comments_dataset',insert_comment,dump=True)
transform('vul4c_unexecuted_code_dataset',insert_unexecuted_code,dump=True)
transform('vul4c_rename_identifier_dataset',rename_identifier,dump=True)
