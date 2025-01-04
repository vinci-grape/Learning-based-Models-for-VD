import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


CWE_7PK = {
    'Security Features': [254, 256, 258, 259, 260, 261, 272, 284, 285, 330, 359, 798],
    'Time and State': [361, 364, 367, 377, 382, 383, 384, 412],
    'Errors': [388, 391, 395, 396, 397],
    'Input Validation and Representation': [1005, 20, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                                     113, 114, 117, 119,
                                                     120, 134, 15, 170, 190, 466, 470, 73, 785, 77, 79, 89, 99],
    'API Abuse': [227, 242, 243, 244, 245, 246, 248, 250, 251, 252, 558],
    'Code Quality': [398, 401, 404, 415, 416, 457, 474, 475, 476, 477],
    'Encapsulation': [485, 486, 488, 489, 491, 492, 493, 495, 496, 497, 501],
    'Environment': [2, 11, 12, 13, 14, 5, 6, 7, 8, 9]
}

_global_7pk_cnt = {}
for split in  ['train', 'valid' , 'test']:
    _7pk_cnt = {}
    dataset_path = f'../../vul4c_dataset/{split}.json'
    data = json.load(open(dataset_path,mode='r'))
    for item in data:
        cwe:str
        if item['vul'] == 0 :
            continue
        for cwe in item['cwe_list']:
            if cwe == 'CWE-Other':
                continue
            cwe_number = int(cwe.split('-')[1])

            for _7pk_key,_7pk_value in CWE_7PK.items():
                if cwe_number in _7pk_value:
                    _7pk_cnt.setdefault(_7pk_key ,0)
                    _7pk_cnt[_7pk_key] += 1
                    _global_7pk_cnt.setdefault(_7pk_key ,0)
                    _global_7pk_cnt[_7pk_key] += 1


                    if _7pk_key == 'API Abuse':
                        print(cwe_number)

    print(_7pk_cnt)


print(_global_7pk_cnt)