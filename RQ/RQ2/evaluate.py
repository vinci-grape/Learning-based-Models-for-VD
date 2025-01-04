import json
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

dataset_dir = Path(__file__).parent.parent.parent / "vul4c_dataset"
test_dataset = json.load((dataset_dir / "test.json").open(mode='r'))
llm_test_dataset = pd.read_json(dataset_dir / "test.json").sample(n=1039, random_state=2023).to_dict("records")
datasetid_2_listid = {}
llm_datasetid_2_listid = {}

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

TOP_CWE_25 = ["CWE-787", "CWE-79", "CWE-89", "CWE-416", "CWE-78", "CWE-20", "CWE-125", "CWE-22", "CWE-352", "CWE-434",
              "CWE-862", "CWE-476", "CWE-287", "CWE-190", "CWE-502", "CWE-77", "CWE-119", "CWE-798", "CWE-918",
              "CWE-306",
              "CWE-362", "CWE-269", "CWE-94", "CWE-863", "CWE-276"]

for idx, item in enumerate(test_dataset):
    datasetid_2_listid[item['id']] = idx

for idx, item in enumerate(test_dataset):
    datasetid_2_listid[item['id']] = idx


def id_2_listid(id, is_llm=False):
    if is_llm:
        return llm_datasetid_2_listid[id]
    else:
        return datasetid_2_listid[id]


"""
    {
        id : 
        pred: 
    }
"""


def evaluate_all_cwe(preds, top25_cwe=False):
    cwe_dict = {}
    correct_vul_cnt = 0

    for item in preds:
        list_id = id_2_listid(item['id'], is_llm=False)
        ground_truth = test_dataset[list_id]
        if ground_truth['vul'] != 1:  # only evaluate vulnerable function
            continue
        if bool(ground_truth['vul']) != bool(item['pred']):
            continue
        correct_vul_cnt += 1
        cwe_list = ground_truth['cwe_list']

        for cwe in cwe_list:
            if cwe == 'CWE-Other':
                continue
            if top25_cwe and (cwe not in TOP_CWE_25):
                continue
            cwe_dict.setdefault(cwe, 0)
            cwe_dict[cwe] += 1
    print(f'correct vul cnt :{correct_vul_cnt}')
    return sort_dict_by_value(cwe_dict)


def sort_dict_by_value(dict, top_k=None, key_func=None):
    sorted_value_key_pairs = sorted(dict.items(), key=(lambda x: x[1]) if key_func is None else key_func, reverse=True)
    if top_k is not None:
        sorted_value_key_pairs = sorted_value_key_pairs[:top_k]
    return {v: k for v, k in sorted_value_key_pairs}


def dataset_cwe_cnt(is_llm=False):
    dataset = llm_test_dataset if is_llm else test_dataset
    cwe_dict = {}

    for item in dataset:
        if item['vul'] != 1:
            continue

        for cwe in item['cwe_list']:
            if cwe == 'CWE-Other':
                continue
            cwe_dict.setdefault(cwe, 0)
            cwe_dict[cwe] += 1

    print('*' * 100)
    print(f'{"LLM" if is_llm else "Raw"} Dataset CWE type:{len(cwe_dict.keys())}')
    cwe_dict = sort_dict_by_value(cwe_dict)
    print(cwe_dict)
    print('*' * 100)

    return cwe_dict


raw_cwe_cnt = dataset_cwe_cnt()
llm_cwe_cnt = dataset_cwe_cnt(is_llm=True)


def cwe_accuracy(predict_cwe: dict, cwe_cnt: dict, top_k=None):
    result = {}
    for cwe_key, cnt in predict_cwe.items():
        result[cwe_key] = (cnt, cwe_cnt[cwe_key], round(float(cnt / cwe_cnt[cwe_key]), 4))
    return sort_dict_by_value(result, top_k=top_k, key_func=lambda x: x[1][2])

def evaluate_cwe_in_7PK(preds):

    _7pk_categories_cnt = { }
    _7pk_categories_correct = { }

    for item in preds:
        list_id = id_2_listid(item['id'], is_llm=False)
        ground_truth = test_dataset[list_id]
        if ground_truth['vul'] != 1:  # only evaluate vulnerable function
            continue

        pred_correct = ground_truth['vul'] == item['pred'] 
        cwe_list = ground_truth['cwe_list']

        cwe:str
        for cwe in cwe_list:
            if cwe == 'CWE-Other':
                continue
            cwe_number = int(cwe.split('-')[1])
            which_category = None
            for key_7pk, value_7pk in CWE_7PK.items(): # find this cwe belong to which category
                if cwe_number in value_7pk:
                    which_category = key_7pk
            if which_category is None:
                continue

            _7pk_categories_cnt.setdefault(which_category,0)
            _7pk_categories_cnt[which_category] += 1

            if pred_correct:
                _7pk_categories_correct.setdefault(which_category,0)
                _7pk_categories_correct[which_category] += 1

    _7pk_result = {}

    for key , total_cnt in _7pk_categories_cnt.items():
        pred_correct_cnt = _7pk_categories_correct.get(key ,0)
        _7pk_result[f"{key}[{pred_correct_cnt}/{total_cnt}]"] = pred_correct_cnt / total_cnt

    _7pk_result = sort_dict_by_value(_7pk_result )

    return _7pk_result

def report_result(model_name: str, test_result_path: str, is_llm=False):
    print(f'***** {model_name} *****')
    data = json.load(open(test_result_path, mode='r'))
    if "chatgpt" in model_name:
        newdata = []
        for item in data:
            newdata.append({"id": item['id'], "pred": item['output'].lower() == 'yes'})
        data = newdata

    _7pk_result = evaluate_cwe_in_7PK(data)
    print(f'Model Seven Pernicious Kingdoms: {_7pk_result}')


    correct_predict_cwe = evaluate_all_cwe(data, top25_cwe=True)
    cwe_cnt = llm_cwe_cnt if is_llm else raw_cwe_cnt
    cwe_acc = cwe_accuracy(correct_predict_cwe, cwe_cnt, top_k=10)  
    cwe_acc_str = ""
    for cwe_key, item in cwe_acc.items():
        cwe_acc_str += f"{cwe_key}[{item[0]}/{item[1]}] "
    print(f'Model Top 10 CWE: {cwe_acc_str}')

    # report Top 25 CWE classification result

    top_25_result = []
    for cwe in TOP_CWE_25:
        predict_cnt = correct_predict_cwe[cwe] if cwe in correct_predict_cwe else 0
        total_cnt = cwe_cnt[cwe] if cwe in cwe_cnt else 0
        top_25_result.append((cwe, predict_cnt, total_cnt))
    cwe_top25_str = ""
    for item in top_25_result:
        cwe_top25_str += f"{item[0]}[{item[1]}/{item[2]}] "
    print(f'In Top 25 CWE: {cwe_top25_str}')


    return cwe_acc, top_25_result , _7pk_result


@dataclass
class ModelAndResult:
    model_name: str
    result_path: str


models = [
    ModelAndResult("devign", "../../Graph/storage/results/devign/vul4c_dataset/test.json"),
    ModelAndResult("reveal", "../../Graph/storage/results/reveal/vul4c_dataset/test.json"),
    ModelAndResult("ivdetect", "../../Graph/storage/results/ivdetect/vul4c_dataset/test.json"),
    ModelAndResult("LineVul", "../../sequence/LineVul/storage/test.json"),
    ModelAndResult("SVulD", "../../sequence/SVulD/storage/test.json"),
    ModelAndResult("chatgpt-icl-same-repo", "../../LLM/result/chatgpt/icl/same_repo/raw_0.0_vul4c.json"),
]


def save_top25_results(results):
    model_names = [model.model_name for model in models]
    rows = [[cwe] for cwe in TOP_CWE_25]
    print(rows)
    for result in results:
        for idx, cwe_result in enumerate(result):
            rows[idx].append(f"{cwe_result[1]}/{cwe_result[2]}")
        #
        # print(result)
        print()
    print(rows)
    df = pd.DataFrame(rows, columns=[""] + model_names)
    df.to_csv("top25cwe_results.csv")


def save_model_top10_results(results):
    print('-----')
    print(results)
    rows = []
    for model, result in zip(models, results):
        data = [model.model_name]
        for cwe_key, item in result.items():
            print(item)

            data.append(f"{cwe_key}[{item[0]}/{item[1]}]")
        rows.append(data)
    df = pd.DataFrame(rows, )
    df = df.fillna("")
    df.to_csv("model_top10cwe_results.csv")

def save_7pk_results(results):
    print(results)
    rows = []

    for model, result in zip(models, results):
        data = [model.model_name]
        for item in result.keys():
            data.append(item)
        rows.append(data)
    df = pd.DataFrame(rows, )
    df = df.fillna("")
    df.to_csv("model_7pk_results.csv")


if __name__ == '__main__':

    top_25_results = []
    cwe_acc_results = []
    _7pk_results = []

    for model in models:
        cwe_acc, top_25_result, _7pk_result = report_result(model.model_name, model.result_path,
                                               is_llm="chatgpt" in model.model_name)
        top_25_results.append(top_25_result)
        cwe_acc_results.append(cwe_acc)
        _7pk_results.append(_7pk_result)

    # to csv file
    save_model_top10_results(cwe_acc_results)
    save_top25_results(top_25_results)
    save_7pk_results(_7pk_results)

    print(raw_cwe_cnt)
