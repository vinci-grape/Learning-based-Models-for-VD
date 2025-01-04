import json
import os.path

import pandas
from utils import read_json_file
from transformers import AutoTokenizer, LlamaTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

# This file create LLM and graph model input data

proxies = {
    'http': 'http://172.31.112.1:7890',
    'https': 'http://172.31.112.1:7890',
}

llm_models = [
    'huggyllama/llama-7b',
    'facebook/opt-6.7b',
    'mosaicml/mpt-7b',
    'EleutherAI/gpt-j-6b',
    'EleutherAI/gpt-neo-2.7B',
    'EleutherAI/pythia-6.9b',
    'Salesforce/codegen2-7B',
]

tokenizers: list[AutoTokenizer] = []
flatten_data: list[dict] = []


def load_tokenizers():
    for model in llm_models:
        tokenizer = AutoTokenizer.from_pretrained(model, proxies=proxies)
        tokenizers.append(tokenizer)
        print(tokenizer.encode('hello'))


MAX_LENGTH = 1900


def check_tokenizen_len(func: str) -> bool:
    tokenizer: LlamaTokenizer
    for tokenizer in tokenizers:
        if len(tokenizer.encode(func)) > MAX_LENGTH:
            return False
    return True


def save_func(cve_id: str, cwe_list: [str], repo_name: str, commit_hash: str, git_url: str, func: str, func_after: str,
              func_graph_idx: int, lang: str, vul: int):
    if not check_tokenizen_len(func):
        return
    flatten_data.append(
        {
            'cve': cve_id, 'cwe_list': cwe_list, 'repo_name': repo_name, 'commit_hash': commit_hash,
            'git_url': git_url, 'func': func, 'func_after': func_after, 'graph_idx': func_graph_idx,
            'lang': lang, 'vul': vul
        }
    )


load_tokenizers()

vul4c_dataset_path = '../vul4c_dataset/vul4c_dataset.json'


def begin_flatten_data():
    cve_commit_with_graph_infos = read_json_file('result/cve_commit_with_graph_infos.json')
    for cve in tqdm(cve_commit_with_graph_infos):  # for test
        cve_id = cve['cve']
        cwe_list = cve['cwe_list']
        for commit in cve['commits']:
            repo_name = commit['repo_name']
            commit_hash = commit['commit_hash']
            git_url = commit['git_url']
            for file in commit['diff_files']:
                lang = file['lang']
                for diff_func in file['diff_funcs']:
                    save_func(cve_id, cwe_list, repo_name, commit_hash, git_url, diff_func['func_before'],
                              diff_func['func_after'],
                              diff_func['func_before_graph_idx'], lang, 1)

                    save_func(cve_id, cwe_list, repo_name, commit_hash, git_url, diff_func['func_after'],
                              diff_func['func_after'],
                              diff_func['func_after_graph_idx'], lang, 0)

                for func in file['non_vul_funcs']:
                    save_func(cve_id, cwe_list, repo_name, commit_hash, git_url, func['func'], func['func'],
                              func['func_graph_idx'],
                              lang, 0)
    vul_cnt = 0
    non_vul_cnt = 0
    for item in flatten_data:
        if item['vul'] == 1:
            vul_cnt += 1
        else:
            non_vul_cnt += 1

    json.dump(flatten_data, open(vul4c_dataset_path, mode='w'))
    print(f'vulnerable:{vul_cnt}  non-vulnerable:{non_vul_cnt}')


if os.path.exists(vul4c_dataset_path):
    print('using cache json file')
else:
    begin_flatten_data()

vul4c_dataset = json.load(open(vul4c_dataset_path, mode='r'))


def get_labels(data: list):
    labels = []
    for item in data:
        labels.append(item['vul'])
    return labels


train, test = train_test_split(vul4c_dataset, train_size=0.8, random_state=2013, shuffle=True,
                               stratify=get_labels(vul4c_dataset))
valid, test = train_test_split(test, train_size=0.5, random_state=2013, shuffle=True, stratify=get_labels(test))


def count_distribution(name: str, funcs: list):
    vul_cnt = 0
    for func in funcs:
        if func['vul'] == 1: vul_cnt += 1
    non_vul_cnt = len(funcs) - vul_cnt
    print(f'{name} split distribution: vul:{vul_cnt} non-vul:{non_vul_cnt} {vul_cnt / len(funcs) * 100: .2f}%')


splits = {
    'train': train,
    'valid': valid,
    'test': test
}


def copy_graph(split_name: str, funcs: list):
    save_path = Path(f'../vul4c_dataset/{split_name}')
    save_path.mkdir(exist_ok=True, parents=True)

    for id, f in enumerate(funcs):
        graph_idx = f['graph_idx']
        for suffix in ['nodes', 'edges']:
            src = f'result/graph/{graph_idx}.{suffix}.json'
            dst = save_path / f'{id}.{suffix}.json'
            if suffix == 'nodes':
                shutil.copy(src, dst)
            elif suffix == 'edges':
                with open(src, mode='r') as f:
                    edges = json.load(f)
                    new_edges = []
                    for e in edges:
                        new_edges.append([e['inNode'], e['outNode'], e['label'], e['variable']])
                    json.dump(new_edges, dst.open(mode='w'))


for split in splits:
    data = splits[split]
    for id, d in enumerate(data):
        d['id'] = id
    print(f'{split}:{len(data)}')
    count_distribution(split, data)
    save_path = Path(f'../vul4c_dataset/{split}.json')
    if not save_path.exists():
        json.dump(data, save_path.open(mode='w'))
        # copy graph
        # copy_graph(split,data)
    else:
        print(f'{save_path} exists, require manually delete!!')
    print()
