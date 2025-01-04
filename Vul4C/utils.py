import dataclasses
import json
import inspect
import time
import re
from dataclasses import dataclass
from multiprocessing import Lock
from typing import Union

import requests.exceptions
from requests import request,get

from tinydb import TinyDB, Query
from pathlib import Path
import chardet
DEBUG = False


def debug(*values: object,end = ...):
    if DEBUG:
        print(*values,end=end)


proxies = {
    'https': 'http://127.0.0.1:7890',
    'http': 'http://127.0.0.1:7890',
}

class FunctionSignatureError(Exception):

    def __init__(self, msg:str) -> None:
        self.msg = msg

@dataclass
class ExtractedFunction:
    func_name:str
    parameter_list_signature:str
    parameter_list:list
    return_type:str
    func:list[str]

def get_safe(url: str):
    while True:
        try:
            res = get(url, timeout=10, proxies=proxies)
            if len(res.text) ==0 or res.text is None:
                continue
            if res.status_code == 503:  # Service Temporarily Unavailable
                continue
            return res
        except (requests.exceptions.ReadTimeout , requests.exceptions.ProxyError , requests.exceptions.SSLError,requests.exceptions.ChunkedEncodingError):
            time.sleep(1)
            print('retrying...')

@dataclass
class CommitInfo:
    id: str
    repo_name:str
    commit_hash: str
    parent_commit_hash: Union[str, None]
    diff_file_paths: list[str]
    tree_url: str
    commit_msg: str
    git_url:str


@dataclass
class DiffFuncPair:
    func_name:str
    parameter_list_signature:str
    parameter_list:list
    return_type:str
    func_before:str
    func_before_abstract:str
    func_before_abstract_symbol_map:dict
    func_after:str
    func_after_abstract:str
    func_after_abstract_symbol_map:dict
    raw_diff :str
    parse_diff:dict

@dataclass
class NonVulFunction:
    func_name:str
    parameter_list_signature:str
    parameter_list:list
    return_type:str
    func:str


@dataclass
class DiffFile:
    file_path:str
    lang:str
    diff_funcs:list[DiffFuncPair]   # vulnerable function
    non_vul_funcs:list[NonVulFunction]  # non vulnerable function


@dataclass
class CommitInfoWithDiffInfo:
    repo_name:str
    commit_hash: str
    parent_commit_hash: Union[str, None]
    diff_file_paths: list[str]
    commit_msg: str
    git_url:str
    diff_files:list[DiffFile]



c = CommitInfo('20', 'gg', '123', None, ['path/gg'], '12211', '3333',"http")
gg = { 'hello' : 10}
print(json.dump({**gg,'commit':[dataclasses.asdict(c),dataclasses.asdict(c)]},open('test.json',mode='w')))

def repo_name_try_merge(repo_name:str):
    repo_name = repo_name.replace('.git','')
    if len((split_repo := repo_name.split('/'))) == 2:
        owner , repo = split_repo
        if owner == repo:
            return owner
    return repo_name

def repo_name_try_recover(repo_name:str):
    repo_name_split = repo_name.split('/')
    if len(repo_name_split) == 1:
        repo_name_split = repo_name_split[0]
        return f'{repo_name_split}/{repo_name_split}'
    return repo_name

def convert_short_url_to_origin_url(url:str) -> str:
    # convert short url back to origin url
    while True:
        try:
            res = requests.head(url,proxies=proxies)
            break
        except (requests.exceptions.ReadTimeout , requests.exceptions.ProxyError , requests.exceptions.SSLError,requests.exceptions.ChunkedEncodingError):
            time.sleep(1)
            print('retrying...')
            continue
    if res.status_code == 200 or ('Location' not in res.headers.keys()):
        return url
    new_url = res.headers['Location']
    if (res.status_code//100) == 3 : # recursively unshorten url
        return convert_short_url_to_origin_url(new_url)
    return new_url


def from_dict_to_dataclass(cls, data):
    return cls(
        **{
            key: (data[key] if val.default == val.empty else data.get(key, val.default))
            for key, val in inspect.signature(cls).parameters.items()
        }
    )

def read_json_file_to_dataclass_list(path,cls) -> list:
    data = read_json_file(path)
    res = []
    for d in data:
        res.append(from_dict_to_dataclass(cls,d))
    return res

db_cache = {}
lock = Lock()

def read_commit_info_from_cache(id: str, db: str) -> Union[None, CommitInfo]:
    if db in db_cache:
        db = db_cache[db]
    else:
        tinydb = TinyDB(f'cache/db/{db}')
        db_cache[db] = tinydb
        db = tinydb
    query = Query()
    if len(cache_res := db.search(query.id == id)) != 0:
        print('using cache')
        res = cache_res[0]
        return from_dict_to_dataclass(CommitInfo,res)
    return None

def write_commit_info_to_cache(c:CommitInfo, db: str) :
    with lock:
        if db in db_cache:
            db = db_cache[db]
        else:
            tinydb = TinyDB(f'cache/db/{db}',cache_size=0)
            db_cache[db] = tinydb
            db = tinydb
        db.insert(c.__dict__)

def read_json_file(file_path: str):
    with open(file_path, mode='r') as f:
        return json.load(f)

def cache_commit_file_dir(repo_name:str,commit_hash:str,current_hash:str,create_dir = True) -> Path:
    # avoid commit hash collision
    dir_path  = Path(f'cache/download/{repo_name}/{commit_hash}/{current_hash}')
    if create_dir:
        dir_path.mkdir(exist_ok=True,parents=True)
    return dir_path

def trunc_commit_file_name(file_name:str) -> str:
    return file_name.split('/')[-1]

# filter cpp extension
AcceptedFileExtension = ['h','cc','cpp','c','cxx','hpp','hxx','hh']

def filter_accepted_files(files:list[str]) -> list[str]:
    res = []
    # for large commit (file changes > 50 , we ignore)
    if len(files) > 50:
        return res
    for f in files:
        trunc_f = trunc_commit_file_name(f)
        if '.' in trunc_f:
            if trunc_f.split('.')[-1] in AcceptedFileExtension:
                res.append(f)
    return res


def try_decode_binary_data(data:bytes) -> str:
    try:
        return data.decode('utf-8')
    except:
        try:
            return data.decode('ascii')
        except:
            try:
                detect_encoding = chardet.detect(data)['encoding']
                return data.decode(detect_encoding)
            except:
                assert False, print(f'decode error {detect_encoding} {data} ')

def save_str2file(data:str,save_path:Path):
    with open(str(save_path),mode='w') as f:
        f.write(data)

def try_decode_binary_data2file(data:bytes,save_path:Path):
    with open(str(save_path),mode='w') as f:
        file_content = try_decode_binary_data(data)
        f.write(file_content)

def check_file_exist(path:Path) -> bool:
    return path.exists() and (path.stat().st_size != 0)

def write_dataclasses_to_json(data:list,file_path:Path):
    file_path.parent.mkdir(exist_ok=True,parents=True)
    data = list(map(lambda x:dataclasses.asdict(x),data))
    with file_path.open(mode='w') as f:
        json.dump(data,f,indent=4)

def remove_c_code_comment(code):
    """ remove c/cpp source code comment """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, code)
