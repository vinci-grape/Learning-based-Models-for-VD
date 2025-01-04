import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from utils import debug, CommitInfo, read_commit_info_from_cache, write_commit_info_to_cache, proxies, get_safe, \
    cache_commit_file_dir, trunc_commit_file_name, save_str2file, try_decode_binary_data2file , check_file_exist
from typing import Union
import base64
# android chrome
# https://gerrit.googlesource.com/gitiles/

def gitiles_get_safe(url:str) -> Union[str,None] :
    while True:
        responses = get_safe(url)
        responses , status_code= responses.text , responses.status_code
        if responses.find('RESOURCE_EXHAUSTED: Resource has been exhausted (e.g. check quota)') != -1:
            debug(f'[Gitiles] rate limit exceeded. retrying...')
            continue
        if responses.find('NOT_FOUND: Requested entity was not found') != -1:
            debug(f'[Gitiles] entity not found {url} {responses}')
            return None
        break
    return responses

def gitiles_get_commit_info(url:str) -> Union[CommitInfo,None]:
    debug(f'[Gitiles] {url}')
    id = url

    base_url = urlparse(url)
    repo_name = base_url.netloc.split('.')[0]
    base_url = f'{base_url.scheme}://{base_url.netloc}'
    responses = gitiles_get_safe(url)
    if responses is None:
        return None
    gitiles_page = BeautifulSoup(responses,'html.parser')
    commit_table = gitiles_page.find('table')
    if commit_table is None:
        debug(f'[Gitiles] missing commit table {url}')
        return None
    commit_table = commit_table.find_all('tr')
    if len(commit_table) != 5:  # filter multiple parent commit
        debug(f'[Gitiles] commit table error {url}')
        return None

    commit_hash = commit_table[0].td.text
    tree_url = f'{commit_table[3].a["href"]}'
    tree_url = f"{'/'.join(tree_url.split('/')[:-2])}/"
    tree_url = f'{base_url}{tree_url}'

    # https://android.googlesource.com/platform/frameworks/base/+/ebc250d16c747f4161167b5ff58b3aea88b37acf/
    # https://android.googlesource.com/platform/frameworks/base/+/{commit_hash}/
    parent_commit_hash = commit_table[4].a.text
    if len(parent_commit_hash) == 0:
        parent_commit_hash = None

    commit_msg = gitiles_page.find('pre',class_='u-pre u-monospace MetadataMessage').text

    diff_tree =  gitiles_page.find('ul',class_='DiffTree')
    if diff_tree is None:
        debug(f'[Gitiles] missing diff tree {url}')
        return None
    diff_tree = diff_tree.find_all('li')

    diff_files = []
    for d in diff_tree:
        if d.find('span',class_='DiffTree-action DiffTree-action--add') is not None:
            # this file is added , skip this file.
            continue
        if d.find('span',class_='DiffTree-action DiffTree-action--delete') is not None:
            # this file is deleted , skip this file
            continue
        if d.find('span',class_='DiffTree-action DiffTree-action--rename') is not None:
            # this file is rename from some file, skip this file
            continue
        diff_files.append(d.a.text)

    c = CommitInfo(id,repo_name,commit_hash,parent_commit_hash,diff_files,tree_url,commit_msg,url)
    return c

def gitiles_download_tree_files(tree_url:str,tree_hash:str,files:list[str],save_dir:Path) -> list[str]:
    dl_files = []
    for f in files:
        trunc_name = trunc_commit_file_name(f)
        # cache
        if check_file_exist(save_dir/trunc_name):
            dl_files.append(f)
            continue
        download_url = f'{tree_url}{tree_hash}/{f}?format=TEXT'
        txt = gitiles_get_safe(download_url)
        if txt is None:
            debug(f'[Gitiles] download error or file missing {tree_url} {tree_hash} {f} {download_url}')
            continue
        file_content_b = base64.b64decode(txt)
        try_decode_binary_data2file(file_content_b,save_dir/trunc_name)
        dl_files.append(f)
    return dl_files

def gitiles_download_commit_files(commit_info:CommitInfo):
    # download file
    debug(f'[Gitiles] {commit_info.repo_name} {commit_info.commit_hash}')
    files = commit_info.diff_file_paths
    c_hash , p_hash = commit_info.commit_hash , commit_info.parent_commit_hash
    # commit tree
    a_dl_files = gitiles_download_tree_files(commit_info.tree_url,c_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,c_hash))
    # parent commit tree
    b_dl_files = gitiles_download_tree_files(commit_info.tree_url,p_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,p_hash))
    return list(set(a_dl_files) & set(b_dl_files))



# c = gitiles_get_commit_info(
#     'https://android.googlesource.com/platform/frameworks/av/+/e7142a0703bc93f75e213e96ebc19000022afed9')
# print(c)
# print(gitiles_download_commit_files(c))
