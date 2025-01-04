import time
from http.client import HTTPResponse
from pathlib import Path

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from utils import debug, CommitInfo, read_commit_info_from_cache, write_commit_info_to_cache, proxies, get_safe, \
    save_str2file, trunc_commit_file_name, cache_commit_file_dir , check_file_exist
from typing import Union
# commit parent
# get = partial(get,timeout=10)
def cgit_get_commit_info(url:str) -> Union[CommitInfo,None]:
    debug(f'[CGit] {url}')
    id = url
    cache_db = 'cgit'
    # commit_info  = read_commit_info_from_cache(id,cache_db)
    # if commit_info is not None:
    #     return commit_info

    base_url = urlparse(url)
    base_url = f'{base_url.scheme}://{base_url.netloc}'
    responses = get_safe(url)
    cgit_page = BeautifulSoup(responses.text,'html.parser')
    if cgit_page.find('div',class_='error'):
        debug(f'[CGit] repositories not found {url}')
        return None
    commit_info = cgit_page.find('table', class_='commit-info')
    # todo obejct-header
    if commit_info is None:
        debug(f'[CGit] commit info not found {url}')
        return None

    commit_info = commit_info.find_all('tr')
    commit_hash = commit_info[2].a.text
    tree_hash = commit_info[3].a.text
    tree_url = urlparse(f'{base_url}{commit_info[3].a["href"]}')
    tree_url = f'{base_url}{tree_url.path}'
    tree_url = tree_url.replace('/tree/','/plain/')
    parent_commit_hash = commit_info[4].a.text
    if len(parent_commit_hash) == 0:
        parent_commit_hash = None
    commit_msg = cgit_page.find('div',class_='commit-msg').text

    diff_url = cgit_page.find('div',class_='diffstat-header').a['href']
    diff_url = f'{base_url}{diff_url}'
    diff_page = BeautifulSoup(get_safe(diff_url).text,'html.parser')
    diff_heads = diff_page.find('table', class_='diff').find_all('div', class_='head')
    diff_files = []

    for head in diff_heads:
        file = head.a.text
        diff_files.append(file)

    repo_name = cgit_page.find('td',class_='main').find('a',{'title':True}).text

    c = CommitInfo(id,repo_name,commit_hash,parent_commit_hash,diff_files,tree_url,commit_msg,url)
    # write_commit_info_to_cache(c, cache_db)
    return c

def cgit_download_tree_files(tree_url:str,tree_hash:str,files:list[str],save_dir:Path):
    for f in files:
        trunc_name = trunc_commit_file_name(f)
        # cache
        if check_file_exist(save_dir/trunc_name):
            continue
        download_url = f'{tree_url}{f}?id={tree_hash}'
        save_str2file(get_safe(download_url).text,save_dir/trunc_name)

def cgit_download_commit_files(commit_info:CommitInfo):
    # download file
    debug(f'[CGit-Download] {commit_info.repo_name} {commit_info.commit_hash}')
    files = commit_info.diff_file_paths
    c_hash , p_hash = commit_info.commit_hash , commit_info.parent_commit_hash
    # commit tree
    cgit_download_tree_files(commit_info.tree_url,c_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,c_hash))
    # parent commit tree
    cgit_download_tree_files(commit_info.tree_url,p_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,p_hash))



# c= cgit_get_commit_info('https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=b525c06cdbd8a3963f0173ccd23f9147d4c384b5')
# c= cgit_get_commit_info('https://git.savannah.gnu.org/cgit/emacs.git/commit/?id=01a4035c869b91c153af9a9132c87adb7669ea1c')
# print(c)
# cgit_download_commit_files(c)