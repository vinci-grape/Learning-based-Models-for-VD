import time
from pathlib import Path

import github.GithubException
import requests
from github import Github
from github.PaginatedList import PaginatedList
from github.Commit import Commit
from github.GithubException import RateLimitExceededException
from tinydb import TinyDB, Query
from utils import CommitInfo, debug, read_commit_info_from_cache, write_commit_info_to_cache, repo_name_try_merge, \
    repo_name_try_recover, cache_commit_file_dir, trunc_commit_file_name, try_decode_binary_data2file, \
    filter_accepted_files,check_file_exist
from typing import Union
import re
from urllib3.util import Retry
import random
import base64

GitHub_Tokens = ['your github token'

                 ]
g_maps = {}
for idx, token in enumerate(GitHub_Tokens):
    g_maps[idx] = Github(token, retry=Retry())


def random_g() -> Github:
    idx = random.randint(0, len(g_maps) - 1)
    return g_maps[idx]


def github_find_commit_from_commit_msg(repo_name: str, msg: str, regex_match: Union[str, None] = None):
    while True:
        try:
            search_result: PaginatedList = random_g().search_commits(query=f'repo:{repo_name} merge:false {msg}')

            candidate_commit = []
            c: Commit
            for c in search_result:
                if (regex_match is not None) and (re.search(regex_match, c.commit.message.lower()) is not None):
                    candidate_commit.append(c.html_url)
                elif (regex_match is None) and (msg.lower() in c.commit.message.lower()):
                    candidate_commit.append(c.html_url)

            return candidate_commit
        except RateLimitExceededException as e:
            debug('[GitHub] rate limit exceeded sleep 60s retrying...')
            time.sleep(60)


# github_find_commit_from_commit_msg('xen-project/xen','xsa-80')
# github_find_commit_from_commit_msg('chromium/chromium','[FileAPI] Clean up WebFileSystemImpl before Blink shutdown')
def github_get_commit_info(repo_name: str, commit_hash: str,url:str) -> Union[CommitInfo, None]:
    debug(f'[GitHub] {repo_name} {commit_hash}')
    id = f'{repo_name}/{commit_hash}'
    cache_db = 'github'
    # commit_info  = read_commit_info_from_cache(id,cache_db)
    # if commit_info is not None:
    #     return commit_info
    while True:
        try:
            repo = random_g().get_repo(repo_name)
            commit = repo.get_commit(commit_hash)
            commit_msg = commit.commit.message
            diff_files = []
            for f in commit.files:
                diff_files.append(f.filename)
            parent_commit_hash = commit.parents[0].sha if len(commit.parents) > 0 else None
            tree_url = ''
            repo_name = repo_name_try_merge(repo_name)
            commit_info = CommitInfo(id, repo_name, commit_hash, parent_commit_hash, diff_files, tree_url, commit_msg,url)
            # write_commit_info_to_cache(commit_info,cache_db)
            return commit_info
        except github.GithubException as e:
            if isinstance(e, github.RateLimitExceededException):
                debug(f'[GitHub] RateLimitExceededException sleep 60s')
                time.sleep(60)
                continue
            if isinstance(e, github.UnknownObjectException):
                debug(f'[Github] repo {repo_name}:{commit_hash} missing')
                return None
            debug(f'[GitHub] unknown exception {repo_name}/{commit_hash} {e}')
            return None
        except requests.exceptions.RequestException as e:
            continue


def github_download_commit_single_file(repo_name: str, commit_hash: str, file_path: str) -> str:
    repo = random_g().get_repo(repo_name)
    content = repo.get_contents(file_path, commit_hash)
    print(content.encoding)
    return content.decoded_content.decode('utf-8')


def github_download_tree_files(repo_name, tree_hash: str, files: list[str], save_dir: Path) -> list[str]:
    while True:
        try:
            repo = None
            dl_files:list[str] = []
            for f in files:
                trunc_name = trunc_commit_file_name(f)
                # cache
                if check_file_exist(save_dir/trunc_name):
                    dl_files.append(f)
                    continue
                if repo is None:
                    repo = random_g().get_repo(repo_name)
                try:
                    content = repo.get_contents(f, tree_hash)
                    if content.encoding != 'base64':
                        debug(f'[Github] file {repo_name}:{tree_hash} {f} encoding none')
                        # if file size > 1MB , we should download the file from git glob
                        file_content_b = base64.b64decode(repo.get_git_blob(content.sha).content)
                    else:
                        file_content_b = content.decoded_content
                    try_decode_binary_data2file(file_content_b, save_dir / trunc_name)
                    dl_files.append(f)
                except github.GithubException as e:
                    debug(f'[Github] file {repo_name}:{tree_hash} {f} missing {e}')
                    continue
            return dl_files
        except github.RateLimitExceededException:
            debug(f'[GitHub] RateLimitExceededException sleep 60s')
            time.sleep(60)
            continue
        except requests.exceptions.RequestException as e:
            continue


def github_download_commit_files(commit_info: CommitInfo) -> list[str]:
    # download file
    debug(f'[GitHub-Download] downloading... {commit_info.repo_name} {commit_info.commit_hash}')
    repo_name = repo_name_try_recover(commit_info.repo_name)
    files = commit_info.diff_file_paths
    c_hash, p_hash = commit_info.commit_hash, commit_info.parent_commit_hash
    # commit tree
    a_dl_files = github_download_tree_files(repo_name, c_hash, files, cache_commit_file_dir(commit_info.repo_name, c_hash, c_hash))
    # parent commit tree
    b_dl_files = github_download_tree_files(repo_name, p_hash, files, cache_commit_file_dir(commit_info.repo_name, c_hash, p_hash))
    return list(set(a_dl_files) & set(b_dl_files))

# , 'tests/TESTLIST', 'tests/rx_ubik-oobr.out', 'tests/rx_ubik-oobr.pcap']
# print(
#     github_download_commit_files(github_get_commit_info('the-tcpdump-group/tcpdump', 'aa0858100096a3490edf93034a80e66a4d61aad5',"")))



# print(
#     github_download_commit_files(github_get_commit_info('wireshark/wireshark', '00d5e9e9fb377f52ab7696f25c1dbc011ef0244d')))
