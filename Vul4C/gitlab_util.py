from pathlib import Path

import gitlab
from gitlab.v4.objects.projects import Project
import requests

from utils import CommitInfo, read_commit_info_from_cache, write_commit_info_to_cache, debug, proxies, \
    repo_name_try_merge, repo_name_try_recover, cache_commit_file_dir, trunc_commit_file_name, \
    try_decode_binary_data2file, check_file_exist
from typing import Union
session = requests.Session()
session.proxies = proxies
gl = gitlab.Gitlab(timeout=5,session=session)

# https://python-gitlab.readthedocs.io/en/stable/gl_objects/projects.html

def gitlab_get_commit_info(repo_name: str, commit_hash: str,url) -> Union[CommitInfo,None]:
    debug(f'[GitLab] {repo_name} {commit_hash}')
    id = f'{repo_name}/{commit_hash}'
    cache_db = 'gitlab'
    # commit_info  = read_commit_info_from_cache(id,cache_db)
    # if commit_info is not None:
    #     return commit_info
    project: Project = gl.projects.get(repo_name)
    commit = project.commits.get(commit_hash)
    commit_json : dict = commit.asdict()
    parent_commit_hash = commit_json['parent_ids'][0] if len(commit_json['parent_ids']) > 0 else None
    commit_msg = commit_json['message']
    diff_files = []
    for diff in commit.diff(get_all=True):
        diff_files.append(diff['new_path'])
    tree_url = ''
    repo_name = repo_name_try_merge(repo_name)
    commit_info = CommitInfo(id,repo_name ,commit_hash, parent_commit_hash, diff_files, tree_url, commit_msg,url)
    # write_commit_info_to_cache(commit_info, cache_db)
    return commit_info

def gitlab_download_tree_files(repo_name,tree_hash:str,files:list[str],save_dir:Path):
    repo = gl.projects.get(repo_name)
    for f in files:
        trunc_name = trunc_commit_file_name(f)
        # cache
        if check_file_exist(save_dir/trunc_name):
            continue
        file_content_b = repo.files.raw(f,ref=tree_hash)
        try_decode_binary_data2file(file_content_b,save_dir/trunc_name)


def gitlab_download_commit_files(commit_info:CommitInfo):
    # download file
    debug(f'[GitLab-Download] {commit_info.repo_name} {commit_info.commit_hash}')
    repo_name  = repo_name_try_recover(commit_info.repo_name)
    files = commit_info.diff_file_paths
    c_hash , p_hash = commit_info.commit_hash , commit_info.parent_commit_hash
    # commit tree
    gitlab_download_tree_files(repo_name,c_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,c_hash))
    # parent commit tree
    gitlab_download_tree_files(repo_name,p_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,p_hash))



# c = gitlab_get_commit_info('gnutls/gnutls', '94fcf1645ea17223237aaf8d19132e004afddc1a')
# print(c)
# gitlab_download_commit_files(c)
