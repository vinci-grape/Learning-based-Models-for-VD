
from http.client import HTTPResponse
from pathlib import Path

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from utils import debug, CommitInfo, read_commit_info_from_cache, write_commit_info_to_cache, proxies, get_safe, \
    cache_commit_file_dir, trunc_commit_file_name , save_str2file , check_file_exist
from typing import Union
import unicodedata
# commit parent
# get = partial(get,timeout=10)
def gitweb_get_commit_info(url:str) -> Union[CommitInfo,None]:
    debug(f'[GitWeb] {url}')
    base_url = urlparse(url)
    base_url = f'{base_url.scheme}://{base_url.netloc}'
    gitweb_page = BeautifulSoup(get_safe(url).text,'html.parser')
    object_headers = gitweb_page.find('table', class_='object_header')
    if object_headers is None:
        debug(f'[GitWeb] table not found! {url}')
        return None

    object_headers = object_headers.find_all('tr')
    # https://sourceware.org/git/gitweb.cgi?p=glibc.git;a=commitdiff;h=d527c860f5a3f0ed687bd03f0cb464612dc23408
    # some url missing commit hash
    if len(object_headers) != 7:
        debug(f'[GitWeb] table len != 7 {url}')
        return None
    commit_hash = object_headers[4].find_all('td')[1].text
    tree_hash = commit_hash
    parent_commit_hash = object_headers[6].find_all('td')[1].text

    # remove tree_hash and commit_hash
    tree_url_prefix,tree_url_postfix = object_headers[5].find('td',class_='link').a['href'].split('?')
    tree_url_postfix  = tree_url_postfix.split(';')
    tree_url_postfix = list(filter(lambda s:s.startswith('p='),tree_url_postfix))
    tree_url = f'{base_url}{tree_url_prefix}?{";".join(tree_url_postfix)}'

    # parse diff file
    diff_files = []
    diff_tree = gitweb_page.find('table',class_='diff_tree').find_all('tr')
    for file in diff_tree:
        diff_files.append(file.find('td').a.text)

    commit_msg = unicodedata.normalize('NFKD',gitweb_page.find('div',class_='page_body').text)

    repo_name = ''
    for idx,a in enumerate(gitweb_page.find('div',class_='page_header').find_all('a')[2:]):
        if idx > 0 :
            repo_name += '/'
        repo_name += a.text

    commit_info = CommitInfo(url,repo_name,commit_hash,parent_commit_hash,diff_files,tree_url,commit_msg,url)
    return commit_info

def gitweb_search_page_commit_url(search_page_url:str) -> list[str]:
    # find search page all commit url
    # https://git.moodle.org/gw?p=moodle.git&a=search&h=HEAD&st=commit&s=MDL-76810
    assert 'a=search' in search_page_url
    base_url = urlparse(search_page_url)
    base_url = f'{base_url.scheme}://{base_url.netloc}'
    search_page = get_safe(search_page_url)
    search_page = BeautifulSoup(search_page.text,'html.parser')
    commit_links = search_page.find('table', class_='commit_search')
    if commit_links is None:
        debug('[GitWeb] search page not found')
        return []
    commit_links = commit_links.find_all('td', class_='link')

    commit_urls = []
    for c in commit_links:
        url = f'{base_url}{c.a["href"]}'
        commit_urls.append(url)
    debug('[GitWeb] search page found',commit_urls)
    return commit_urls

def gitweb_download_tree_files(tree_url:str,tree_hash:str,files:list[str],save_dir:Path):
    download_base_url = f'{tree_url};a=blob_plain;hb={tree_hash};'
    for f in files:
        trunc_name = trunc_commit_file_name(f)
        # cache
        if check_file_exist(save_dir/trunc_name):
            continue
        download_url = f'{download_base_url}f={f}'
        save_str2file(get_safe(download_url).text,save_dir/trunc_name)

def gitweb_download_commit_files(commit_info:CommitInfo):
    # download file
    debug(f'[GitWeb-Download] {commit_info.repo_name} {commit_info.commit_hash}')
    files = commit_info.diff_file_paths
    c_hash , p_hash = commit_info.commit_hash , commit_info.parent_commit_hash
    # commit tree
    gitweb_download_tree_files(commit_info.tree_url,c_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,c_hash))
    # parent commit tree
    gitweb_download_tree_files(commit_info.tree_url,p_hash,files,cache_commit_file_dir(commit_info.repo_name,c_hash,p_hash))


# gitweb_search_page_commit_url('https://git.moodle.org/gw?p=moodle.git&a=search&h=HEAD&st=commit&s=MDL-76810')
# https://sourceware.org/git/?p=glibc.git;a=tree;hb=5171f3079f2cc53e0548fc4967361f4d1ce9d7ea
# c = gitweb_get_commit_info(
#     'https://sourceware.org/git/gitweb.cgi?p=glibc.git;h=ddc650e9b3dc916eab417ce9f79e67337b05035c')
# gitweb_download_commit_files(c)
# print(c)
