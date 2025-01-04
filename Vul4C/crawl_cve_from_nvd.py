import json
import math
import time

import requests
import urllib
import pandas as pd
from pathlib import Path
import datetime
from urllib.request import Request, urlopen
from tqdm import tqdm

proxies = {
   'http': 'http://127.0.0.1:7890',
   'https': 'https://127.0.0.1:7890',
}

proxy_handler = urllib.request.ProxyHandler(proxies)
opener = urllib.request.build_opener()
RESULT_PER_PAGE = 1000
def compose_page_url(page_index: int, results_per_page: int = RESULT_PER_PAGE):
    return f'https://services.nvd.nist.gov/rest/json/cves/2.0/?resultsPerPage={results_per_page}&startIndex={page_index * RESULT_PER_PAGE}'


def read_json_from_network(url: str) -> dict:
    while True:
        try:
            content = opener.open(url).read()
            data = json.loads(content)
            break
        except UnicodeDecodeError as e:
            print('decode error' , e)
            time.sleep(5)
    return data

def read_json_from_local(path: str) -> list:
    return json.load(Path(path).open(mode='r'))

def cache_json(file_path:str,data:list,overwrite:bool = False):
    file_path = Path(file_path)
    if file_path.exists() and not overwrite:
        print(f'{file_path} exists, using `overwrite` flag to overwrite this file')
        return
    json.dump(data,file_path.open(mode='w'),indent=4)

def crawl_nvd():
    cnt_pages = int(math.ceil(read_json_from_network(compose_page_url(0))['totalResults'] / RESULT_PER_PAGE))
    # ['resultsPerPage', 'startIndex', 'totalResults', 'format', 'version', 'timestamp', 'vulnerabilities']
    time.sleep(5)
    all_cve = []
    cache_file_path ='cache/cve_data'
    if Path(cache_file_path).exists():
        print('NVD databases has been crawl. return')
        return
    for p in tqdm(range(cnt_pages)):
        cache_path = f'cache/pages/{RESULT_PER_PAGE}_{p}'
        if Path(cache_path).exists():
            print(f'using cache in {cache_path}')
            all_cve.extend(read_json_from_local(cache_path))
            continue
        url = compose_page_url(p)
        print()
        print(url)
        data = read_json_from_network(url)
        all_cve.extend(data['vulnerabilities'])
        cache_json(cache_path,data['vulnerabilities'])
        time.sleep(5)

    print('crawl DONE! cache file...')
    cache_json(cache_file_path,all_cve)


if __name__ == '__main__':
    crawl_nvd()
