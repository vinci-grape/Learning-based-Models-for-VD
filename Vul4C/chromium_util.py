
from utils import get_safe,convert_short_url_to_origin_url,debug
from bs4 import BeautifulSoup
from typing import Union
import pandas as pd
def chromium_find_commit_from_code_review(review_url:str) -> Union[None,str]:
    review_page = BeautifulSoup(get_safe(review_url).text,'html.parser')
    issue_desc = review_page.find('div',id='issue-description')
    if issue_desc is None:
        return None
    issue_desc = issue_desc.text
    issue_desc = list(filter(lambda x:'Committed: http' in x , issue_desc.split('\n')))
    if len(issue_desc) > 0:
        url = issue_desc[0]
        url = url[url.index('http'):]
        # filter http://src.chromium.org/viewvc/chrome?view=rev&revision=86862 url
        if 'viewvc' in url:
            return None
        # convert https://crrev.com/6703b5a51cedaa0ead73047d969f8c04362f51f1 to https://chromium.googlesource.com/xxx
        url = convert_short_url_to_origin_url(url)
        # only select contains `googlesource` keyword url
        if 'googlesource' not in url:
            return None
        return url
    return None


chromium_log = open('cache/chromium_log.txt', 'r')
chromium_log_lines = chromium_log.readlines()
chromium_log_msg2hash = {  }
for l in chromium_log_lines:
    hash,msg = l.split('||||||||||')
    msg = msg.strip()
    if msg in chromium_log_msg2hash.keys():
        chromium_log_msg2hash[msg] = None
    chromium_log_msg2hash[msg] = hash


def chromium_find_commit_from_viewvc(viewvc_url:str) -> Union[None,str]:

    viewvc_page = BeautifulSoup(get_safe(viewvc_url).text,'html.parser')
    vc_log = viewvc_page.find('pre',class_='vc_log')
    if vc_log is None:
        debug(f'[Chromium] vc_log is not found! {viewvc_url}')
        return None
    vc_log = vc_log.text.split('\n')[0]
    if vc_log in chromium_log_msg2hash.keys():
        hash = chromium_log_msg2hash[vc_log]
        if hash is None:
            debug(f'[Chromium] commit subject found duplicate! {viewvc_url}')
            return None
        url = f'https://github.com/chromium/chromium/commit/{hash}'
        return url
    else:
        debug(f'[Chromium] commit subject not found! {viewvc_url}')
        return None


# chromium_find_commit_from_code_review('https://src.chromium.org/viewvc/blink?revision=200098&view=revision')