import dataclasses
import json

from extract_commit_info import extract_commit_info_from_urls
from xen_util import xen_find_github_commit_from_advisory
from utils import *
from tqdm import tqdm
from dataclasses import dataclass
from typing import Union
from urllib.parse import urlparse
from tqdm.contrib.concurrent import  process_map
import re
from chromium_util import chromium_find_commit_from_code_review,chromium_find_commit_from_viewvc
from gitweb_util import gitweb_search_page_commit_url
@dataclass
class Cvss2Metrics:
    access_vector: Union[str, None] = None  # [Local, Adjacent Network, Network]
    access_complexity: Union[str, None] = None  # [High, Medium, Low]
    authentication: Union[str, None] = None  # [Multiple, Single, None]
    confidentiality_impact: Union[str, None] = None  # [None, Partial, Complete]
    integrity_impact: Union[str, None] = None  # [None, Partial, Complete]
    availability_impact: Union[str, None] = None  # [None, Partial, Complete]
    cvss_score: Union[float, None] = None  # [None, Partial, Complete]
    severity: Union[str, None] = None  # [None, Partial, Complete]


def extract_en_description(descriptions: list) -> str:
    for d in descriptions:
        if d['lang'] == 'en':
            return d['value']
    return ''


def extract_cvss2_metrics(metrics: dict) -> Cvss2Metrics:
    if 'cvssMetricV2' in metrics:
        metrics = metrics['cvssMetricV2'][0]
        cvss_data = metrics['cvssData']
        access_vector = cvss_data['accessVector'].capitalize()
        access_complexity = cvss_data['accessComplexity'].capitalize()
        authentication = cvss_data['authentication'].capitalize()
        confidentiality_impact = cvss_data['confidentialityImpact'].capitalize()
        integrity_impact = cvss_data['integrityImpact'].capitalize()
        availability_impact = cvss_data['availabilityImpact'].capitalize()
        base_score = cvss_data['baseScore']
        severity = metrics['baseSeverity'].capitalize()
        return Cvss2Metrics(access_vector, access_complexity, authentication, confidentiality_impact, integrity_impact,
                            availability_impact, base_score, severity)
    else:
        return Cvss2Metrics()


def extract_cwe_id(weaknesses: list):
    cwe_other = ['NVD-CWE-noinfo', 'NVD-CWE-Other']
    cwe_id = set()
    for w in weaknesses:
        if w['type'] == 'Primary':
            for d in w['description']:
                if d['lang'] == 'en':
                    if d['value'] not in cwe_other:
                        cwe_id.add(d['value'])
    if len(cwe_id) == 0:
        for w in weaknesses:
            for d in w['description']:
                if d['lang'] == 'en':
                    if d['value'] not in cwe_other:
                        cwe_id.add(d['value'])
    if len(cwe_id) == 0:
        cwe_id.add('CWE-Other')
    cwe_id = list(cwe_id)
    # print(cve_id,cwe_id)
    return cwe_id


def extract_reference_url(url_references: list):
    """ extract url and fix some corrupted url  """
    url_result = []
    for ref in url_references:
        url: str = ref['url']
        # if url !='https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git':
        #     continue
        parse_url = urlparse(url)
        nloc = parse_url.netloc
        if nloc == 'github.com'  and ('commit' in url):
            url_result.append(url)
        elif nloc == 'git.kernel.org' and ('commit' in url):
            # print(url)
            url = url.replace('%3B', ';').replace('a=commitdiff_plain',
                                                  'a=commitdiff')  # some commit url has `%3B` ,  replace it with `;`
            # remove linux version number (e.g. linux-2.6 linux-2.3.65 to linux ,  testing-2.6.git to testing.git )
            if (version_re := re.search(r'(\w*?-[0-9]\.[\w.]*?\.git)', url)) is not  None:
                replace_str = version_re.group(0)
                new_name = f"{replace_str.split('-')[0]}.git"
                url = url.replace(replace_str,new_name)
            # filter missing commit hash url
            if not(("id=" in url) or ("h=" in url)):
                continue
            url_result.append(url)
        elif nloc == 'sourceware.org' and ('h=' in url):
            url_result.append(url)
            pass
        elif nloc == 'android.googlesource.com':
            url_result.append(url)
            pass
        elif nloc == 'xenbits.xen.org' and ('advisory' in url) :
            github_urls = xen_find_github_commit_from_advisory(url)
            url_result.extend(github_urls)
            pass
        elif nloc == 'code.wireshark.org' and ('h=' in url):
            # wireshark url transform to GitHub url
            # https://code.wireshark.org/review/gitweb?p=wireshark.git;a=commit;h=5b4ada17723ed8af7e85cb48d537437ed614e417
            url = url.replace('%3B', ';')
            commit_hash = url[url.index('h=') + 2:]
            url = f'https://github.com/wireshark/wireshark/commit/{commit_hash}'
            url_result.append(url)
        elif nloc == 'codereview.chromium.org':
            commit_url = chromium_find_commit_from_code_review(url)
            if commit_url is not None:
                url_result.append(commit_url)
            pass
        elif nloc == 'git.videolan.org' and ('h=' in url):
            url = url.replace('%3B', ';').replace('a=commitdiff','a=commit')
            url_result.append(url)
        elif nloc == 'git.moodle.org':
            pass
            if 'a=commit' in url:
                url_result.append(url)
            elif 'a=search' in url:
                url_result.extend(gitweb_search_page_commit_url(url))
        elif nloc == 'cgit.freedesktop.org':
            pass
            if 'commit' in url:
                url_result.append(url)
            elif 'diff' in url:
            #     replace diff to commit
                url = url.replace('diff','commit')
                url_result.append(url)
        elif nloc == 'git.gnome.org':
            # transform Gnome self hosted gitlab url to github url
            github_url = convert_short_url_to_origin_url(url)
            if github_url != 'https://gitlab.gnome.org/users/sign_in':
                github_url = github_url.replace('http://','https://').replace('gitlab.gnome.org','github.com').replace('browse','GNOME').replace('/-/','/')
                url_result.append(github_url)
        elif nloc == 'gitlab.gnome.org':
            # transform Gnome self hosted gitlab url to github url
            pass
            if 'commit' in url:
                url = url.replace('http://','https://').replace('gitlab.gnome.org','github.com').replace('browse','GNOME').replace('/-/','/')
                url_result.append(url)
        elif nloc == 'git.openssl.org':
            pass
            url = url.replace('%3B', ';').replace('a=commitdiff', 'a=commit')
            if 'a=commit' in url:
                url_result.append(url)
        elif nloc == 'git.savannah.gnu.org':
            pass
            url = url.replace('%3B', ';').replace('a=commitdiff','a=commit').replace('patch','commit')
            if 'commit' in url:
                url_result.append(url)
        elif nloc == 'git.ghostscript.com':
            # ghostscript server is down. transform url to github url
            # http://git.ghostscript.com/?p=mupdf.git;h=96751b25462f83d6e16a9afaf8980b0c3f979c8b
            url = url.replace('%3B', ';').replace('a=commitdiff','a=commit')
            url = url.split('?')[1].split(';')
            repo = None
            commit_hash = None
            for u in url:
                if u.startswith('p='):
                    repo = u[2:].split('.')[0]
                elif u.startswith('h='):
                    commit_hash = u[2:]
            url = f'https://github.com/ArtifexSoftware/{repo}/commit/{commit_hash}'
            url_result.append(url)
        elif nloc == 'gitlab.freedesktop.org':
            path = parse_url.path
            if 'commit' in path:
                path = path.replace('/-/','/').replace('/commit/','/')
                commit_hash = path.split('/')[-1]
                repo_mapping = {
                    'xorg/lib/libxpm' : 'xorg/lib/libXpm',
                    'virgl/virglrenderer':'virglrenderer',
                    'xorg/lib/libx11' : 'xorg/lib/libX11',
                    'polkit/polkit' : 'polkit',
                }
                repo = '/'.join(path[1:].split('/')[:-1])
                if repo in repo_mapping.keys():
                    repo = repo_mapping[repo]
                url = f'https://cgit.freedesktop.org/{repo}/commit/?id={commit_hash}'
                url_result.append(url)
        elif nloc =='src.chromium.org':
            if 'viewvc' in url:
                commit_url = chromium_find_commit_from_viewvc(url)
                if commit_url is not None:
                    url_result.append(commit_url)
        else:
            pass
    return url_result


cve_data = read_json_file('cache/cve_data')
print(len(cve_data))
all_url = []
DUMP_URL_OCCUR_100_FLAG = False

def parse_cve(cve_row):
    cve_row = cve_row['cve']
    # print(cve_row)
    # dict_keys(['id', 'sourceIdentifier', 'published', 'lastModified', 'vulnStatus', 'descriptions', 'metrics', 'weaknesses', 'configurations', 'references'])
    # print(cve_row.keys())
    cve_id = cve_row['id']
    publish_date = cve_row['published']
    update_date = cve_row['lastModified']
    description = extract_en_description(cve_row['descriptions'])

    metrics = cve_row['metrics']
    cvss2_metric = extract_cvss2_metrics(metrics)

    if 'weaknesses' not in cve_row.keys():
        return
    cwe_list = extract_cwe_id(cve_row['weaknesses'])

    # print(cve_id)
    references = extract_reference_url(cve_row['references'])
    references = list(set(references))  # remove duplicate url
    commit_infos = extract_commit_info_from_urls(references)    # get commit info and download


    # if len(references) > 1 :
    #     print(cve_id , references)
    if DUMP_URL_OCCUR_100_FLAG:
        all_url.extend(references)
    # if len(cwe_list) == 0:
    #     print(cve_id)
    # assert len(weaknesses) == 1 , print(cve_id,weaknesses)
    # print(cve_id , weaknesses)
    res = { 'cve' : cve_id , 'cwe_list' : cwe_list , 'description':description ,
    'publish_data' : publish_date ,'update_data' : update_date,
            **dataclasses.asdict(cvss2_metric)
    ,'commits' :  [dataclasses.asdict(c) for c in commit_infos  ] }
    return res

cve_result:list = process_map(parse_cve,list(cve_data),chunksize=100)
cve_result =  [i for i in cve_result if i is not None]
print(f'result:{len(cve_result)}')
json.dump(cve_result, open('result/cve_commit_infos.json', mode='w'))
# process_map(parse_cve,cve_data,chunksize=100,max_workers=1)


def dump_url_occur_100():
    url_cnt_dict = {}
    url_res_dict = {}

    for u in all_url:
        netloc = urlparse(u).netloc
        if netloc in url_cnt_dict:
            url_cnt_dict[netloc] += 1
            url_res_dict[netloc].append(u)
        else:
            url_cnt_dict[netloc] = 1
            url_res_dict[netloc] = list()
            url_res_dict[netloc].append(u)

    print(len(all_url))

    url_occur_100 = {k: v for k, v in
                     sorted(filter(lambda item: item[1] > 100, url_cnt_dict.items()), key=lambda item: item[1],
                            reverse=True)}
    print(len(url_occur_100))
    for k, v in url_occur_100.items():
        with open(f'./tmp/{k}', mode='w') as f:
            f.write('\n'.join(url_res_dict[k]))


if DUMP_URL_OCCUR_100_FLAG:
    dump_url_occur_100()
