from utils import get_safe
from bs4 import BeautifulSoup
from github_util import github_find_commit_from_commit_msg
def xen_find_github_commit_from_advisory(advisory_url:str):
    response = get_safe(advisory_url)
    advisory_page = BeautifulSoup(response.text,'html.parser')
    xsa_id : str = advisory_page.find('table').tr.td.a.text.lower()

    # only find the commit contains the following msg
    # This is XSA-83.
    # This is CVE-2014-2986 / XSA-94.
    # This is XSA-91 / CVE-2014-3125.
    # This is part of CVE-2014-5147 / XSA-102.
    # This is part of XSA-222.

    match_re = rf'this is (cve[^\s]* / )?{xsa_id}|this is part of (cve[^\s]* / )?{xsa_id}'
    return github_find_commit_from_commit_msg('xen-project/xen',xsa_id,regex_match=match_re)


# print(xen_find_github_commit_from_advisory('http://xenbits.xen.org/xsa/advisory-98.html'))