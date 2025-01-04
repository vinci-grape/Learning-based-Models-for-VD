
from typing import Union
from urllib.parse import ParseResult,urlparse
import re
from github_util import github_get_commit_info,github_download_commit_files
from cgit_util import cgit_get_commit_info, cgit_download_commit_files
from gitlab_util import gitlab_get_commit_info, gitlab_download_commit_files
from gitweb_util import gitweb_get_commit_info, gitweb_download_commit_files
from gitiles_util import gitiles_get_commit_info, gitiles_download_commit_files
from utils import CommitInfo , repo_name_try_merge,filter_accepted_files

    def extract_commit_info_from_url(url:str) ->  Union[None,CommitInfo] :
    parse_url : ParseResult = urlparse(url)
    nloc = parse_url.netloc
    commit_info = None
    if nloc == 'github.com':
        # match common commit url
        if (re_result := re.match(r"https://github.com/(?P<owner>.*)/(?P<repo>.*)/commit/(?P<commit_hash>.*)", url)) is not None:
            owner =re_result.group('owner')
            repo = re_result.group('repo')
            commit_hash = re.split(r'[#.]',re_result.group('commit_hash'))[0]
            commit_info = github_get_commit_info(f'{owner}/{repo}',commit_hash,url)
    elif nloc == 'gitlab.com':
        if (re_result := re.match(r"https://gitlab.com/(?P<owner>.*)/(?P<repo>.*)/-/commit/(?P<commit_hash>.*)", url)) is not None:
            owner =re_result.group('owner')
            repo = re_result.group('repo')
            commit_hash = re_result.group('commit_hash')
            commit_info = gitlab_get_commit_info(f'{owner}/{repo}',commit_hash,url)
    elif nloc in ['sourceware.org','git.videolan.org','git.moodle.org','git.openssl.org']:
        commit_info = gitweb_get_commit_info(url)
        if commit_info is None:
            return None
        commit_info.repo_name = repo_name_try_merge(commit_info.repo_name)
    elif nloc in ['git.kernel.org','cgit.freedesktop.org','git.savannah.gnu.org']:
        commit_info = cgit_get_commit_info(url)
        if commit_info is None:
            return None
        if nloc == 'git.kernel.org':
            commit_info.repo_name = 'torvalds/linux'
        elif nloc in ['cgit.freedesktop.org','git.savannah.gnu.org']:
            repo_name = commit_info.repo_name
            repo_name = repo_name_try_merge(repo_name)
            commit_info.repo_name = repo_name
    elif 'googlesource.com' in url:
        # android.googlesource.com , chromium.googlesource.com , skia.googlesource.com
        url = url.replace('%5E%21/','').replace('%5E!/','') # diff page to normal page
        # https://android.googlesource.com/platform/frameworks/av/+/d07f5c14e811951ff9b411ceb84e7288e0d04aaf
        # https://android.googlesource.com/platform/frameworks/av/+/d07f5c14e811951ff9b411ceb84e7288e0d04aaf%5E%21/
        url = url.replace(',','')   # some url contain `,`
        commit_info = gitiles_get_commit_info(url)

    if commit_info:
        # only download C,CPP file
        # you can change download behavior in `AcceptedFileExtension`
        commit_info.diff_file_paths = filter_accepted_files(commit_info.diff_file_paths)
        if len(commit_info.diff_file_paths) == 0:
            commit_info = None


    if commit_info:

        if nloc == 'github.com':
            commit_info.diff_file_paths =  github_download_commit_files(commit_info)
        elif nloc == 'gitlab.com':
            commit_info = gitlab_download_commit_files(commit_info)
        elif nloc in ['sourceware.org', 'git.videolan.org', 'git.moodle.org', 'git.openssl.org']:
            gitweb_download_commit_files(commit_info)
        elif nloc in ['git.kernel.org', 'cgit.freedesktop.org', 'git.savannah.gnu.org']:
            cgit_download_commit_files(commit_info)
        elif 'googlesource.com' in url:
            commit_info.diff_file_paths = gitiles_download_commit_files(commit_info)
        else:
            pass

        if len(commit_info.diff_file_paths) == 0:
            commit_info = None

    return commit_info


def extract_commit_info_from_urls(url_references : list[str]) -> list[CommitInfo]:
    infos = []
    for r in url_references:
        commit_info = extract_commit_info_from_url(r)
        if commit_info:
            # we need filter out same commit from different urls
            # e.g. https://nvd.nist.gov/vuln/detail/CVE-2011-3637
            find_duplicate = False
            for p_c in infos:
                if p_c.repo_name == commit_info.repo_name and p_c.commit_hash == commit_info.commit_hash:
                    find_duplicate = True

            if not find_duplicate:
                infos.append(commit_info)
    return infos