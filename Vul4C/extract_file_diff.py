import dataclasses
import difflib
import json
import pandas as pd

from abstract_code import CodeAbstracter
from utils import read_json_file, CommitInfo, cache_commit_file_dir, trunc_commit_file_name, from_dict_to_dataclass, \
    read_json_file_to_dataclass_list, ExtractedFunction, write_dataclasses_to_json, DiffFuncPair, DiffFile, \
    CommitInfoWithDiffInfo,NonVulFunction
from parser_c import ParserC
from parser_cpp import ParserCPP
from pathlib import Path
import shutil
from collections.abc import Iterator
from typing import Union
cve_commit_infos = read_json_file('result/cve_commit_infos.json')
def get_commit_key(commit_info:Union[dict,CommitInfo]):
    if isinstance(commit_info,dict):
        return f'{commit_info["repo_name"]}$${commit_info["commit_hash"]}'
    else:
        return f"{commit_info.repo_name}$${commit_info.commit_hash}"


def filter_parse_error_commit(cve_commit_infos):
    """
        some commits are difficult to parse, filter them

        cve_commit_infos
        ['cve', 'cwe_list', 'description', 'publish_data', 'update_data', 'access_vector', 'access_complexity',
        'authentication', 'confidentiality_impact', 'integrity_impact', 'availability_impact', 'cvss_score', 'severity', 'commits']

        commit
        ['id', 'repo_name', 'commit_hash', 'parent_commit_hash', 'diff_file_paths', 'tree_url', 'commit_msg']
    """
    filtered_error_commits_cnt = 0
    parse_error_commits = read_json_file("parse_error_commits.json")

    def is_parse_error_commit(commit) -> bool:
        repo_name = commit['repo_name']
        commit_hash = commit['commit_hash']
        if repo_name in parse_error_commits:
            error_commits = parse_error_commits[repo_name]
            if commit_hash in error_commits:
                return True
        return False

    for row in cve_commit_infos:
        new_commits = []
        for commit in row['commits']:
            if not is_parse_error_commit(commit):
                new_commits.append(commit)
            else:
                filtered_error_commits_cnt += 1
        row['commits'] = new_commits

    print(f'[Filter Parse Error Commit]:{filtered_error_commits_cnt}')
    return cve_commit_infos

def filter_duplicate_commit(cve_commit_infos):
    """
        a single commit may exist in multiple CVEs
        e.g.
            https://nvd.nist.gov/vuln/detail/CVE-2018-1000039
            https://nvd.nist.gov/vuln/detail/CVE-2018-1000037

            they all have commit
            http://git.ghostscript.com/?p=mupdf.git;a=commitdiff;h=71ceebcf56e682504da22c4035b39a2d451e8ffd;hp=7f82c01523505052615492f8e220f4348ba46995

        we will remove duplicate commits for now
    """
    commit_occur_cnt =  { }
    for row in cve_commit_infos:
        new_commits = []
        for commit in row['commits']:
            commit_key = get_commit_key(commit)
            commit_occur_cnt.setdefault(commit_key,0)
            commit_occur_cnt[commit_key] += 1
            if commit_occur_cnt[commit_key] <= 1:
                new_commits.append(commit)
        row['commits'] = new_commits


    # for row in cve_commit_infos:
    #     new_commits = []
    #     for commit in row['commits']:
    #         commit_key = get_commit_key(commit)
    #         if commit_occur_cnt[commit_key] <= 1:
    #             new_commits.append(commit)
    #     row['commits'] = new_commits
    return cve_commit_infos

cve_commit_infos = filter_parse_error_commit(cve_commit_infos)
cve_commit_infos = filter_duplicate_commit(cve_commit_infos)
cve_commit_infos = list(filter(lambda x:len(x['commits']) != 0 , cve_commit_infos))



# 15696 total file
def _traverse_commit_file(commit_info,repo_name:str,a_hash,b_hash):
    save_dir = cache_commit_file_dir(repo_name, a_hash, b_hash)
    for f in commit_info.diff_file_paths:
        trunc_name = trunc_commit_file_name(f)
        file = save_dir/trunc_name
        yield file
        assert file.exists()

def collect_all_commit_file(commit_info:CommitInfo):
    c_hash, p_hash = commit_info.commit_hash, commit_info.parent_commit_hash
    repo_name = commit_info.repo_name
    res = list(_traverse_commit_file(commit_info,repo_name,c_hash,c_hash)) + list(_traverse_commit_file(commit_info,repo_name,c_hash,p_hash))
    return list(map(lambda x:str(x),res))

project_type_mapping = { }  # we using this mapping determine `.h` file is from C project or CPP project
cmix_project = set()
def determine_project_type(all_f:list[str],repo_name):
    has_cpp_file = False
    has_c_file = False
    for f in all_f:
        ext = f.split('.')[-1]
        if ext == 'c':
            has_c_file = True
        elif ext in ['cpp','cc','hh','hxx','cxx','hpp']:
            has_cpp_file = True

    project_type = None
    if has_cpp_file and not has_c_file:
        project_type = 'CPP'
    elif has_c_file and not has_cpp_file:
        project_type = 'C'
    else:
        if repo_name in project_type_mapping.keys() and repo_name not in cmix_project:
            project_type = project_type_mapping[repo_name]

    if repo_name in cmix_project:
        return project_type

    if repo_name not in project_type_mapping.keys():
        project_type_mapping[repo_name] = project_type
    else:
        if project_type != project_type_mapping[repo_name]:
            cmix_project.add(repo_name)
            del project_type_mapping[repo_name]
            return None
    return project_type



fail_file = []
file_ext_cnt = {  }
# guess = Guess()


def traverse_all_commit() -> Iterator[CommitInfo]:
    for row in cve_commit_infos:
        for commit in row['commits']:
            commit_info = from_dict_to_dataclass(CommitInfo, commit)
            yield commit_info

# filling project type mapping


def extract_all_commit_function():
    total_commit_cnt = 0
    cannot_determine = 0
    commit_all_file_parse_success_cnt = 0
    cannot_determine_repo_name = []
    total_cnt = 0
    file_language_mapping =  { }    # record this file as C file or CPP file
    for row in cve_commit_infos:
        total_commit_cnt += len(row['commits'])
    for commit_info in traverse_all_commit():
        all_f = collect_all_commit_file(commit_info)
        project_type = determine_project_type(all_f, commit_info.repo_name)
    for commit_info in traverse_all_commit():
        commit_key = get_commit_key(commit_info)
        file_language_mapping.setdefault(commit_key,{})
        all_f = collect_all_commit_file(commit_info)
        project_type = determine_project_type(all_f, commit_info.repo_name)
        if project_type is None:
            cannot_determine += 1
            cannot_determine_repo_name.append(commit_info.repo_name)
            project_type = 'CPP'
        this_commit_parse_success = True
        print(commit_info)

        for f in all_f:
            print(f)
            total_cnt += 1
            ext = f.split('.')[-1]
            file_ext_cnt.setdefault(ext, 0)
            file_ext_cnt[ext] += 1
            file_type = 'C'
            if ext == 'c':
                if not ParserC.extract_function(f):
                    this_commit_parse_success = False
                    fail_file.append(f)
            elif ext in ['cpp', 'cc', 'hh', 'hxx', 'cxx', 'hpp']:
                if not ParserCPP.extract_function(f):
                    this_commit_parse_success = False
                    fail_file.append(f)
                else:
                    file_type = 'CPP'
            elif ext == 'h':
                if project_type == 'C':
                    if ParserC.extract_function(f):
                        file_type = 'C'
                    else:
                        if ParserCPP.extract_function(f):
                            file_type = 'CPP'
                        else:
                            fail_file.append(f)
                elif project_type == 'CPP':
                    if ParserCPP.extract_function(f):
                        file_type = 'CPP'
                    else:
                        if ParserC.extract_function(f):
                            file_type= 'C'
                        else:
                            fail_file.append(f)

                # guess_lang = guess.language_name(open(f_path, mode='r').read())
                # print(guess_lang)
            assert file_type in ['C','CPP']

            file_language_mapping[commit_key][f] = file_type

        if this_commit_parse_success:
            commit_all_file_parse_success_cnt += 1
        else:  # remove generated output directory
            delete_dir = f"processed/{'/'.join(str(Path(all_f[0]).parent.parent).split('/')[1:])}"
            if Path(delete_dir).exists():
                shutil.rmtree(delete_dir)


    print(f'Total Commit:[{total_commit_cnt}] , Parse success:[{commit_all_file_parse_success_cnt}]')
    print(cannot_determine)
    print(f'total:{total_cnt} failed:{len(fail_file)}')
    print(cannot_determine_repo_name)

    with open('processed/file_language_mapping.json',mode='w') as f:
        json.dump(file_language_mapping,f)

    with open('error_signature.txt', mode='w') as f:
        f.write('\n'.join(fail_file))


clean_cache = False
if clean_cache:
    shutil.rmtree('processed/download')

if not Path('processed/download').exists():
    extract_all_commit_function()

# Total Commit:[8334] , Parse success:[8187]

# {'cpp': 2490, 'c': 20796, 'h': 4932, 'cc': 2952, 'hh': 36, 'hxx': 4, 'cxx': 134, 'hpp': 48}

# C language
# total:20796 failed:1462
# total:20796 failed:1320   
# total:20796 failed:427
# total:20796 failed:270
# total:20796 failed:248

# CPP language
# total:5664 failed:305
# total:5664 failed:75
# total:5664 failed:53


# diff function

def traverse_commit_file_pair(commit_info:CommitInfo):
    save_dir = cache_commit_file_dir(commit_info.repo_name, commit_info.commit_hash, commit_info.commit_hash)
    parent_save_dir = cache_commit_file_dir(commit_info.repo_name, commit_info.commit_hash, commit_info.parent_commit_hash)
    for f in commit_info.diff_file_paths:
        trunc_name = trunc_commit_file_name(f)
        file = save_dir/trunc_name
        p_file = parent_save_dir/trunc_name
        yield file,p_file,f

def cache2processed_dir(path:Path):
    return Path(f'processed/{"/".join(str(path).split("/")[1:])}')



def find_diff_function(a_file:Path,b_file:Path) -> (list,list[ExtractedFunction]):
    a_funcs = read_json_file_to_dataclass_list(a_file,ExtractedFunction)
    b_funcs = read_json_file_to_dataclass_list(b_file,ExtractedFunction)
    a_mapping,b_mapping = {} , {}
    def name2mapping(funcs:list[ExtractedFunction],mapping):
        for f in funcs:
            if f.func_name is not None:
                mapping[f'{f.func_name}$$${f.parameter_list_signature}'] = f
                # mapping[f'{f.func_name}'] = f

    name2mapping(a_funcs,a_mapping)
    name2mapping(b_funcs,b_mapping)
    func_name_keys = list(set(a_mapping.keys()) & set(b_mapping.keys()))
    diff_funcs = []
    non_diff_funcs = []
    for func_name in func_name_keys:
        a_func : ExtractedFunction = a_mapping[func_name]
        b_func : ExtractedFunction= b_mapping[func_name]
        if a_func.func != b_func.func:
            # function has difference
            diff_funcs.append((a_func,b_func))
        else:
            # function no difference
            non_diff_funcs.append(a_func)
    return diff_funcs , non_diff_funcs

def diff_func(before:list[str],after:list[str]):
    before = ''.join(before)
    after = ''.join(after)
    added_lines ,deleted_lines  = [] ,[]
    raw_diff = []
    for idx,line in enumerate(difflib.unified_diff(
        before.split('\n'), after.split('\n') , 'func_before' , 'func_after' ,lineterm=''
    )):
        raw_diff.append(line)
        if idx < 2:
            continue
        if line[0] == '-':
            deleted_lines.append(line[1:])
        elif line[0] == '+':
            added_lines.append(line[1:])
    parse_diff_dict =  {  'deleted_lines':deleted_lines , 'added_lines' : added_lines }
    raw_diff = '\n'.join(raw_diff)
    # print(raw_diff)
    # print(parse_diff_dict)
    # print('\n')
    # print('*'*60)

    return raw_diff,parse_diff_dict


file_language_mapping = read_json_file('processed/file_language_mapping.json')
def get_file_language(commit_key:str,file_path):
    return file_language_mapping[commit_key][file_path]
vulnerable_func_cnt = 0
non_vulnerable_func_cnt = 0
cve_commit_with_diff_infos = []
record_vul_funcs = {  }
code_abstracter = CodeAbstracter(string_number_literal_need_abstract=True,add_prefix_symbol=True)

abstract_error_commit = set()

for cve_row in cve_commit_infos:
    new_commits = []
    for commit in cve_row['commits']:
        commit_info : CommitInfo = from_dict_to_dataclass(CommitInfo, commit)
        repo_name = commit_info.repo_name
        commit_key = get_commit_key(commit_info)

        print('='*40)
        print(commit_key)

        record_vul_funcs.setdefault(repo_name,set())
        this_commit_diff = []
        for p in traverse_commit_file_pair(commit_info):
            programming_language = get_file_language(commit_key,str(p[0]))
            p = cache2processed_dir(p[0]) , cache2processed_dir(p[1]),p[2]
            non_diff_funcs:list[ExtractedFunction]
            diff_funcs,non_diff_funcs = find_diff_function(p[0],p[1])
            save_path = Path(f'processed/diff/{commit_info.repo_name}/{commit_info.commit_hash}/{p[0].name}')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            data = list(map(lambda x: [dataclasses.asdict(x[0]),dataclasses.asdict(x[1])], diff_funcs))
            vulnerable_func_cnt += len(data)
            non_vulnerable_func_cnt += len(non_diff_funcs)
            # if len(data) != 0:
            #     with save_path.open(mode='w') as f:
            #         json.dump(data, f, indent=4)
            diff_func_pairs = []
            for f in diff_funcs:
                raw_diff,parse_diff_dict = diff_func(f[1].func,f[0].func)
                if len(parse_diff_dict['added_lines']) >= 1000 or len(parse_diff_dict['deleted_lines']) >= 1000\
                        or len(f[1].func) >= 5000 or len(f[0].func) >= 5000:
                    # the file may be generated by some tool ,filter noisy data
                    # e.g. BelledonneCommunications/belle-sip 116e3eb48fe43ea63eb9f3c4b4b30c48d58d6ff0
                    continue
                func_before = ''.join(f[1].func)
                func_after = ''.join(f[0].func)

                # code abstracting
                try:
                    func_before_abstract, func_before_abstract_symbol_map = code_abstracter.abstract_code(func_before,
                                                                                                          programming_language)
                    func_after_abstract, func_after_abstract_symbol_map = code_abstracter.abstract_code(func_after,
                                                                                                        programming_language)
                    func_before_abstract, func_after_abstract = ''.join(func_before_abstract), ''.join(
                        func_after_abstract)
                except AssertionError:
                    abstract_error_commit.add(commit_key)
                    continue

                func_pair = DiffFuncPair(f[1].func_name,f[1].parameter_list_signature,f[1].parameter_list,f[1].return_type,
                             func_before,
                             func_before_abstract,
                             func_before_abstract_symbol_map,
                             func_after,
                             func_after_abstract,
                             func_after_abstract_symbol_map,
                             raw_diff,parse_diff_dict)
                diff_func_pairs.append(func_pair)

                vul_func_key = f"{f[1].func_name}$${f[1].parameter_list_signature}"
                record_vul_funcs[repo_name].add(vul_func_key)    # record this function is vulnerable at least once
            if len(diff_func_pairs) == 0:
                continue

            non_vul_funcs = []
            for f in non_diff_funcs:
                # filter some function more than 5000 lines
                if len(f.func) < 5000:
                    non_vul_funcs.append(NonVulFunction(f.func_name,f.parameter_list_signature,f.parameter_list,f.return_type,''.join(f.func)))
            diff_file = DiffFile(p[2],programming_language,diff_func_pairs,non_vul_funcs)
            this_commit_diff.append(diff_file)

        if len(this_commit_diff) == 0 :
            continue
        new_commit = CommitInfoWithDiffInfo(commit_info.repo_name,commit_info.commit_hash,commit_info.parent_commit_hash,
                               commit_info.diff_file_paths,commit_info.commit_msg,commit_info.git_url,this_commit_diff)
        new_commits.append(dataclasses.asdict(new_commit))

    if len(new_commits) == 0:
        continue
    cve_row['commits'] = new_commits
    cve_commit_with_diff_infos.append(cve_row)


print(f'vulnerable function:{vulnerable_func_cnt}\t raw non-vulnerable function:{non_vulnerable_func_cnt}')

with open('error_abstract.txt',mode='w') as f:
    for c in abstract_error_commit:
        f.write(f'{c}\n')
    f.write(f'Total:{len(abstract_error_commit)}')


def filter_non_vul_funcs(cve_commit_with_diff_infos):
    non_vulnerable_func_cnt = 0

    for cve_row in cve_commit_with_diff_infos:
        for commit in cve_row['commits']:
            repo_name = commit['repo_name']

            for f in commit['diff_files']:
                new_non_vul_funcs = []
                for func in f['non_vul_funcs']:
                    func_key = f"{func['func_name']}$${func['parameter_list_signature']}"
                    if not(func_key in record_vul_funcs[repo_name]):
                        new_non_vul_funcs.append(func)
                f['non_vul_funcs'] = new_non_vul_funcs
                non_vulnerable_func_cnt += len(f['non_vul_funcs'])
    print(f'after filter, non-vulnerable function:{non_vulnerable_func_cnt}')
    return cve_commit_with_diff_infos

cve_commit_with_diff_infos = filter_non_vul_funcs(cve_commit_with_diff_infos)




print('cve_commit_with_diff_infos:',len(cve_commit_with_diff_infos))
json.dump(cve_commit_with_diff_infos, open('result/cve_commit_with_diff_infos.json', mode='w'))




























