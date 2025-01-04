"""
    1. dump all vulnerable function and non-vul function to `processed/graph`
    2. run joern extract AST,CFG,...
"""
import os.path

from utils import read_json_file
from pathlib import Path


def dump_func(path: Path, func: str):
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        assert False, f'[ERROR] this function overwrite other functions {path} \n{func}'
    with path.open(mode='w') as f:
        f.write(func)

set_lang = set()
def dump_func_to_graph_folder():
    cve_commit_infos = read_json_file('result/cve_commit_with_diff_infos.json')
    save_root_path = Path('./processed/graph')
    function_counter = 0
    if save_root_path.exists():
        print(f'{save_root_path} folder has been created! skip dump function')
        return

    for cve_row in cve_commit_infos:
        for commit in cve_row['commits']:
            new_repo_name = commit['repo_name'].replace('/', '-')
            commit_hash = commit['commit_hash']
            print(commit.keys())
            print(new_repo_name)
            for diff_file in commit['diff_files']:
                print(diff_file.keys())
                print(diff_file['file_path'])
                pl_lang : str = diff_file['lang'].lower()
                set_lang.add(pl_lang)
                new_file_path = diff_file['file_path'].replace('/', '-').replace('.', '-')
                save_file_path = save_root_path / new_repo_name / commit_hash / new_file_path

                # vulnerable function
                # in CPP same function name may appear several times because of function overload
                # we use unique identifier to identity function
                for id, vul_func_pair in enumerate(diff_file['diff_funcs']):
                    id = str(id)
                    func_save_dir_path = save_file_path / 'vul'
                    dump_func(func_save_dir_path / 'before' / id / f"{id}.{pl_lang}", vul_func_pair['func_before'])
                    dump_func(func_save_dir_path / 'after' / id / f"{id}.{pl_lang}", vul_func_pair['func_after'])
                    function_counter += 2

                for id, non_vul_func in enumerate(diff_file['non_vul_funcs']):
                    id = str(id)
                    func_save_dir_path = save_file_path / 'non-vul'
                    dump_func(func_save_dir_path / id / f"{id}.{pl_lang}", non_vul_func['func'])
                    function_counter += 1
    print(f'dump function {function_counter}')

dump_func_to_graph_folder()

print(set_lang)