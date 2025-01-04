import json
import os.path
import sys

from utils import read_json_file
from pathlib import Path
from tqdm import tqdm
import shutil

cve_commit_infos = read_json_file('result/cve_commit_with_diff_infos.json')
save_root_path = Path('./processed/graph')
function_counter = 0
if not save_root_path.exists():
    print(f'{save_root_path} folder does not exist, run extract_graph_dump.py first')
    sys.exit(0)

cnt = []
vul_cnt = 0


def find_label_in_nodes(nodes: list[dict], label: str):
    res = []
    for n in nodes:
        if n['_label'] == label:
            res.append(n)
    return res


def find_method_in_nodes(nodes: list[dict], method_name=None):
    methods = find_label_in_nodes(nodes, 'METHOD')
    if method_name is None:
        return methods
    new_method = []
    for m in methods:
        if m['name'] == method_name:
            new_method.append(m)
    return new_method


def find_unknown_in_nodes(nodes: list[dict]):
    return find_label_in_nodes(nodes, 'UNKNOWN')


def get_node_in_out_map(edges: list[dict]):
    in_nodes, out_nodes = {}, {}  #
    for e in edges:
        # in node ----> out node
        in_node = e['inNode']
        out_node = e['outNode']

        out_nodes.setdefault(in_node, [])
        in_nodes.setdefault(out_node, [])
        if out_node not in out_nodes[in_node]:
            # maybe duplicate , because different types of edge
            out_nodes[in_node].append(out_node)
        if in_node not in in_nodes[out_node]:
            in_nodes[out_node].append(in_node)
    return in_nodes, out_nodes


def check_func_graph_complete(path: Path, func_name) -> bool:
    global cnt, vul_cnt

    if not (path / f'{func_name}.c').exists() and not (path / f'{func_name}.cpp').exists():
        assert False, "[ERROR] function doesn't exists"

    node_json = path / f"{func_name}.nodes.json"
    edge_json = path / f"{func_name}.edges.json"
    if not node_json.exists() or not edge_json.exists():
        assert False, f"[ERROR] {path} node or edge not created completely"
    if node_json.stat().st_size == 0 or edge_json.stat().st_size == 0:
        assert False, f"[ERROR] {path} node or edge is empty"

    if node_json.stat().st_size < 5000 or edge_json.stat().st_size < 5000:
        pass
    node_json = read_json_file(str(node_json))
    edge_json = read_json_file(str(edge_json))
    in_nodes, out_nodes = get_node_in_out_map(edge_json)
    method_nodes = find_method_in_nodes(node_json, '<global>')  # <global> METHOD
    # print(path)
    # print(method_nodes)
    # print(out_nodes)
    parse_error_methods = []
    for m in method_nodes:
        # find method block
        m_id = m['id']
        block_nodes = []
        # print(in_nodes[m_id])
        for block_node in in_nodes[m_id]:
            block_node = node_json[block_node - 1]  # get the node
            if block_node['_label'] == 'BLOCK':  # really block node
                block_nodes.append(block_node)
        # assert len(block_nodes) == 1 , print(block_nodes,path)
        block_node = block_nodes[0]
        # find block node next node
        # if next node is UNKNOWN node, we think the function parse errors
        # print('*'*10)
        unknown_found = False
        for next_node in in_nodes[block_node['id']]:
            next_node = node_json[next_node - 1]
            if next_node['_label'] == 'UNKNOWN':
                unknown_found = True

        if unknown_found:
            parse_error_methods.append(m)


    if len(parse_error_methods) > 0:
        #
        cnt.append(path)
        return False

    return True
    # print(namespace_block)
    # if len(find_unknown_in_nodes(node_json)) != 0:
    #     cnt.append(path)
    #     if 'before' in str(path) or 'after' in str(path):
    #         vul_cnt+=1

    # node_json = read_json_file(str(node_json))
    # edge_json = read_json_file(str(edge_json))
    # method_nodes = find_method_in_nodes(node_json)
    # print(len(method_nodes))


def copy_graph_with_dir(path: Path, old_name: str, new_name: str):
    # path = processed/graph/torvalds-linux/5ecfbae093f0c37311e89b29bfc0c9d586eace87/fs-exec-c/non-vul/18
    # old_name = 18  new_name = 15
    #
    # copy to
    # result/graph/torvalds-linux/5ecfbae093f0c37311e89b29bfc0c9d586eace87/fs-exec-c/non-vul/15
    old_path = str(path).split('/')
    new_path = ['result'] + old_path[1:-1] + [new_name]
    new_path = Path('/'.join(new_path))
    shutil.copytree(path, new_path)  # copy directory
    # rename file
    for suffix in ['c', 'cpp']:
        if (new_path / f'{old_name}.{suffix}').exists():
            os.rename((new_path / f'{old_name}.{suffix}'), (new_path / f'{new_name}.{suffix}'))

    new_edge_path = new_path / f'{new_name}.edges.json'
    new_node_path = new_path / f'{new_name}.nodes.json'
    os.rename(new_path / f'{old_name}.edges.json', new_edge_path)
    os.rename(new_path / f'{old_name}.nodes.json', new_node_path)
    return str(Path(*new_node_path.parts[1:])), str(Path(*new_edge_path.parts[1:]))



raw_vul_cnt, raw_non_vul_cnt = 0, 0
filtered_vul_cnt, filtered_non_vul_cnt = 0, 0

new_cve_commit_infos = []
global_graph = []  # save all graph in this list
global_graph_id = 0


Path("result/graph").mkdir(exist_ok=True,parents=True)
def copy_graph(path: Path, old_name: str):
    global global_graph_id
    old_id = global_graph_id
    shutil.copy(path / f'{old_name}.edges.json', f'result/graph/{old_id}.edges.json')
    shutil.copy(path / f'{old_name}.nodes.json', f'result/graph/{old_id}.nodes.json')
    global_graph_id += 1
    return old_id

def read_graph(path: Path, old_name: str):
    global global_graph
    edges = read_json_file(str(path / f'{old_name}.edges.json'))
    nodes = read_json_file(str(path / f'{old_name}.nodes.json'))
    old_id = len(global_graph)
    global_graph.append({'nodes': nodes, 'edges': edges})
    return old_id


for cve_row in tqdm(cve_commit_infos):
    new_commits = []
    for commit in cve_row['commits']:
        new_repo_name = commit['repo_name'].replace('/', '-')
        commit_hash = commit['commit_hash']

        new_diff_files = []

        for diff_file in commit['diff_files']:
            new_file_path = diff_file['file_path'].replace('/', '-').replace('.', '-')
            save_file_path = save_root_path / new_repo_name / commit_hash / new_file_path
            new_id = 0
            new_diff_func = []
            for id, vul_func_pair in enumerate(diff_file['diff_funcs']):
                id = str(id)
                func_save_dir_path = save_file_path / 'vul'
                raw_vul_cnt += 1
                if check_func_graph_complete(func_save_dir_path / 'before' / id, id) and \
                        check_func_graph_complete(func_save_dir_path / 'after' / id, id):
                    filtered_vul_cnt += 1
                    # vul_func_pair['func_before_node'] , vul_func_pair['func_before_edge']  = \
                    #     copy_graph_with_dir(func_save_dir_path / 'before' / id , id, str(new_id))
                    # vul_func_pair['func_after_node'] , vul_func_pair['func_after_edge']  = \
                    #     copy_graph_with_dir(func_save_dir_path / 'after' / id , id, str(new_id))
                    vul_func_pair['func_before_graph_idx'] = copy_graph(func_save_dir_path / 'before' / id, id)
                    vul_func_pair['func_after_graph_idx'] = copy_graph(func_save_dir_path / 'after' / id, id)
                    new_diff_func.append(vul_func_pair)
                    new_id += 1

            new_id = 0
            new_non_vul_funcs = []
            for id, non_vul_func in enumerate(diff_file['non_vul_funcs']):
                id = str(id)
                func_save_dir_path = save_file_path / 'non-vul'
                raw_non_vul_cnt += 1
                if check_func_graph_complete(func_save_dir_path / id, id):
                    filtered_non_vul_cnt += 1
                    # non_vul_func['func_node'], non_vul_func['func_edge'] = \
                    #     copy_graph_with_dir(func_save_dir_path /  id, id, str(new_id))
                    non_vul_func['func_graph_idx'] = copy_graph(func_save_dir_path / id, id)
                    new_non_vul_funcs.append(non_vul_func)
                    new_id += 1

            if len(new_diff_func) > 0:  # prefer consider vulnerable function
                diff_file['diff_funcs'] = new_diff_func
                diff_file['non_vul_funcs'] = new_non_vul_funcs
                new_diff_files.append(diff_file)

        if len(new_diff_files) > 0:
            commit['diff_files'] = new_diff_files
            new_commits.append(commit)

    if len(new_commits) > 0:
        cve_row['commits'] = new_commits
        new_cve_commit_infos.append(cve_row)

print(cnt)

with open('error_joern.txt', mode='w') as f:
    for c in cnt:
        f.write(f'{str(c)}\n')



# print(len(cnt))
# print(vul_cnt)

json.dump(new_cve_commit_infos, open('result/cve_commit_with_graph_infos.json', mode='w'))
print(f'[Raw] vul:{raw_vul_cnt}  non-vul:{raw_non_vul_cnt}')
print(f'[Graph filtered] vul:{filtered_vul_cnt}  non-vul:{filtered_non_vul_cnt}')
print()
