from functools import cmp_to_key
from pathlib import Path
import json
from visualize import visualize_code_tokens, visualize_code_lines, VisualizeInfo
from treesitter_parse import parse_source_code_statement_type, filter_statement_on_given_lines , StatementAttention
import pickle
from tqdm import tqdm
import torch
from functools import reduce

project_dir = Path(__file__).parent.parent.parent
dataset_dir = project_dir / "vul4c_dataset"
test_set = json.load((dataset_dir / "test.json").open(mode='r'))

save_base_dir = Path(__file__).parent / "result"


def sort_line_attention_return_sort_indices(token_level_results, top_k = 10):
    # [bs_size, ]
    result = []
    for item in token_level_results:
        line_attentions = []
        for line in item:
            cur_line_attention = 0
            for token,attention in line:
                cur_line_attention += attention
            line_attentions.append(cur_line_attention)

        _ , indices = torch.topk(torch.tensor(line_attentions), k= min(top_k ,len(line_attentions)))
        indices = indices.tolist()
        result.append(indices)
    return result


class StatementTypeLogger:

    def __init__(self):
        self.types_counter = {}

    def print(self):
        print('=========== StatementType ===========')
        sorted_types = list(sorted(self.types_counter.items(), key=lambda x: x[1],reverse=True))
        max_len = max(len(i[0]) for i in sorted_types)
        for type,cnt in sorted_types:
            print(f'{type:>{max_len}s}: {cnt}')
        print(sorted_types)

    def logger(self,statements:list[StatementAttention]):
        for s in statements:
            self.types_counter.setdefault(s.type,0)
            self.types_counter[s.type] += 1



def get_ground_truth(id):
    return test_set[id]


def get_save_dir(model_name):
    save_dir = save_base_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def linevul():
    model_name = 'LineVul'
    save_dir = get_save_dir(model_name)
    data = json.load(open('../../sequence/LineVul/storage/test_interpret_7_18.json'))
    statement_logger = StatementTypeLogger()

    for item in data:
        id, pred, line_scores, line_token_level_scores = item['id'], item['pred'], item['line_scores'], item[
            'line_token_level_scores']
        # print(line_scores)
        # print(line_token_level_scores)
        ground_truth = get_ground_truth(id)
        # print(ground_truth.keys())
        if pred == ground_truth['vul'] and pred:
            if id != 22854:
                continue
            print(id)
            print(ground_truth['git_url'])
            print(line_scores)
            # add space
            new_token_level_scores = []
            compose_func = ''
            for line in line_token_level_scores:
                new_line = []
                for idx, (token, score) in enumerate(line):
                    if idx > 0:
                        new_line.append((" " + token, score))
                    else:
                        new_line.append((token, score))
                compose_line = reduce(lambda x,y: f"{x}{y[0]}", new_line , "")
                compose_func += compose_line + '\n'

                new_token_level_scores.append(new_line)
            compose_func = compose_func[:-1] # remove last line break
            # print(new_token_level_scores)
            # print(compose_func)
            statements = parse_source_code_statement_type(compose_func , new_token_level_scores  )
            top10_line_number = sort_line_attention_return_sort_indices([new_token_level_scores])[0]
            in_top10_line_statements = filter_statement_on_given_lines(top10_line_number,statements)
            statement_logger.logger(in_top10_line_statements)


            # print(sort_line_attention_return_sort_indices(new_token_level_scores)[0])
            # print(statements)
            visualize_code_lines(id, save_dir, line_scores,
                                  VisualizeInfo(model_name, pred, ground_truth['vul'], ground_truth['cwe_list'][0],
                                                ground_truth['cve'], ground_truth['git_url'], ground_truth['func'])
                                  )

    statement_logger.print()



def svuld():
    model_name = 'SVulD'
    save_dir = get_save_dir(model_name)
    data = json.load(open('../../sequence/SVulD/storage/test_interpret_7_24.json'))
    statement_logger = StatementTypeLogger()

    for item in data:
        id, pred, line_scores, line_token_level_scores = item['id'], item['pred'], item['line_scores'], item[
            'line_token_level_scores']
        print(line_scores)
        print(line_token_level_scores)
        ground_truth = get_ground_truth(id)
        print(ground_truth.keys())
        if pred == ground_truth['vul'] and pred:
            print(id)
            print(ground_truth['git_url'])
            print(line_token_level_scores)
            # add space
            new_token_level_scores = []
            compose_func = ''
            for line in line_token_level_scores:
                new_line = []
                for idx, (token, score) in enumerate(line):
                    if idx > 0:
                        new_line.append((" " + token, score))
                    else:
                        new_line.append((token, score))
                compose_line = reduce(lambda x,y: f"{x}{y[0]}", new_line , "")
                compose_func += compose_line + '\n'

                new_token_level_scores.append(new_line)
            compose_func = compose_func[:-1] # remove last line break
            # print(new_token_level_scores)
            # print(compose_func)
            statements = parse_source_code_statement_type(compose_func , new_token_level_scores  )
            top10_line_number = sort_line_attention_return_sort_indices([new_token_level_scores])[0]
            in_top10_line_statements = filter_statement_on_given_lines(top10_line_number,statements)
            statement_logger.logger(in_top10_line_statements)


            # print(sort_line_attention_return_sort_indices(new_token_level_scores)[0])
            # print(statements)
            visualize_code_tokens(id, save_dir, new_token_level_scores,
                                  VisualizeInfo(model_name, pred, ground_truth['vul'], ground_truth['cwe_list'][0],
                                                ground_truth['cve'], ground_truth['git_url'], ground_truth['func'])
                                  )

    statement_logger.print()

def graph_node_to_token_level(model_name, ids, nodes_with_attentions):
    line_level_results = []
    token_level_results = []
    line_level_results_cache_path = Path(f'./result/{model_name}/line_level_results.pkl')
    token_level_results_cache_path = Path(f'./result/{model_name}/token_level_results.pkl')
    if line_level_results_cache_path.exists() and token_level_results_cache_path.exists():
        return pickle.load(token_level_results_cache_path.open(mode='rb')), pickle.load(
            line_level_results_cache_path.open('rb'))

    for idx, node_with_attention in tqdm(zip(ids, nodes_with_attentions)):
        ground_truth = get_ground_truth(idx)
        raw_func = ground_truth['func']
        token_level_attention_list: list[list] = [[(l, 0)] for l in raw_func.split('\n')]

        # split multi line nodes and filter out empty nodes
        new_node_with_attention = []


        node_code: str
        for node_line, node_code, node_attention in node_with_attention:
            if len(node_code) == 0:
                continue
            for code_line in node_code.split('\n'):
                # print(f'split {code_line}')
                if len(code_line) == 0:
                    continue
                new_node_with_attention.append((node_line, code_line, node_attention))
                node_line += 1
        node_with_attention = new_node_with_attention

        def code_compare(item1, item2):
            if item1[0] < item2[0]:
                return -1
            elif item1[0] == item2[0]:
                if len(item1[1]) > len(item2[1]):
                    return -1
            return 1

        # print(node_with_attention)
        # smaller line numbers come first, longer code lengths come first
        node_with_attention = list(sorted(node_with_attention, key=cmp_to_key(code_compare)))
        # print(node_with_attention)

        # print('----------- start ------------')
        # print(token_level_attention_list)

        # token level attention calculate
        for node_line, node_code, node_attention in node_with_attention:
            # print('------------------------------')
            # candidate
            tokens = token_level_attention_list[node_line]
            candidate_idxs = []

            start_id = 0
            new_tokens = []
            token: str
            for token, attention_score in tokens:
                # print(f'token:{token} score:{attention_score}  find node:{node_code}')
                token_find_index = token.find(node_code)
                if token_find_index == -1:
                    new_tokens.append((token, attention_score))
                    continue
                # split
                if len(token) == len(node_code):  # no need split
                    new_tokens.append((token, attention_score + node_attention))
                elif token_find_index == 0:
                    after = token[len(node_code):]
                    new_tokens.append((node_code, attention_score + node_attention))
                    new_tokens.append((after, attention_score))
                elif token_find_index + len(node_code) == len(token):
                    before = token[:token_find_index]
                    new_tokens.append((before, attention_score))
                    new_tokens.append((node_code, attention_score + node_attention))
                else:
                    # split 3 segments      [pre, node_code, after]
                    before = token[:token_find_index]
                    after = token[token_find_index + len(node_code):]
                    new_tokens.append((before, attention_score))
                    new_tokens.append((node_code, attention_score + node_attention))
                    new_tokens.append((after, attention_score))

            token_level_attention_list[node_line] = new_tokens

        line_level_attention_list = []

        for tokens in token_level_attention_list:
            code_line = ""
            score = 0.0
            for token, attention_score in tokens:
                code_line += token
                score += attention_score
            line_level_attention_list.append((code_line, score))

        line_level_results.append(line_level_attention_list)
        token_level_results.append(token_level_attention_list)

    pickle.dump(token_level_results, token_level_results_cache_path.open(mode='wb'))
    pickle.dump(line_level_results, line_level_results_cache_path.open(mode='wb'))
    return token_level_results, line_level_results




def devign():
    model_name = 'Devign'
    nodes_with_attentions = pickle.load(
        open('../../Graph/storage/results/devign/vul4c_dataset/test_interpret_7_24.pkl', mode='rb'))
    print(nodes_with_attentions)
    devign_test_result = json.load(open('../../Graph/storage/results/devign/vul4c_dataset/test.json', mode='r'))
    print(len(devign_test_result))
    assert len(nodes_with_attentions) == len(devign_test_result)

    ids = [item['id'] for item in devign_test_result]

    token_level_results, line_level_results = graph_node_to_token_level(model_name,ids, nodes_with_attentions)
    top_attention_line_indices = sort_line_attention_return_sort_indices(token_level_results)
    statement_logger = StatementTypeLogger()

    save_dir = get_save_dir(model_name)
    for idx in range(len(nodes_with_attentions)):
        pred = devign_test_result[idx]['pred']
        real_id = devign_test_result[idx]['id']
        ground_truth = get_ground_truth(real_id)

        if pred == ground_truth['vul'] and pred:
            print(real_id)
            visualize_code_tokens(real_id, save_dir, token_level_results[idx],
                                  VisualizeInfo(model_name, pred, ground_truth['vul'], ground_truth['cwe_list'][0],
                                                ground_truth['cve'], ground_truth['git_url'], ground_truth['func']))

            statements = parse_source_code_statement_type(ground_truth['func'] , token_level_results[idx]  )
            top10_line_number = top_attention_line_indices[idx]
            in_top10_line_statements = filter_statement_on_given_lines(top10_line_number,statements)
            statement_logger.logger(in_top10_line_statements)
    statement_logger.print()

def ivdetect():
    model_name = 'IVdetect'
    line_with_attentions = pickle.load(
        open('../../Graph/storage/results/ivdetect/vul4c_dataset/test_interpret_merge_7_22.pkl', mode='rb'))
    line_with_attentions = [[(item[1], item[2]) for item in line] for line in line_with_attentions]
    ivdetect_test_result = json.load(open('../../Graph/storage/results/ivdetect/vul4c_dataset/test.json', mode='r'))
    assert len(ivdetect_test_result) == len(line_with_attentions)
    save_dir = get_save_dir(model_name)
    print(line_with_attentions)
    print(len(ivdetect_test_result))

    def transform_to_token_level_results(line_with_attentions) -> list:
        token_level_results = []
        for item in line_with_attentions:
            item_result = []
            for line in item:
                item_result.append([line])
            token_level_results.append(item_result)
        return token_level_results

    token_level_results = transform_to_token_level_results(line_with_attentions)
    top_attention_line_indices = sort_line_attention_return_sort_indices(token_level_results)
    statement_logger = StatementTypeLogger()

    for idx in range(len(line_with_attentions)):
        pred = ivdetect_test_result[idx]['pred']
        real_id = ivdetect_test_result[idx]['id']
        ground_truth = get_ground_truth(real_id)

        if pred == ground_truth['vul'] and pred:
            print(real_id)
            print(ground_truth['git_url'])
            visualize_code_lines(real_id, save_dir, line_with_attentions[idx],
                                 VisualizeInfo(model_name, pred, ground_truth['vul'], ground_truth['cwe_list'][0],
                                               ground_truth['cve'], ground_truth['git_url'], ground_truth['func']))
            statements = parse_source_code_statement_type(ground_truth['func'] , token_level_results[idx]  )
            top10_line_number = top_attention_line_indices[idx]
            in_top10_line_statements = filter_statement_on_given_lines(top10_line_number,statements)
            statement_logger.logger(in_top10_line_statements)

    statement_logger.print()


def reveal():
    model_name = 'Reveal'
    nodes_with_attentions = pickle.load(
        open('../../Graph/storage/results/reveal/vul4c_dataset/test_interpret.pkl', mode='rb'))
    reveal_test_result = json.load(open('../../Graph/storage/results/reveal/vul4c_dataset/test.json', mode='r'))
    assert len(nodes_with_attentions) == len(reveal_test_result) , print(f'nodes_with_attentions[{len(nodes_with_attentions)}] != reveal_test_result[{len(reveal_test_result)}]')

    ids = [item['id'] for item in reveal_test_result]
    token_level_results, line_level_results = graph_node_to_token_level(model_name, ids, nodes_with_attentions)
    # token_level_results :  [ [ [(token,attention) , (token,attention)] , [line2] ,[line3] ... ] , [example2] , ... ]
    top_attention_line_indices = sort_line_attention_return_sort_indices(token_level_results)
    statement_logger = StatementTypeLogger()

    save_dir = get_save_dir(model_name)
    cnt = 0
    for idx in range(len(nodes_with_attentions)):
        pred = reveal_test_result[idx]['pred']
        real_id = reveal_test_result[idx]['id']
        ground_truth = get_ground_truth(real_id)

        if pred == ground_truth['vul'] and pred:
            # print(real_id)
            # print(ground_truth['git_url'])
            visualize_code_tokens(real_id, save_dir, token_level_results[idx],
                                  VisualizeInfo(model_name, pred, ground_truth['vul'], ground_truth['cwe_list'][0],
                                                ground_truth['cve'], ground_truth['git_url'], ground_truth['func']))

            # print(ground_truth['func'])
            statements = parse_source_code_statement_type(ground_truth['func'] , token_level_results[idx]  )
            top10_line_number = top_attention_line_indices[idx]
            in_top10_line_statements = filter_statement_on_given_lines(top10_line_number,statements)
            statement_logger.logger(in_top10_line_statements)
            # print(top10_line_number)
            # for item in in_top10_line_statements:
            #     print(item)
            # if cnt == 1:
            #     assert False
            # cnt += 1

    statement_logger.print()

linevul()
