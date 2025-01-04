from enum import Enum
from transformers import PreTrainedTokenizer
import pandas as pd
from typing import Union, Callable

train_set = pd.read_json('../vul4c_dataset/train.json')
test_set = pd.read_json('../vul4c_dataset/test.json')
filter_train_set = train_set[
    (train_set['func'].str.split('\n').str.len() < 30) & (train_set['func'].str.split('\n').str.len() > 10)]
vul_train_set = filter_train_set[filter_train_set['vul'] == 1]
non_vul_train_set = filter_train_set[filter_train_set['vul'] == 0]

print(f'train set {len(filter_train_set)}')
fix_select_example = filter_train_set[filter_train_set['id'] in [11299,88703,17002,228194,92053,123609]]
fix_select_example = fix_select_example.to_dict('records')

fix_select_example_len = None


class ExampleSelectStrategy(Enum):
    Fix = "fix"
    Random = "random"
    Diversity = "diversity"
    CosineSimilar = "cosine"
    SameRepo = "same_repo"


train_closest_to_center_idxs = pd.read_pickle('../vul4c_dataset/train_closest_to_center_idxs.pkl').iloc[:,
                               0].values.tolist()

train_closest_to_center_examples = train_set.iloc[train_closest_to_center_idxs].to_dict('records')
train_closest_to_center_examples_len = None

test_similar_to_train = pd.read_pickle('../vul4c_dataset/test_similar_to_train.pkl').to_dict('records')


def select_examples(test_id: int, strategy: ExampleSelectStrategy, tokenize: Callable):
    if strategy == ExampleSelectStrategy.Fix:
        global fix_select_example_len
        if fix_select_example_len is None:
            fix_select_example_len = []
            for e in fix_select_example:
                fix_select_example_len.append(len(tokenize(e['func'])))
        return fix_select_example, fix_select_example_len
    elif strategy == ExampleSelectStrategy.Random:
        examples =  pd.concat(
            [filter_train_set[filter_train_set['vul'] == 1].sample(n=3, random_state=test_id),
             filter_train_set[filter_train_set['vul'] == 0].sample(n=3, random_state=test_id),
             ]
        ).sample(frac=1, random_state=4096).to_dict('records')
        example_lens = []
        for e in examples:
            example_lens.append(len(tokenize(e['func'])))
        return examples, example_lens

    elif strategy == ExampleSelectStrategy.Diversity:
        global train_closest_to_center_examples_len
        if train_closest_to_center_examples_len is None:
            train_closest_to_center_examples_len = []
            for e in train_closest_to_center_examples:
                train_closest_to_center_examples_len.append(len(tokenize(e['func'])))
        return train_closest_to_center_examples, train_closest_to_center_examples_len
    elif strategy == ExampleSelectStrategy.CosineSimilar:
        test_similar_items = test_similar_to_train[test_id]
        test_similar_items_idxs = test_similar_items['idxs'][:6]  # select top 6
        test_similar_items_scores = test_similar_items['similar_scores'][:6]  # select top 6
        examples = train_set.iloc[test_similar_items_idxs].to_dict('records')
        example_lens = []
        for e in examples:
            example_lens.append(len(tokenize(e['func'])))
        return examples, example_lens
    elif strategy == ExampleSelectStrategy.SameRepo:
        e = test_set[test_set["id"] == test_id].iloc[0]
        repo_name = e['repo_name']
        # print(f'same repo find {repo_name}')
        e_id = e['id']
        vul = vul_train_set[(vul_train_set['repo_name'] == repo_name) & (vul_train_set['id'] != e_id)]
        non_vul = non_vul_train_set[
            (non_vul_train_set['repo_name'] == repo_name) & (non_vul_train_set['id'] != e_id)]
        vul = vul.sample(n=3 if vul.shape[0] > 3 else vul.shape[0], random_state=2023)
        non_vul = non_vul.sample(n=3 if non_vul.shape[0] > 3 else non_vul.shape[0], random_state=2023)
        examples = pd.concat([vul, non_vul]).sample(frac=1, random_state=4096).to_dict('records')
        example_lens = []
        for e in examples:
            example_lens.append(len(tokenize(e['func'])))
        return examples, example_lens


