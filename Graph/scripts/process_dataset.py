import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(str((Path(__file__).parent.parent)))

sys.path.append(str(Path(__file__).parent.parent))

from utils.utils import remove_space_before_newline, remove_comments, remove_empty_lines, remove_space_after_newline, get_dir, \
    processed_dir, cache_dir, dfmp, RunTimeCounter
import utils.glove as glove
import utils.word2vec as word2vec
from utils.git import allfunc, c2dhelper


def cleaned_code(func_code):
    func_code = remove_empty_lines(func_code)
    func_code = remove_comments(func_code)
    func_code = remove_space_before_newline(func_code)
    func_code = remove_space_after_newline(func_code)

    return func_code
    pass


def cleaned_dataset(data, dataset):
    print('Data shape:', data.shape)
    print('Data columns:', data.columns)
    print('Cleaning Code...')
    data['func_before'] = data['func_before'].apply(lambda x: cleaned_code(x))
    data['func_after'] = data['func_after'].apply(lambda x: cleaned_code(x))
    data = data[~data['func_before'].duplicated(keep=False)]
    # remove func_before == func_after
    print('Removing (func_before == func_after) for vulnerable function...')
    data = data[(data['vul'] == 0) | (data['vul'] == 1 & (data['func_before'] != data['func_after']))]
    print('Data shape:', data.shape)

    print('Cleaning Code Done!')

    # Save codediffs
    data = data.reset_index(drop=True).reset_index().rename(columns={'index': '_id'})
    data['dataset'] = dataset
    cols = ["func_before", "func_after", "_id", "dataset"]
    dfmp(data, c2dhelper, columns=cols, ordr=False, cs=300, workers=32)
    # Assign info and save
    data["info"] = dfmp(data, allfunc, cs=500, workers=32)
    data = pd.concat([data, pd.json_normalize(data["info"])], axis=1)
    return data


def mix_patch(df):
    df['mix'] = False
    origin_df = df.copy()
    print(df.shape)
    vul_df = df[df.vul == 1]
    # print(vul_df.shape)
    func_after = vul_df['func_after']
    pat_id = vul_df['_id'] + 190000
    pat = vul_df.copy()
    assert len(pat) == len(func_after)
    pat['func_before'] = func_after
    pat['vul'] = 0
    pat['mix'] = True
    pat['_id'] = pat_id
    df = pd.concat([origin_df, pat])
    # print(df[df.vul == 1])
    # print(df.shape)
    # print(df[df.mix == True]._id -190000)
    # assert False
    return df


def prepare_glove(dataset='devign'):
    # generate GloVe
    glove.generate_glove(dataset, sample=False)

    # Cache IVDetectData
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    # df = pd.read_pickle(cache_path)
    # train_df = df[df.partition == 'train']
    # valid_df = df[df.partition == 'valid']
    # test_df = df[df.partition == 'test']
    # val_ds = BigVulDatasetIVDetect(df=valid_df, partition="valid", dataset=dataset)
    # val_ds.cache_features()
    # train_ds = BigVulDatasetIVDetect(df=train_df, partition="train", dataset=dataset)
    # train_ds.cache_features()
    # test_ds = BigVulDatasetIVDetect(df=test_df, partition="test", dataset=dataset)
    # test_ds.cache_features()


def prepare_w2v(dataset='devign'):
    # generate GloVe
    word2vec.generate_w2v(dataset, sample=False)

    # Cache IVDetectData
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    # df = pd.read_pickle(cache_path)
    # train_df = df[df.partition == 'train']
    # valid_df = df[df.partition == 'valid']
    # test_df = df[df.partition == 'test']
    # val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
    # val_ds.cache_features()
    # train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset=dataset)
    # train_ds.cache_features()
    # test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset)
    # test_ds.cache_features()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    dataset : str = args.dataset
    print('cpu count', os.cpu_count())
    assert dataset.startswith('vul4c')

    time_counter = RunTimeCounter()
    prepare_glove(dataset)
    time_counter.stop('Processing Glove')

    prepare_w2v(dataset)
    time_counter.stop('Processing Word2Vec')
