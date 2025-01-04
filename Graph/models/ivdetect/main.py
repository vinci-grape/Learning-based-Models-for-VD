import pickle

import dgl
import sys

from dgl import DGLError

from dataset import BigVulDatasetIVDetect
from model import IVDetect
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import math
import numpy as np

warnings.filterwarnings('ignore')

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils.utils import debug, get_run_id, processed_dir, get_metrics_logits, cache_dir, set_seed, result_dir, get_dir, \
    get_metrics_new , RunTimeCounter , ModelParameterCounter
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd
from gnn_explainer import GNNExplainer

dataset2id = {
    'reveal': '202303211525_v1',
    'vul4c_dataset': '202306162028_v1',
    'vul4c_rm_comments_dataset': '202306162028_v1',
    'vul4c_insert_comments_dataset': '202306162028_v1',
    'vul4c_unexecuted_code_dataset': '202306162028_v1',
    'vul4c_rename_identifier_dataset': '202306162028_v1',
}

def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        all_prob = []
        all_true = []
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL").long()
            val_logits = model(val_batch, val_ds)
            # val_preds = logits.argmax(dim=1)
            all_prob.extend(val_logits.tolist())
            all_true.extend(val_labels.tolist())
        val_mets = get_metrics_new(all_prob, all_true)
    return val_mets


def test(model, test_dl, test_ds, logger, args):
    # logger.load_best_model()
    path = result_dir() / f"ivdetect/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()
    all_prob = []
    all_true = []
    all_ids = []
    with torch.no_grad():
        for test_batch in tqdm(test_dl, total=len(test_dl)):
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL").long()
            test_logits = model(test_batch, test_ds)
            test_ids = dgl.max_nodes(test_batch, "_SAMPLE")
            all_prob.extend(test_logits.tolist())
            all_true.extend(test_labels.tolist())
            all_ids.extend(test_ids.tolist())
        test_mets = get_metrics_new(all_prob, all_true)

    print(test_mets)

    with open(result_dir() / f"ivdetect/{args.dataset}" / f"test.json", mode='w') as f:
        all_pred = np.argmax(all_prob, axis=-1).tolist()
        ans = []
        for id, p in zip(all_ids, all_pred):
            ans.append({'id': int(id), 'pred': p})
        print(ans)
        json.dump(ans, f, indent=4)

    if logger:
        logger.test(test_mets)
    else:
        print(test_mets)
    return test_mets

def tSNE_embedding(model, test_dl, test_ds, logger, args):
    # logger.load_best_model()
    path = result_dir() / f"ivdetect/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()
    all_prob = []
    all_true = []
    all_ids = []
    max_node_num = 180
    # with torch.no_grad():
    #     g:dgl.DGLGraph
    #     for g in tqdm(mini_dl, total=len(mini_dl)):
    #         max_node_num = max(max_node_num , g.num_nodes())

    # print(max_node_num)
    embedding_shape = max_node_num * 2

    all_tSNE_embedding = []

    with torch.no_grad():
        for test_batch in tqdm(test_dl, total=len(test_dl)):
            test_batch = test_batch.to(args.device)
            test_labels = dgl.max_nodes(test_batch, "_LABEL").long()
            test_ids = dgl.max_nodes(test_batch, "_SAMPLE")
            _ , embedding = model(test_batch, test_ds , output_tSNE_embedding = True , tSNE_embedding_shape = embedding_shape)
            all_tSNE_embedding.extend(
                [(embedding, label) for embedding, label in zip(embedding.tolist(), test_labels.tolist())])


    print(f'tSNE done!! len:{len(all_tSNE_embedding)}')
    with open(result_dir() / f"ivdetect/{args.dataset}"   / "test_tSNE_embedding.pkl" , mode='wb') as f:
        pickle.dump(all_tSNE_embedding , f)


def interpret(model, mini_dl, mini_ds, ground_truth_df, split_idx,interpret_total_split , args):
    # logger.load_best_model()
    path = result_dir() / f"ivdetect/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()
    explainer = GNNExplainer(model, num_hops=1)
    testid_2_idxid = { }
    for idx,g in enumerate(ground_truth_df):
        testid_2_idxid[g['id']] = idx
    # print(testid_2_idxid)
    def get_ground_truth(id:int):
        return ground_truth_df[testid_2_idxid[id]]

    all_node_with_code_info_and_attention = []
    g: dgl.DGLGraph
    for g in tqdm(mini_dl, total=len(mini_dl)):
        g, node_code_lines = g
        node_code_lines = [line[0]  for line in node_code_lines]
        g = g.to(args.device)
        # print(g)
        # print(node_code_lines)
        edge_nodes = []  # [(0,1), (1,2)]
        for x, y in zip(g.edges()[0].tolist(), g.edges()[1].tolist()):
            edge_nodes.append((x, y))
        node_lines = (g.ndata['_LINE'].int() - 1).tolist()  # joern start number from line 1
        # print(f'node_lines {node_lines}')
        test_label = dgl.max_nodes(g, "_LABEL").long()
        test_id = dgl.max_nodes(g, "_SAMPLE").int().item()  
        test_logits, feat_vec = model(g, mini_ds, need_graph_feat_vec=True)
        feat_vec.requires_grad = True
        # print('-----')
        feat_mask, edge_mask = explainer.explain_graph(g, feat=feat_vec)
        edge_mask = edge_mask.tolist()

        node_attentions = np.zeros(g.num_nodes())
        for e_idx, attention in enumerate(edge_mask):
            edge = edge_nodes[e_idx]
            node_attentions[edge[0]] += attention
            node_attentions[edge[1]] += attention


        # map node to corresponding code line
        ground_truth = get_ground_truth(test_id)
        raw_func = ground_truth['func'].split('\n')
        lines_attentions = [0] * len(raw_func)
        # print(f'func len:{len(raw_func)} node_lines {len(node_lines)}')

        for n_idx in range(g.num_nodes()):
            if n_idx == 0 : # this node is `_label = method` node, assign attentions for each line
                # joern truncated the very long code
                continue
            line_number = node_lines[n_idx]
            # print(node_code_lines[n_idx])
            node_number_of_lines = len(node_code_lines[n_idx].split("\n"))
            for _ in range(node_number_of_lines):
                lines_attentions[line_number] += node_attentions[n_idx]
                line_number += 1

        res = []
        for line_no, line in enumerate(raw_func):
            # (line_no, line, line_attention)
            res.append((line_no,line,lines_attentions[line_no]))
            # print(f"[{line_no}][{round(lines_attentions[line_no],4)}] {line}")

        all_node_with_code_info_and_attention.append(res)

    pickle.dump(all_node_with_code_info_and_attention,
                (result_dir() / f"ivdetect/{args.dataset}" / f"test_interpret_{split_idx}.pkl").open(mode='wb'))



def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    # %% Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_LABEL").long()
            # print('labels:', labels.shape)
            # print('logits:', logits.shape)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_mets = get_metrics_new(logits.tolist(), labels.tolist())

            # Evaluation
            # pred = logits.argmax(dim=1).cpu().detach().numpy()
            # print('pred:', pred.shape)
            val_mets = None
            if logger.log_val():
                val_mets = evaluate(model, val_dl=val_dl, val_ds=val_ds, logger=logger, args=args)
            logger.log(train_mets, val_mets)
            logger.save_logger()

        # Early Stopping
        if logger.stop():
            break
        logger.epoch()

    # Print test results
    test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # split for faster interpret result
    parser.add_argument("--interpret_split_idx" , type=int , required=False,default=-1)
    parser.add_argument("--interpret_total_split" , type=int , required=False,default=-1)

    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    interpret_split_idx = args.interpret_split_idx
    interpret_total_split = args.interpret_total_split

    # %% Load data
    dataset :str= args.dataset
    assert dataset.startswith('vul4c')
    cache_path = Path(f'{Path(__file__).parent.parent.parent.parent}/{dataset}')

    train_df = pd.read_json(cache_path / 'train.json')
    valid_df = pd.read_json(cache_path / 'valid.json')
    # test_df = pd.read_json(cache_path / 'test.json')
    test_df = pd.read_json(cache_path / 'test.json')

    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    g = dgl.graph((src_ids, dst_ids))
    print(g.to("cuda:0"))


    # args.max_patience = args.val_every * args.max_patience
    # %% Create model
    dev = args.device
    model = IVDetect(input_size=args.input_size, hidden_size=args.hidden_size)
    model.to(dev)
    ModelParameterCounter().summary(model,'IVdetect')

    set_seed(args)
    # Train loop
    # logger.load_logger()
    time_counter = RunTimeCounter()

    if args.do_train:
        train_ds = BigVulDatasetIVDetect(df=train_df, partition="train", dataset=dataset)
        print(len(train_ds))
        val_ds = BigVulDatasetIVDetect(df=valid_df, partition="valid", dataset=dataset)
        test_ds = BigVulDatasetIVDetect(df=test_df, partition="test", dataset=dataset)

        dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
        train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
        args.val_every = int(len(train_dl))
        args.log_every = int(len(train_dl) / 10)
        time_counter.stop('IVdetect Dataset Preprocessing')

        # %% Create Logger
        ID = get_run_id(args={})
        # ID = "202108121558_79d3273"
        logger = LogWriter(
            model, args, path=get_dir(result_dir() / f"ivdetect/{args.dataset}" / ID)
        )
        debug(args)
        logger.info(f'[Dataset] {dataset}')
        logger.info(args)
        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
        time_counter.stop('IVdetect Train Done!')
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)
        time_counter.stop('IVdetect Test Done!')

    if args.do_test:
        test_ds = BigVulDatasetIVDetect(df=test_df, partition="test", dataset=dataset)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=False)
        time_counter.stop('IVdetect Test Done!')


    if args.do_interpret:
        assert interpret_split_idx >= 0 and interpret_total_split >= 0
        test_df = pd.read_json(cache_path / 'test.json')
        print(f"interpret total:{len(test_df)}")
        split_num = math.ceil(len(test_df) / interpret_total_split)
        test_df = test_df[interpret_split_idx * split_num : (interpret_split_idx+1) * split_num ]
        print(f"split [{interpret_split_idx}/{interpret_total_split}]")
        print(f"split [{interpret_split_idx * split_num }:{(interpret_split_idx+1) * split_num }]")
        mini_ds = BigVulDatasetIVDetect(need_code_lines = True, df=test_df, partition="test", dataset=dataset)
        mini_dl = GraphDataLoader(mini_ds, batch_size=1, shuffle=False, num_workers=0)
        interpret(model, mini_dl, mini_ds, test_df.to_dict('records'),  interpret_split_idx,interpret_total_split ,args)

    if args.do_tsne:
        test_ds = BigVulDatasetIVDetect(df=test_df, partition="test", dataset=dataset)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
        tSNE_embedding(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=False)

if __name__ == '__main__':
    main()
