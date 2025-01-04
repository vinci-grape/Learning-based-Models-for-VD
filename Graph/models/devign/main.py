import math
import pickle

import dgl
import sys

import numpy as np

from dataset import BigVulDatasetDevign
from model import DevignModel
from pathlib import Path
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str((Path(__file__).parent.parent.parent)))
from utils.utils import debug, get_run_id, processed_dir, get_metrics_probs_bce, cache_dir, set_seed, result_dir, \
    get_dir, get_metrics_new, ModelParameterCounter , RunTimeCounter
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from gnn_explainer import GNNExplainer
import argparse
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')

dataset2id = {
    'vul4c_dataset': '202307191025_v1',
    'vul4c_insert_comments_dataset': '202307191025_v1',
    'vul4c_rename_identifier_dataset': '202307191025_v1',
    'vul4c_rm_comments_dataset': '202307191025_v1',
    'vul4c_unexecuted_code_dataset': '202307191025_v1',
}

def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        all_probs = []
        all_true = []
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL")
            val_probs = model(val_batch)
            all_probs.extend(val_probs.tolist())
            all_true.extend(val_labels.tolist())
        val_mets = get_metrics_new(all_probs, all_true)
    return val_mets


def test(model, test_dl, test_ds,mini_dl, logger, args):

    path = result_dir() / f"devign/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()


    all_probs  = []
    all_true   = []
    all_ids    = []

    with torch.no_grad():
        test_batch : dgl.DGLGraph
        for g in tqdm(test_dl):
            g = g.to(args.device)
            test_labels = dgl.max_nodes(g, "_LABEL")
            test_ids = dgl.max_nodes(g, "_SAMPLE")
            test_probs = model(g)
            all_probs.extend(test_probs.tolist())
            all_true.extend(test_labels.tolist())
            all_ids.extend(test_ids.tolist())
        assert len(all_probs) == len(all_true) == len(all_ids)
        test_mets = get_metrics_new(all_probs , all_true )
    if logger:
        logger.test(test_mets)
    else:
        print(test_mets)

    with open(result_dir() / f"devign/{args.dataset}" / f"test.json",mode='w') as f:
        pred = np.argmax(all_probs,axis=-1).tolist()
        ans = []
        for id , p in zip(all_ids,pred):
            ans.append({ 'id' : int(id) , 'pred':p })
        json.dump( ans,f , indent=4)

    return test_mets

def tSNE_embedding(model, test_dl, test_ds, logger, args):
    path = result_dir() / f"devign/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()

    all_probs  = []
    all_true   = []
    all_ids    = []
    embedding_shape = 248

    all_tSNE_embedding = []
    with torch.no_grad():
        test_batch : dgl.DGLGraph
        for g in tqdm(test_dl):
            g = g.to(args.device)
            test_labels = dgl.max_nodes(g, "_LABEL")
            test_ids = dgl.max_nodes(g, "_SAMPLE")
            _ , embedding = model(g, output_tSNE_embedding = True , tSNE_embedding_shape  = embedding_shape)
            embedding_shape = max(embedding.shape[1] , embedding_shape)
            all_tSNE_embedding.extend(
                [(embedding, label) for embedding, label in zip(embedding.tolist(), test_labels.tolist())])

            # all_probs.extend(test_probs.tolist())
            # all_true.extend(test_labels.tolist())
            # all_ids.extend(test_ids.tolist())

    print(f'tSNE done!! len:{len(all_tSNE_embedding)}')
    with open(result_dir() / f"devign/{args.dataset}"   / "test_tSNE_embedding.pkl" , mode='wb') as f:
        pickle.dump(all_tSNE_embedding , f)



    # pickle.dump(all_node_with_code_info_and_attention,
    #             (result_dir() / f"devign/{args.dataset}" / f"test_tSNE_embedding.pkl").open(mode='wb'))


def interpret(model, mini_dl,split_idx,total_splits, args):
    path = result_dir() / f"devign/{args.dataset}" / f"{dataset2id[args.dataset]}/best_f1.model"
    model.load_state_dict(torch.load(path))
    model.eval()

    explainer = GNNExplainer(model,num_hops=1,log=False)

    mini_batch: dgl.DGLGraph


    all_node_with_code_info_and_attention = []
    for mini_batch in tqdm(mini_dl):
        g: dgl.DGLGraph
        g, code_infos = mini_batch
        # print(g)
        # print(code_infos)
        g = g.to('cuda:0')
        num_nodes = g.num_nodes()
        node_attentions = np.zeros(num_nodes)
        edge_nodes = []  # [(0,1), (1,2)]
        for x, y in zip(g.edges()[0].tolist(), g.edges()[1].tolist()):
            edge_nodes.append((x, y))

        feat_mask, edge_mask = explainer.explain_graph(g, feat=g.ndata['_WORD2VEC'])
        edge_mask = edge_mask.tolist()

        for e_idx, attention in enumerate(edge_mask):
            edge = edge_nodes[e_idx]
            if 0 <= edge[0] <= num_nodes and  0 <= edge[1] <= num_nodes:
                node_attentions[edge[0]] += attention
                node_attentions[edge[1]] += attention

        # print(node_attentions, len(node_attentions))
        # print(code_infos, len(code_infos))

        # merge code , line , attention
        res = []
        for code_info, attention in zip(code_infos, node_attentions):
            res.append((code_info[0].item(), code_info[1][0], attention))

        all_node_with_code_info_and_attention.append(res)

    print(f'saving test_interpret_{split_idx}.pkl...')
    pickle.dump(all_node_with_code_info_and_attention,
                (result_dir() / f"devign/{args.dataset}" / f"test_interpret_{split_idx}.pkl").open(mode='wb'))


def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    # %% Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        batch : dgl.DGLGraph
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            probs = model(batch)
            labels = dgl.max_nodes(batch, "_LABEL")
            labels = labels.long()

            # print('labels:', labels.shape)
            # print('logits:', logits.shape)
            # print(logits, labels)
            loss = loss_function(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_mets = get_metrics_new( probs.tolist() , labels.tolist())

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
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    interpret_split_idx = args.interpret_split_idx
    interpret_total_split = args.interpret_total_split

    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    g = dgl.graph((src_ids, dst_ids))
    print(g.to("cuda:0"))


    # %% Load data
    dataset:str = args.dataset
    assert dataset.startswith('vul4c')
    # cache_path = Path(f'/home/icy/source/Vul4C/{dataset}')
    cache_path = Path(f'{Path(__file__).parent.parent.parent.parent}/{dataset}')

    train_df = pd.read_json(cache_path / 'train.json')
    valid_df = pd.read_json(cache_path / 'valid.json')
    # test_df = pd.read_json(cache_path / 'test.json')
    test_df = pd.read_json(cache_path / 'test.json')

    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
    if args.do_train or args.do_test or args.do_tsne :
        test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset, vulonly=False)
        test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
        mini_dl = GraphDataLoader(test_ds, batch_size=1, **dl_args)
    # print(test_ds.df[test_ds.df.vul == 1]._id.values)
    # args.max_patience = args.val_every * args.max_patience
    # %% Create model
    dev = args.device
    model = DevignModel(input_dim=args.input_size, output_dim=args.hidden_size)
    model.to(dev)
    ModelParameterCounter().summary(model,'Devign')
    print(args)

    set_seed(args)
    # Train loop
    # logger.load_logger()
    time_counter = RunTimeCounter()

    if args.do_train:
        train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset=dataset, not_balance=args.not_balance)
        val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
        dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
        train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
        args.val_every = int(len(train_dl))
        args.log_every = int(len(train_dl) / 5)
        # %% Create Logger
        ID = get_run_id(args={})
        # ID = "202108121558_79d3273"
        logger = LogWriter(
            model, args, path=get_dir(result_dir() / f"devign/{args.dataset}" / ID)
        )
        debug(args)
        logger.info(f'[Dataset] {dataset}')
        logger.info(args)
        time_counter.stop('Devign Dataset Preprocessing')
        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
        time_counter.stop('Devign Train Done!')
        test(model, test_dl=test_dl, test_ds=test_ds, mini_dl = mini_dl,args=args, logger=logger)
        time_counter.stop('Devign Test Done!')

    if args.do_test:
        test(model, test_dl=test_dl, test_ds=test_ds,mini_dl = mini_dl, args=args, logger=False)
        time_counter.stop('Devign Test Done!')

    if args.do_tsne:
        tSNE_embedding(model, test_dl=test_dl, test_ds=test_ds ,args=args, logger=False)
    if args.do_interpret:
        assert interpret_split_idx >= 0 and interpret_total_split >= 0
        test_df = pd.read_json(cache_path / 'test.json')
        print(f"interpret total:{len(test_df)}")
        split_num = math.ceil(len(test_df) / interpret_total_split)
        test_df = test_df[interpret_split_idx * split_num : (interpret_split_idx+1) * split_num ]
        print(f"split [{interpret_split_idx}/{interpret_total_split}]")
        print(f"split [{interpret_split_idx * split_num }:{(interpret_split_idx+1) * split_num }]")

        test_ds = BigVulDatasetDevign(need_code_info=True,df=test_df, partition="test", dataset=dataset, vulonly=False)
        mini_dl = GraphDataLoader(test_ds, batch_size=1, **dl_args)
        interpret(model,mini_dl,interpret_split_idx,interpret_total_split,args)
        # interpret results


if __name__ == '__main__':
    main()
