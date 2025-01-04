import pickle

import dgl
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from models.devign.dataset import BigVulDatasetDevign
from models.reveal.ggnn.model import GGNNSum
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')
from utils.utils import debug, get_run_id, processed_dir, get_metrics_probs_bce, cache_dir, set_seed, result_dir, get_dir , RunTimeCounter , ModelParameterCounter
from utils.dclass import BigVulDataset
from utils.my_log import LogWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import pandas as pd
from captum.attr import DeepLift
from models.reveal.model import MetricLearningModel


def evaluate(model, val_dl, val_ds, logger, args):
    model.eval()
    with torch.no_grad():
        # all_pred = torch.empty((0)).float().to(args.device)
        all_probs = torch.empty((0)).float().to(args.device)
        all_logits = torch.empty((0)).float().to(args.device)
        all_true = torch.empty((0)).float().to(args.device)
        for val_batch in tqdm(val_dl, total=len(val_dl), desc='Validing...'):
            val_batch = val_batch.to(args.device)
            val_labels = dgl.max_nodes(val_batch, "_LABEL")
            val_probs, val_logits = model(val_batch, val_ds)
            # val_preds = logits.argmax(dim=1)
            # all_pred = torch.cat([all_pred, val_logits])
            all_true = torch.cat([all_true, val_labels])
            all_probs = torch.cat([all_probs, val_probs])
            all_logits = torch.cat([all_logits, val_logits])
        # val_mets = get_metrics_probs_bce(all_true, all_pred) #
        val_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    return val_mets


def test(model, test_dl, test_ds, logger, args):
    model.eval()
    all_pred = torch.empty((0)).float().to(args.device)
    all_probs = torch.empty((0)).float().to(args.device)
    all_logits = torch.empty((0)).float().to(args.device)
    all_true = torch.empty((0)).float().to(args.device)
    # test_batch === 1
    with torch.no_grad():
        for test_batch in test_dl:
            g:dgl.DGLGraph
            g, code_info = test_batch
            g = g.to(args.device)
            assert g.num_nodes() == len(code_info)
            test_labels = dgl.max_nodes(g, "_LABEL")
            test_probs, test_logits = model(g, test_ds)  
            all_probs = torch.cat([all_probs, test_probs])
            all_logits = torch.cat([all_logits, test_logits])
            all_true = torch.cat([all_true, test_labels])
        test_mets = get_metrics_probs_bce(all_true, all_probs, all_logits)
    logger.test(test_mets)
    return test_mets


def train(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, logger, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    loss_function = nn.BCELoss()
    for epoch in range(args.epochs):
        for batch in tqdm(train_dl, total=len(train_dl), desc='Training...'):
            # Training
            model.train()
            batch = batch.to(args.device)
            probs, logits = model(batch, train_ds)
            labels = dgl.max_nodes(batch, "_LABEL")
            # print('labels:', labels.shape)
            # print('logits:', logits.shape)
            # print(logits, labels)
            loss = loss_function(probs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # train_mets = get_metrics_probs_bce(labels, logits)
            train_mets = get_metrics_probs_bce(labels, probs, logits)

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


from interpret_model import InterpretModel
def interpret(model, test_dl, test_ds, logger, args):

    interpret_model = InterpretModel()
    interpret_model.to(args.device)
    dl = DeepLift(interpret_model)



    model.load_state_dict(torch.load('../../../storage/results/reveal/vul4c_dataset/202306170931_v1/best_f1.model'))
    model.eval()
    all_node_with_code_info_and_attention = []

    with torch.no_grad():
        for test_batch in tqdm(test_dl):  # batch_size = 1
            g :dgl.DGLGraph
            g, code_info = test_batch
            g = g.to(args.device)
            assert g.num_nodes() == len(code_info)
            test_labels = dgl.max_nodes(g, "_LABEL")
            _, ggnn_output = model.save_ggnn_output(g, test_ds,is_test=True)

            with torch.enable_grad():
                ggnn_output = ggnn_output.clone().detach()
                ggnn_output.requires_grad = True
                attribution = dl.attribute(ggnn_output,target=1)
                assert len(code_info) == attribution.shape[1]
                node_attentions = (attribution.squeeze(dim = 0).sum(dim = 1) * 10000).tolist()     # node attentions ( scale 100x )
                # print(node_attentions)
                res = []
                for  (line_no,code) , attention  in  zip(code_info ,node_attentions):
                    # print(f'[{line_no}][{round(attention,4)}] {code}')
                    res.append((line_no,code,attention))
                all_node_with_code_info_and_attention.append(res)
                # print(res)
                # print(ggnn_output)

        pickle.dump(all_node_with_code_info_and_attention,
                        (result_dir() / f"reveal/{args.dataset}" / f"test_interpret.pkl").open(mode='wb'))



def save_ggnn_output(model, train_dl, train_ds, val_dl, val_ds, test_dl, test_ds, args):
    print("***** Running save_ggnn_output *****")

    model.eval()

    ID = {
        'vul4c_dataset' : '202306170931_v1',
        'vul4c_insert_comments_dataset': '202306170931_v1',
        'vul4c_rename_identifier_dataset': '202306170931_v1',
        'vul4c_rm_comments_dataset': '202306170931_v1',
        'vul4c_unexecuted_code_dataset': '202306170931_v1',
    }

    path = result_dir() / f"reveal/{args.dataset}" / f"{ID[args.dataset]}/best_f1.model"

    model.load_state_dict(torch.load(path))

    with torch.no_grad():
        cache_all = []
        for data_dl, data_ds in [(train_dl, train_ds), (val_dl, val_ds), (test_dl, test_ds)]:
            all_pred = torch.empty((0)).float().to(args.device)
            all_true = torch.empty((0)).float().to(args.device)
            all_ids = torch.empty((0)).float().to(args.device)
            for test_batch in data_dl:
                test_batch = test_batch.to(args.device)
                test_labels = dgl.max_nodes(test_batch, "_LABEL")
                test_ids = dgl.max_nodes(test_batch, "_SAMPLE")
                test_logits, ggnn_output = model.save_ggnn_output(test_batch, data_ds)
                all_pred = torch.cat([all_pred, ggnn_output])
                all_true = torch.cat([all_true, test_labels])
                all_ids = torch.cat([all_ids, test_ids])
            cache_all.append((all_pred, all_true , all_ids))
        cache_path = get_dir(cache_dir() / f"ggnn_output/{args.dataset}/")  # balanced
        # cache_path = get_dir(cache_dir() / f"ggnn_output/{args.dataset}/not_balance/")
        torch.save(cache_all, cache_path / 'ggnn_output.bin')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_after_ggnn', action='store_true')
    args = parser.parse_args()
    configs = json.load(open('./config.json'))
    for item in configs:
        args.__dict__[item] = configs[item]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    g = dgl.graph((src_ids, dst_ids))
    print(g.to("cuda:0"))

    # %% Load data
    dataset :str= args.dataset
    assert dataset.startswith('vul4c')
    cache_path = Path(__file__).parent.parent.parent.parent.parent / dataset
    train_df = pd.read_json(cache_path / 'train.json')
    valid_df = pd.read_json(cache_path / 'valid.json')
    # test_df = pd.read_json(cache_path / 'test.json')
    test_df = pd.read_json(cache_path / 'test.json')

    time_counter = RunTimeCounter()
    train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset=dataset)
    val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
    test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset)

    dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
    train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
    dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
    val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
    test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)
    time_counter.stop('Reveal GGNN preprocessing')

    args.val_every = int(len(train_dl))
    args.log_every = int(len(train_dl) / 5)
    # args.max_patience = args.val_every * args.max_patience
    # %% Create model
    dev = args.device
    model = GGNNSum(input_dim=args.input_size, output_dim=args.hidden_size, num_steps=args.num_steps)
    model.to(dev)
    ModelParameterCounter().summary(model,'Reveal GGNN')

    set_seed(args)
    ID = get_run_id(args={})
    # ID = "202108121558_79d3273"
    logger = LogWriter(
        model, args, path=get_dir(result_dir() / f"reveal/{args.dataset}" / ID)
    )
    debug(args)
    logger.info(args)

    if args.do_interpret:
        def my_collate_fn(data):
            assert len(data) == 1
            return data[0]

        test_ds = BigVulDatasetDevign(need_code_info=True, df=test_df, partition="test", dataset=dataset)
        test_dl = GraphDataLoader(test_ds, batch_size=1, collate_fn=my_collate_fn,**dl_args)
        interpret(model,test_dl,test_ds,logger,args)

        return


    if args.save_after_ggnn:  # save after train
        vul4c_cache_path = Path(__file__).parent.parent.parent.parent.parent / 'vul4c_dataset'
        train_df = pd.read_json(vul4c_cache_path / 'train.json')
        valid_df = pd.read_json(vul4c_cache_path / 'valid.json')
        # test_df = pd.read_json(cache_path / 'test.json')
        test_df = pd.read_json(cache_path / 'test.json')

        time_counter = RunTimeCounter()
        train_ds = BigVulDatasetDevign(df=train_df, partition="train", dataset='vul4c_dataset')
        val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset='vul4c_dataset')
        test_ds = BigVulDatasetDevign(df=test_df, partition="test", dataset=dataset)

        dl_args = {"drop_last": False, "shuffle": True, "num_workers": 6}
        train_dl = GraphDataLoader(train_ds, batch_size=args.train_batch_size, **dl_args)
        dl_args = {"drop_last": False, "shuffle": False, "num_workers": 6}
        val_dl = GraphDataLoader(val_ds, batch_size=args.test_batch_size, **dl_args)
        test_dl = GraphDataLoader(test_ds, batch_size=args.test_batch_size, **dl_args)

        save_ggnn_output(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
                         test_dl=test_dl, test_ds=test_ds, args=args)
        return
        # %% Create Logger


    # Train loop
    # logger.load_logger()

    if args.do_train:
        train(model, train_dl=train_dl, train_ds=train_ds, val_dl=val_dl, val_ds=val_ds,
              test_dl=test_dl, test_ds=test_ds, logger=logger, args=args)
        time_counter.stop('Reveal GGNN train done!')
    if args.do_test:
        test(model, test_dl=test_dl, test_ds=test_ds, args=args, logger=logger)
        time_counter.stop('Reveal GGNN test done!')



if __name__ == '__main__':
    main()
    # dataset = 'reveal'
    # cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
    # df = pd.read_pickle(cache_path)
    # train_df = df[df.partition == 'train']
    # valid_df = df[df.partition == 'valid']
    # test_df = df[df.partition == 'test']
    # val_ds = BigVulDatasetDevign(df=valid_df, partition="valid", dataset=dataset)
    #
    # val_ds.item(10)
