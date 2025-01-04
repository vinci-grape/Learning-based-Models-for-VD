import os
import sys
from torch.optim import Adam
from graph_dataset import create_dataset, DataSet
from pathlib import Path
sys.path.append(str((Path(__file__).parent)))
sys.path.append(str((Path(__file__).parent.parent.parent)))
from models.reveal.model import MetricLearningModel
from trainer import train, show_representation
import numpy as np
import random
import torch
import warnings
from utils.utils import cache_dir

warnings.filterwarnings('ignore')
import argparse
from tsne import plot_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_dir', help='Base Dir',
        default='/data/ReVeal/data/full_experiment_real_data_processed/bigvul/full_graph/balanced/all_balanced'
    )
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()
    assert args.dataset.startswith('vul4c')
    print(args)
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    base_dir = args.base_dir
    ggnn_output = torch.load(cache_dir() / f"ggnn_output/{args.dataset}/ggnn_output.bin")
    dataset = create_dataset(
        ggnn_output=ggnn_output,
        batch_size=128,
        output_buffer=sys.stderr
    )
    num_epochs = 200
    dataset.initialize_dataset(balance=True)
    # train_features, train_targets,_ = dataset.prepare_data(
    #     dataset.train_entries, list(range(len(dataset.train_entries)))
    # )
    # plot_embedding(train_features, train_targets, args.name + '-before-training')
    # plot_embedding(train_features, train_targets, args.dataset + '-before-training')
    print(dataset.hdim, end='\t')
    model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=256)
    model.cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    print(model)
    train(model, dataset, optimizer, num_epochs, dataset_name=args.dataset, cuda_device=0, max_patience=10,
          output_buffer=sys.stderr)
    # show_representation(model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0,
    #                     args.dataset + '-after-training-triplet')
    #
    # model = MetricLearningModel(input_dim=dataset.hdim, hidden_dim=256, lambda1=0, lambda2=0)
    # model.cuda()
    # optimizer = Adam(model.parameters(), lr=0.001)
    # train(model, dataset, optimizer, num_epochs, cuda_device=0, max_patience=10, output_buffer=sys.stderr)
    # show_representation(model, dataset.get_next_test_batch, dataset.initialize_test_batches(), 0,
    #                     args.dataset + '-after-training-no-triplet')
    pass
