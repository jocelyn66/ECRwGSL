"""Evaluation script."""

import argparse
import json
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F

import models
import numpy as np
from utils.name2object import name2model
from optimizers import *
from dataset.graph_dataset import GDataset
from sklearn.model_selection import PredefinedSplit

import torch.optim

import models.ecr_model
import optimizers.regularizers as regularizers
from optimizers.ecr_optimizer import *
from rs_hyperparameter import rs_tunes, rs_hp_range, rs_set_hp_func
from utils.train import *
from utils.evaluate import *
from utils.visual import *
from dataset.dataset_process import preprocess_function
from dataset.graph_dataset import GDataset, get_examples_indices
from datasets import load_dataset
from utils.evaluate import visual_graph

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertModel
)

DATA_PATH = './data/'

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--model-dir', '--dir', help="Model path")  # /home/h3c/00/running/logs/07_24/ecb+/gs_ecr-gsl_gae+_02_19_55/
parser.add_argument('--num', '--n', type=int, default=-1)


def test(model_dir, num=-1):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    args.use_cuda = torch.cuda.is_available()  # 有无
    # args.use_cuda = False
    if not args.use_cuda:
        ValueError("WARNING: CUDA is not available!!!")  # 用cpu的话, 直接注释
    args.device = torch.device("cuda" if args.use_cuda else "cpu")  # atth

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    dataset = GDataset(args)

    pos_weight = {}
    norm = {}
    adj_norm = {}
    for split in ['Train', 'Dev', 'Test']:
        adj = dataset.adjacency[split]
        adj_norm[split] = preprocess_adjacency(adj)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        pos_weight[split] = torch.tensor([pos_weight], device=args.device)
        norm[split] = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        
    event_true_sub_indices = {}
    event_false_sub_indices = {}
    entity_true_sub_indices = {}
    entity_false_sub_indices = {}
    recover_true_sub_indices = {}
    recover_false_sub_indices = {}

    for split in ['Train', 'Dev', 'Test']:
        event_true_sub_indices[split], event_false_sub_indices[split] = get_examples_indices(dataset.event_coref_adj[split])
        # entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        entity_true_sub_indices[split], entity_false_sub_indices[split] = get_examples_indices(dataset.entity_coref_adj[split])
        recover_true_sub_indices[split], recover_false_sub_indices[split] = get_examples_indices(dataset.adjacency[split])

        # bert################################
     #Load Datasets
    data_files = {}
    data_files["train"] = args.train_file
    data_files["dev"] = args.dev_file
    data_files["test"] = args.test_file
    datasets = load_dataset("json", data_files=data_files)
    #Load Schema
    with open(args.schema_path, 'r') as f:
        schema_list = json.load(f)
        doc_schema = schema_list[0]
        event_schema = schema_list[1]
        entity_schema = schema_list[2]

    #introduce PLM
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    plm = BertModel.from_pretrained(args.plm_name)

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.train_cache_file
    )

    dev_dataset = datasets["dev"]
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.dev_cache_file
    )

    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.test_cache_file
    )

    datasets = {'Dev':dev_dataset, 'Test':test_dataset}
    ######################

    # load pretrained model weights
    model = getattr(models, name2model[args.model])(args, tokenizer, plm, schema_list)
    if args.use_cuda:
        model.cuda()

    model_name = "model{}_feat-d{}_h1-d{}_h2-d{}.pt".format(num, args.feat_dim, args.hidden1, args.hidden2)
    print(model_name)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = GAEOptimizer(model, optim_method, args.n_nodes, norm, pos_weight, args.valid_freq, args.use_cuda)

    model.eval()
    # eval#############
    # 计算边
    # 画图
    # 统计
    for split in ['Train', 'Dev', 'Test']:

        print(split, ':\n')

        print('\tmetrics:')
        _, mu = optimizer.eval(datasets[split], adj_norm[split], dataset.adjacency[split], split)

        hidden_emb = mu.data.detach().cpu().numpy()
        metrics1 = test_model(hidden_emb, dataset.event_idx[split], event_true_sub_indices[split], event_false_sub_indices[split])
        print("\t\tevent coref:" + format_metrics(metrics1, split))

        entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        metrics2 = test_model(hidden_emb, entity_idx, entity_true_sub_indices[split], entity_false_sub_indices[split])
        print("\t\tentity coref:" + format_metrics(metrics2, split))

        metrics3 = test_model(hidden_emb, list(range(args.n_nodes[split])), recover_true_sub_indices[split], recover_false_sub_indices[split])
        print("\t\treconstruct adj:" + format_metrics(metrics3, split))

        # 计算边:
        pred_adj = sigmoid(np.dot(hidden_emb, hidden_emb.T))
        # 统计

        # 画图: 
        visual_graph(model_dir, split, dataset.adjacency[split], pred_adj, num)



if __name__ == "__main__":
    args = parser.parse_args()
    test(args.model_dir, args.num)
