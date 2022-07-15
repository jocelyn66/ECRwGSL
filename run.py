import json
import logging
import os
import sys
import time

import torch.optim
import math
import numpy as np

import models
import optimizers.regularizers as regularizers
from optimizers.ecr_optimizer import *
from config import parser
from utils.name2object import name2model
from rs_hyperparameter import rs_tunes, rs_hp_range, rs_set_hp_func
from utils.train import *
from utils.evaluate import *
from dataset.dataset_process import preprocess_function


from dataset.graph_dataset import GDataset
from datasets import load_dataset

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertModel
)
import json


def set_logger(args):
    save_dir = get_savedir(args.dataset, args.model, args.encoder, args.decoder, args.rand_search)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.logs")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    print("Saving logs in: {}".format(save_dir))
    return save_dir


def train(args, hps=None, set_hp=None, save_dir=None):

    start_model = datetime.datetime.now()
    torch.manual_seed(2022)

    if args.rand_search:
        set_hp(args, hps)

    if not args.rand_search:
        save_dir = set_logger(args)
        with open(os.path.join(save_dir, "config.json"), 'a') as fjson:
            json.dump(vars(args), fjson)

    model_name = "model_feat-d{}_h1-d{}_h2-d{}.pt".format(
        args.feat_dim, args.hidden1, args.hidden2)
    logging.info(args)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    # load data############################处理成图
    # dataset = load_dataset(args.dataset)
    # args.sizes = dataset.get_shape()
    # logging.info("\t " + str(dataset.get_shape()))
    # train_data = dataset['train']
    # valid_data = dataset['valid']
    # test_data = dataset['test']
    #######

    dataset = GDataset(args)

    adj_train, args.n_nodes, event_coref_adj = dataset.train_adjacency, dataset.n_nodes, dataset.event_coref_adj

    # Some preprocessing:
    adj_norm = preprocess_graph(adj_train)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

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
    ######################

    # create 
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!")
    args.device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(models, name2model[args.model])(args, train_dataset, tokenizer, plm, schema_list, adj_norm)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(args.device)    # GUP

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    regularizer = None
    if args.regularizer:
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optimizer = GAEOptimizer(model, optim_method, args.n_nodes, norm, pos_weight, args.valid_freq, use_cuda, args.dropout)

    # start train######################################
    counter = 0
    best_f1 = None
    best_epoch = None
    best_model_path = ''
    hidden_emb = None
    logging.info("\t ---------------------------Start Optimization-------------------------------")
    for epoch in range(args.max_epochs):
        t = time.time()
        model.train()
        if use_cuda:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        loss, mu = optimizer.epoch()
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(epoch, loss))
        if math.isnan(loss.item()):
            break

        # valid training set
        hidden_emb = mu.data.numpy()

        model.eval()

        metrics = test_model(hidden_emb, event_coref_adj, dataset.event_idx)
        logging.info(format_metrics(metrics, 'Trian'))
        logging.info("time=", "{:.5f}".format(time.time() - t))

        # val#####################################

        # # 无监督
        # dev_loss, mu = optimizer.eval()
        # hidden_emb = mu.data.numpy()
        # metrics = test_model(hidden_emb, adj_dev['Dev'], dataset.event_idx['Dev'])
        # logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, dev_loss))

        # # 有监督
        # model.eval()
        # if (epoch + 1) % args.valid_freq == 0:
        #     # valid loss
        #     # valid metircs
        #     metrics = test_model()   # F1
        #     logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))
        #     logging.info(format_conll('Valid_F1'+valid_f1))  
        #     if not best_f1 or valid_f1 > best_f1:
        #         best_f1 = valid_f1
        #         counter = 0
        #         best_epoch = epoch
        #         logging.info("\t Saving model at epoch {} in {}".format(epoch, save_dir))
        #         best_model_path = os.path.join(save_dir, '{}_{}'.format(epoch, model_name))
        #         torch.save(model.cpu().state_dict(), best_model_path)
        #         if use_cuda:
        #             model.cuda()

        #     else:
        #         counter += 1
        #         if counter == args.patience:
        #             logging.info("\t Early stopping")
        #             break
        #         elif counter == args.patience // 2:
        #             pass
        # ###################

    logging.info("\t ---------------------------Optimization finished---------------------------")

    # test#########################
    # if not best_f1:
    #     best_model_path = os.path.join(save_dir, model_name)
    #     torch.save(model.cpu().state_dict(), best_model_path)
    # else:
    #     logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
    #     model.load_state_dict(torch.load(best_model_path))  # load best model
    # if use_cuda:
    #     model.cuda()
    # model.eval()  # no BatchNormalization Dropout

    # # Test metrics
    # logging.info("Evaluation Test Set:")
    # test_f1 = None
    # conll_f1 = run_conll_scorer(args.output_dir)
    # logging.info(conll_f1)

    end_model = datetime.datetime.now()
    logging.info('this model runtime: %s' % str(end_model - start_model))
    logging.info("\t ---------------------------done---------------------------")
    return best_f1


def rand_search(args):
    pass

    best_f1 = 0
    best_hps = []
    best_f1s = []

    save_dir = set_logger(args)
    logging.info("** Random Search **")

    args.tune = rs_tunes
    logging.info(rs_hp_range)
    hyperparams = args.tune.split(',')

    if args.tune == '' or len(hyperparams) < 1:
        logging.info("No hyperparameter specified.")
        sys.exit(0)
    grid = rs_hp_range[hyperparams[0]]
    for hp in hyperparams[1:]:
        grid = zip(grid, rs_hp_range[hp])

    grid = list(grid)
    logging.info('* {} hyperparameter combinations to try'.format(len(grid)))

    for i, grid_entry in enumerate(list(grid)):
        if not (type(grid_entry) is list):
            grid_entry = [grid_entry]
        grid_entry = flatten(grid_entry)    # list
        hp_values = dict(zip(hyperparams, grid_entry))
        logging.info('* Hyperparameter Set {}:'.format(i))
        logging.info(hp_values)

        test_metrics = train(args, hp_values, rs_set_hp_func, save_dir)
        logging.info('{} done'.format(grid_entry))
        if test_metrics['F'] > best_f1:
            best_f1 = test_metrics['F']
            best_f1s.append(best_f1)
            best_hps.append(grid_entry)
    logging.info("best hyperparameters: {}".format(best_hps))


if __name__ == "__main__":
    start = datetime.datetime.now()
    if parser.rand_search:
        rand_search(parser)
    else:
        train(parser)
    end = datetime.datetime.now()
    logging.info('total runtime: %s' % str(end - start))
    sys.exit()
