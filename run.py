import json
import logging
import os
import sys
import itertools

import torch.optim
import math
import numpy as np

from dataset.graph_dataset import load_dataset
from utils.evaluate import compute_metrics, format_metrics
import models
import optimizers.regularizers as regularizers
from optimizers import *
from config import parser
from utils.name2object import name2model
from rs_hyperparameter import rs_tunes, rs_hp_range, rs_set_hp_func
from utils.train import *


def set_logger(args):
    save_dir = get_savedir(args.dataset, args.model, args.init_g, args.updater, args.grid_search)
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

    if not (args.grid_search or args.rand_search):
        save_dir = set_logger(args)
        with open(os.path.join(save_dir, "config.json"), 'a') as fjson:
            json.dump(vars(args), fjson)

    model_name = "model_d{}_l{}.pt".format(
        args.rank, args.n_layers)
    logging.info("Init graph = {}".format(args.init_g))
    logging.info("#Layer = {}".format(args.n_layers))
    logging.info("lr = {}".format(args.learning_rate))
    logging.info("Dropout = {}".format(args.dropout))
    logging.info(args)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    # load data############################
    dataset = load_dataset(args.dataset)
    args.sizes = dataset.get_shape()
    logging.info("\t " + str(dataset.get_shape()))
    train_data = dataset['train']
    valid_data = dataset['valid']
    test_data = dataset['test']

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!")
    args.device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(models, name2model[args.model])(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(args.device)    # GUP

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    regularizer = None
    if args.regularizer:
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optimizer = ECROptimizer(model, optim_method, args.valid_freq, args.batch_size, regularizer,
                            use_cuda, args.dropout)

    # start train######################################
    counter = 0
    best_f = None
    best_epoch = None
    best_model_path = ''
    logging.info("\t ---------------------------Start Optimization-------------------------------")
    for epoch in range(args.max_epochs):
        model.train()
        if use_cuda:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        loss = optimizer.epoch(train_data)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(epoch, loss))
        if math.isnan(loss.item()):
            break

        # val#####################################
        model.eval()
        valid_loss, ranks = optimizer.evaluate(test_data)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))
        if (epoch + 1) % args.valid_freq == 0:
            valid_metrics = compute_metrics(ranks)
            logging.info(format_metrics(valid_metrics, split="valid_"+args.metrics))
            valid_f = valid_metrics["F"]
            if not best_f or valid_f > best_f:
                best_f = valid_f
                counter = 0
                best_epoch = epoch
                logging.info("\t Saving model at epoch {} in {}".format(epoch, save_dir))
                best_model_path = os.path.join(save_dir, '{}_{}'.format(epoch, model_name))
                torch.save(model.cpu().state_dict(), best_model_path)
                if use_cuda:
                    model.cuda()

            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass

    # test#########################
    logging.info("\t ---------------------------Optimization finished---------------------------")
    if not best_f:
        best_model_path = os.path.join(save_dir, model_name)
        torch.save(model.cpu().state_dict(), best_model_path)
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(best_model_path))  # load best model
    if use_cuda:
        model.cuda()
    model.eval()  # no BatchNormalization Dropout

    # Validation metrics
    logging.info("Evaluation Valid Set:")
    _, rank = optimizer.evaluate(valid_data)
    valid_metrics = compute_metrics(rank)
    logging.info(format_metrics(valid_metrics))

    # Test metrics
    logging.info("Evaluation Test Set:")
    _, rank = optimizer.evaluate(train_data)
    test_metrics = compute_metrics(rank)
    logging.info(format_metrics(test_metrics))

    logging.info("{:.3f}({:.3f})".format(test_metrics['F'], valid_metrics['F']))

    logging.info("\t ---------------------------done---------------------------")
    end_model = datetime.datetime.now()
    logging.info('this model runtime: %s' % str(end_model - start_model))
    logging.info("\t ---------------------------end---------------------------")
    return test_metrics


def rand_search(args):
    pass

    best_f = 0
    best_hps = []
    best_fs = []

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
        if test_metrics['F'] > best_f:
            best_f = test_metrics['F']
            best_fs.append(best_f)
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
