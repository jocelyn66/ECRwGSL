from cProfile import label
from pydoc import describe
from statistics import median
import numpy as np
from sklearn.metrics import label_ranking_loss, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from utils.train import sigmoid
from utils.visual import *


def format_metrics(metrics, split):
    # f_score, roc_score, ap_score, p, r
    str = 'AUC={:.5f}, AP={:.5f}'.format(metrics[0], metrics[1])
    return str


# cluster####################
def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(cdict[el1] & cdict[el2]))


def mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(ldict[el1] & ldict[el2]))
        

def precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts."""
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
        for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])


def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
        for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])


def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta**2) * (p_val * r_val / (beta**2 * p_val + r_val))   


def bcubed(gold_lst, predicted_lst):
    # in: gold_list: cluster set
    """
    Takes gold, predicted.
    Returns recall, precision, f1score
    """
    gold = {i:{cluster} for i,cluster in enumerate(gold_lst)}
    pred = {i:{cluster} for i,cluster in enumerate(predicted_lst)}
    p = precision(pred, gold)
    r = recall(pred, gold)
    return r, p, fscore(p, r)
    
###################################


def get_bcubed(labels, preds):
    return f1_score(labels, preds)


def test_model(emb, indices, true_indices, false_indices):
    # target_adj: 

    # ????????????????????????AUC???
    # ?????????: embedding, ??????????????????
    # event mention????????????????????????,?????????????????????,?????? ??????????????????(???????????????)
    # extract event mentions(trigger)

    emb_ = emb[indices, :]
    # target_event_adj = target_adj[event_idx, :][:, event_idx]

    # Predict on test set of edges
    pred_adj = sigmoid(np.dot(emb_, emb_.T))

    # mask = np.triu_indices(len(indices), 1)  # ????????????????????????list
    # preds = pred_event_adj[mask]
    # target = target_sub_adj[mask]

    preds_true = pred_adj[true_indices]
    preds_false = pred_adj[false_indices]

    # np.random.shuffle(preds_false)
    # preds_false = preds_false[:len(preds_true)] # ???:???=1:1

    preds_all = np.hstack([preds_true, preds_false])
    labels_all = np.hstack([np.ones(len(preds_true)), np.zeros(len(preds_false))])

    # ??????metrics
    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    # f_score = get_bcubed(labels_all, preds_all>threshold)
    # p = precision_score(labels_all, preds_all>threshold)
    # r = recall_score(labels_all, preds_all>threshold) 
    return auc_score, ap_score


def visual_graph(path, split, orig, pred_adj, num=-1, threshold=0.5):  # ??????????????????(??????, ?????????), ??????graph

    # plot_adj(path, split+" original visual graph", orig, num)  # ??????
    
    pred_adj_ = np.where(pred_adj>threshold, 1, 0)
    plot_adj(path, split+" pred graph - visual", pred_adj_, num=num)

    # plot_adj(path, split+" weighted pred visual graph", pred_adj, num=num, weighted=True)


def degree_analysis(path, split, orig, pred_adj, num=-1, threshold=0.5):

    degree = np.sum(orig, axis=1).astype(np.int)
    # degree_list_ = np.bincount(degree)
    max_degree = np.max(degree)
    min_degree = np.min(degree)
    mean_degree = np.mean(degree)
    median_degree = np.median(degree)
    print("\t\torig graph degree:", '\tmean:', mean_degree, '\tmedian:', median_degree, '\tmax:', max_degree, '\tmin', min_degree)

    # plot_hist(path, split+"original degree graph", degree_list_, num=num)

    adj = np.where(pred_adj>threshold, 1., 0.)
    pred_degree = np.sum(adj, axis=1).astype(np.int)
    # degree_list = np.bincount(pred_degree)  # ??????:???, ???:count

    max_degree = np.max(pred_degree)
    min_degree = np.min(pred_degree)
    mean_degree = np.mean(pred_degree)
    median_degree = np.median(pred_degree)
    print("\t\tpred graph degree:", '\tmean:', mean_degree, '\tmedian:', median_degree, '\tmax:', max_degree, '\tmin', min_degree)

    plot_hist(path, split+" original graph - degree", split+" pred graph - degree", degree, pred_degree, num=num)
