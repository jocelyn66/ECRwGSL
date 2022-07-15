import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


def format_metrics(metrics, split):
    # f_score, roc_score, ap_score, p, r
    str = '{}set: F1={:.5f}, ROC={:.5f}, AP={:.5f}, P={:.5f}, R={:.5f}'.format(split, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
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


def test_model(emb, target_adj, event_idx):

    # 根据共指关系计算AUC等
    # 大矩阵: embedding, 共指关系矩阵
    # event mention在大矩阵中的下标,用于提取正负例,方法 取上三角矩阵(不含对角线)
    # extract event mentions(trigger)
    event_emb = emb[event_idx]
    target_event_adj = target_adj[event_idx]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    pred_event_adj = sigmoid(np.dot(emb, emb.T))

    mask = np.triu_indices(len(event_idx), len(event_idx)-1)  # flatten(), 上三角元素的下标
    preds = pred_event_adj.flatten()[mask]
    target = target_event_adj.flatten()[mask].int()
    preds_true = preds[target==1]
    preds_false = preds[target==0]

    preds_all = np.hstack([preds_true, preds_false])
    labels_all = np.hstack([np.ones(len(preds_true)), np.zeros(len(preds_false))])

    # 计算metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f_score = get_bcubed(labels_all, preds_all)
    p = precision_score(labels_all, preds_all)
    r = recall_score(labels_all, preds_all) 
    return f_score, roc_score, ap_score, p, r
