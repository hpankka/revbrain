import numpy as np


# ----------MAE----------
def mae(preds, gts):

    preds = np.mean(preds, axis=1)

    errors = {}

    errors_all = np.abs(gts-preds)

    errors['errors_all'] = errors_all

    errors['mae_all'] = np.mean(errors_all)
    errors['std_all'] = np.std(errors_all)

    errors['mae_true'] = np.mean(errors_all[:, 64:])
    errors['std_true'] = np.std(errors_all[:, 64:])

    errors['mae_func_of_time'] = np.mean(errors_all, axis=0)

    return errors


# ----------Phase detection----------

def get_th_preds(x, th):
    """
    Assign a predicted category based on votes.
    If no category receives at least th votes, assign empty category 4.
    """
    pred_values = np.max(x, axis=1)
    y_pred = np.argmax(x, axis=1)

    y_pred = np.where(pred_values >= th, y_pred, 4)

    return y_pred

def get_precision(y, y_pred, trgt_ind):
    """
    TP / TP+FP
    """
    positives = np.isin(y_pred, trgt_ind)
    tp = np.sum(y[positives] == y_pred[positives])

    return (tp / np.sum(positives))*100

def get_recall(y, y_pred, trgt_ind):
    """
    TP / TP+FN
    """
    positives = np.isin(y_pred, trgt_ind)
    tp = np.sum(y[positives] == y_pred[positives])

    positive_targets = np.isin(y, trgt_ind)
    fn = np.sum(y[positive_targets] != y_pred[positive_targets])

    return (tp / (tp+fn))*100

def phase_detection(x, y):

    trgt_ind = [0, 1, 2, 3]
    thresholds = np.arange(0.26, 0.98, 0.01)

    precision = []  # TP / TP+FP
    recall = []  # TP / TP+FN
    for th in thresholds:
        y_pred = get_th_preds(x, th)
        precision.append(get_precision(y, y_pred, trgt_ind))
        recall.append(get_recall(y, y_pred, trgt_ind))

    return precision, recall, thresholds*100
