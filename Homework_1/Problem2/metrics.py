import numpy as np


def accuracy_score(y_true, y_predict, percent=None):
    y_one = y_predict[:, 1]

    if percent:
        length = int(len(y_one) * percent / 100)
        y_true = y_true[y_one.argsort()][::-1][:length]
        y_one = np.sort(y_one)[::-1][:length]

    prediction = (y_one >= .5).astype('int')
    result = (prediction == y_true).mean()
    return result


def precision_score(y_true, y_predict, percent=None):
    y_one = y_predict[:, 1]

    if percent:
        length = int(len(y_one) * percent / 100)
        y_true = y_true[y_one.argsort()][::-1][:length]
        y_one = np.sort(y_one)[::-1][:length]

    prediction = (y_one > .5).astype('int')
    result = np.logical_and(prediction, y_true).sum() / prediction.sum()
    return result


def recall_score(y_true, y_predict, percent=None):
    y_one = y_predict[:, 1]

    if percent:
        length = int(len(y_one) * percent / 100)
        y_true = y_true[y_one.argsort()][::-1][:length]
        y_one = np.sort(y_one)[::-1][:length]

    prediction = (y_one > .5).astype('int')
    result = np.logical_and(prediction, y_true).sum() / y_true.sum()
    return result


def lift_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict, percent)

    if percent:
        length = int(len(y_true) * percent / 100)
        y_true = y_true[y_predict[:, 1].argsort()][::-1][:length]

    result = precision / y_true.mean()
    return result


def f1_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    result = 2 * (precision * recall) / (precision + recall)
    return result
