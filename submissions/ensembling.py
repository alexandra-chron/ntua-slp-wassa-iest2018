import os
import pickle

import numpy
from scipy.stats import stats
from sklearn.metrics import accuracy_score, f1_score

from config import DATA_DIR

def acc(y, y_hat):
    return accuracy_score(y, y_hat)


def f1_macro(y, y_hat):
    return f1_score(y, y_hat, average='macro')


def f1_micro(y, y_hat):
    return f1_score(y, y_hat, average='micro')

def ensemble_voting(predictions, gold):
    stacked = numpy.stack(predictions, axis=0)
    modals = stats.mode(stacked, axis=0)[0].squeeze().astype(int)

    accuracy = acc(gold, modals)
    f1 = f1_macro(gold, modals)
    print("acc: ",accuracy)
    print("f1: ",f1)

    return modals

def read_dev_gold():
    file = os.path.join(DATA_DIR, "wassa_2018", "trial-v3.labels")

    _y = []

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split("\t")
        emotion = columns[0]
        _y.append(emotion)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)


    gold = label_encoder.transform(_y)
    return gold