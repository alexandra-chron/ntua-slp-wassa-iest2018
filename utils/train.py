"""
This file contains functions with logic that is used in almost all models,
with the goal of avoiding boilerplate code (and bugs due to copy-paste),
such as training pipelines.
"""
import glob
import math
import os
import pickle

import numpy
import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, recall_score, accuracy_score, \
    precision_score, jaccard_similarity_score
from torch.nn import ModuleList
from torch.utils.data import DataLoader

from config import TRAINED_PATH, BASE_PATH, DEVICE
from modules.nn.dataloading import WordDataset
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocess


def load_pretrained_model(name):
    model_path = os.path.join(TRAINED_PATH, "{}.model".format(name))
    # model_conf_path = os.path.join(TRAINED_PATH, "{}.conf".format(name))
    model = torch.load(model_path)
    # model_conf = pickle.load(open(model_conf_path, 'rb'))

    return model
        # , model_conf


def load_pretrained_models(name):
    models_path = os.path.join(TRAINED_PATH)
    # fmodel_confs = sorted(glob.glob(os.path.join(models_path,
    #                                              "{}*.conf".format(name))))
    fmodels = sorted(glob.glob(os.path.join(models_path,
                                            "{}*.model".format(name))))
    for model in fmodels:
        print("loading model {}".format(model))
        yield torch.load(model)


def get_pretrained(pretrained):
    pretrained_model = load_pretrained_model(pretrained)
    return pretrained_model


def load_datasets(datasets, train_batch_size, eval_batch_size, token_type,
                  preprocessor=None,
                  params=None, word2idx=None, label_transformer=None):
    if params is not None:
        name = "_".join(params) if isinstance(params, list) else params
    else:
        name = None

    loaders = {}
    if token_type == "word":
        if word2idx is None:
            raise ValueError

        if preprocessor is None:
            preprocessor = twitter_preprocess()

        print("Building word-level datasets...")
        for k, v in datasets.items():
            _name = "{}_{}".format(name, k)
            dataset = WordDataset(v[0], v[1], word2idx, name=_name,
                                  preprocess=preprocessor,
                                  label_transformer=label_transformer)
            batch_size = train_batch_size if k == "train" else eval_batch_size
            loaders[k] = DataLoader(dataset, batch_size, shuffle=True,
                                    drop_last=True)

    return loaders


def load_embeddings(model_conf):
    word_vectors = os.path.join(BASE_PATH, "embeddings",
                                "{}.txt".format(model_conf["embeddings_file"]))
    word_vectors_size = model_conf["embed_dim"]

    # load word embeddings
    print("loading word embeddings...")
    return load_word_vectors(word_vectors, word_vectors_size)


def calc_pearson(y, y_hat):
    score = pearsonr(y, y_hat)[0]
    if math.isnan(score):
        return 0
    else:
        return score


def get_metrics(task, ordinal):
    _metrics = {
        "reg": {
            "pearson": calc_pearson,
        },
        "bclf": {
            "acc": lambda y, y_hat: accuracy_score(y, y_hat),
            "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                          average='macro'),
            "recall": lambda y, y_hat: recall_score(y, y_hat,
                                                    average='macro'),
            "f1": lambda y, y_hat: f1_score(y, y_hat,
                                            average='macro'),
        },
        "clf": {
            "acc": lambda y, y_hat: accuracy_score(y, y_hat),
            "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                          average='macro'),
            "recall": lambda y, y_hat: recall_score(y, y_hat,
                                                    average='macro'),
            "f1": lambda y, y_hat: f1_score(y, y_hat,
                                            average='macro'),
        },
        "mclf": {
            "jaccard": lambda y, y_hat: jaccard_similarity_score(
                numpy.array(y), numpy.array(y_hat)),
            "f1-macro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='macro'),
            "f1-micro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='micro'),
        },
    }
    _monitor = {
        "reg": "pearson",
        "bclf": "f1",
        "clf": "f1",
        "mclf": "jaccard",
    }
    _mode = {
        "reg": "max",
        "bclf": "max",
        "clf": "max",
        "mclf": "max",
    }

    if ordinal:
        task = "reg"

    metrics = _metrics[task]
    monitor = _monitor[task]
    mode = _mode[task]

    return metrics, monitor, mode


def unfreeze_module(module, optimizer):
    for param in module.parameters():
        param.requires_grad = True

    optimizer.add_param_group(
        {'params': list(
            module.parameters())}
    )


def model_training(trainer, epochs, unfreeze=0, checkpoint=False):
    print("Training...")
    for epoch in range(epochs):
        trainer.train()
        trainer.eval()

        if unfreeze > 0:
            if epoch == unfreeze:
                print("Unfreeze transfer-learning model...")
                subnetwork = trainer.model.feature_extractor
                if isinstance(subnetwork, ModuleList):
                    for fe in subnetwork:
                        unfreeze_module(fe.encoder, trainer.optimizer)
                        unfreeze_module(fe.attention, trainer.optimizer)
                else:
                    unfreeze_module(subnetwork.encoder, trainer.optimizer)
                    unfreeze_module(subnetwork.attention, trainer.optimizer)

        print()

        if checkpoint:
            trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
