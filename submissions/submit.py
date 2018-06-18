import os
import pickle

import numpy
import torch
from pprint import pprint
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import config
from config import DATA_DIR, BASE_PATH
from model.params import WASSA_CONF
from model.pipelines import eval_clf
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import clean_text
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import sort_by_lengths

_config = WASSA_CONF


def load_test_wassa(dataset):
    """
    Read a file and return a dictionary of the data, in the format:
    line_id:{emotion, text}

    Args:
        dataset:

    Returns:

    """
    file = os.path.join(DATA_DIR, "wassa_2018", "{}.csv".format(dataset))

    _X = []

    if dataset=='dev':
        lines = open(file, "r", encoding="utf-8").readlines()
        for line_id, line in enumerate(lines):
            columns = line.rstrip().split("\t")
            # emotion = columns[0]
            text = columns[1:]
            text = clean_text(" ".join(text))
            _X.append(text)
            # y.append(emotion)

    return _X


def process_batch_test_clf(model, batch, device):
    # read batch
    batch = list(map(lambda x: x.to(device), batch))
    inputs, labels, lengths = batch

    # sort batch
    lengths, sort, unsort = sort_by_lengths(lengths)
    inputs = sort(inputs)

    # feed data to model
    outputs, attentions = model(inputs, lengths)

    # unsort outputs
    outputs = unsort(outputs)
    # attentions = unsort(attentions)

    return outputs, attentions, labels


def test_clf(model, data_source, device):
    model.eval()

    posteriors = []
    labels = []

    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(data_source, 1),
                                   desc="calculating..."):
            posts, attentions, y = process_batch_test_clf(model,
                                                          batch,
                                                          device)
            posteriors.append(posts)
            labels.append(y)

    posteriors = torch.cat(posteriors, dim=0)
    predicted = numpy.argmax(posteriors, 1)
    labels = numpy.array(torch.cat(labels, dim=0))

    return labels, predicted

def write_predictions(predictions, encoder):
    fname = "predictions.txt"
    y_pred = [encoder.classes_[x] for x in predictions]
    with open(fname, "w") as outf:
        for prediction in y_pred:
            outf.write(prediction + "\t" + "\n")

def submission(dataset, pretrained_emb=False):

    X = load_test_wassa(dataset)
    with open("label_encoder", "rb") as f:
        label_encoder = pickle.load(f)

    if pretrained_emb:
        # load embeddings
        file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
        word2idx, idx2word, weights = load_word_vectors(file, 300)

        dummy_y = [[0] * 6] * len(X)
        dummy_y = torch.tensor(dummy_y)
        #####################################################################
        # Define Dataloaders
        #####################################################################
        preprocessor = twitter_preprocessor()
        test_set = WordDataset(X, dummy_y, word2idx, name="wassa_test", max_length=50,
                               preprocess=preprocessor)
        sampler = SequentialSampler(test_set)

        test_loader = DataLoader(test_set, batch_size=_config["batch_eval"],
                                 sampler=sampler)
        file = os.path.join(BASE_PATH, "checkpoints", "wassa")
        with open(file, "rb") as f:
            model = torch.load(f)

        print(model)

        labels, predicted = test_clf(model=model, data_source=test_loader,
                                           device=config.DEVICE)
        pprint(labels)
        pprint(predicted)

        write_predictions(predicted, label_encoder)


submission(dataset="dev", pretrained_emb=True)
