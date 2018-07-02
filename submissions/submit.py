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
from modules.neural.dataloading import WordDataset
from submissions.ensembling import ensemble_voting, read_dev_gold
from utils.dataloaders import clean_text
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import sort_by_lengths, load_checkpoint, load_checkpoint_pre_lm, load_checkpoint_with_f1

# _config = WASSA_CONF


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
    _y = []
    if dataset=='dev' or dataset=='train':
        lines = open(file, "r", encoding="utf-8").readlines()
        for line_id, line in enumerate(lines):
            columns = line.rstrip().split("\t")
            emotion = columns[0]
            text = columns[1:]
            text = clean_text(" ".join(text))
            _X.append(text)
            _y.append(emotion)

    return _X, _y


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
                                   desc="calculating posteriors..."):
            posts, attentions, y = process_batch_test_clf(model,
                                                          batch,
                                                          device)
            posteriors.append(posts)
            labels.append(y)

    posteriors = torch.cat(posteriors, dim=0)
    predicted = numpy.argmax(posteriors, 1)
    labels = numpy.array(torch.cat(labels, dim=0))

    return labels, predicted, posteriors

def write_predictions(predictions, encoder):
    fname = "predictions.txt"
    y_pred = [encoder.classes_[x] for x in predictions] # labels to strings
    with open(fname, "w") as outf:
        for prediction in y_pred:
            outf.write(prediction + "\t" + "\n")


def submission(dataset, models=[], lm=[], gold=[]):


    X, _ = load_test_wassa(dataset)


    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # load embeddings
    file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
    word2idx, idx2word, weights = load_word_vectors(file, 300)

    dummy_y = [[0] * 6] * len(X)
    dummy_y = torch.tensor(dummy_y)


    posteriors_list = []
    predicted_list = []

    for i in range(0,len(models)):

        checkpoint_name = models[i]

        if lm[i]:
            model, optimizer, word2idx, idx2word, loss, acc, f1 = \
                load_checkpoint_pre_lm(checkpoint_name)
        else:
            model, optimizer, vocab, loss, acc, f1 = \
                load_checkpoint_with_f1(checkpoint_name)

        #####################################################################
        # Define Dataloaders
        #####################################################################
        preprocessor = twitter_preprocessor()

        # for new experiments remember to empty _cache!
        test_set = WordDataset(X, dummy_y, word2idx, name="wassa_test_submit"+str(i),
                               preprocess=preprocessor)
        sampler = SequentialSampler(test_set)

        test_loader = DataLoader(test_set, batch_size=32,
                                 sampler=sampler)

        #####################################################################
        # Load Trained Model
        #####################################################################
        model.eval()
        model.to(config.DEVICE)
        print(model)

        #####################################################################
        # Evaluate Trained Model on test set & Calculate predictions
        #####################################################################
        labels, predicted, posteriors = test_clf(model=model, data_source=test_loader,
                                                 device=config.DEVICE)
        pprint(labels)
        pprint(predicted)

        predicted_list.append(predicted)
        posteriors_list.append(posteriors)

    pred = ensemble_voting(predicted_list, gold)
    #####################################################################
    # Create submission file with the predictions
    #####################################################################
    # write_predictions(pred, label_encoder)
    return



gold = read_dev_gold()
models=["wassa2M_ep2_GU_18-06-30_20:32:31", "wassa_rnn_600_18-07-02_13:45:15", "wassa_2M_LM_FT_simple_noconc_18-07-02_12:54:22"]
lm = [True, False, True]

submission(dataset="dev", models=models, lm=lm, gold=gold)

    # dataset = "train", "dev", or "test"
    # they should be stored in /datasets/wassa_2018