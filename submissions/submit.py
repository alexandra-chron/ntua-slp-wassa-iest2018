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
from submissions.ensembling import ensemble_voting, read_dev_gold, ensemble_posteriors
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

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split("\t")
        emotion = columns[0]
        text = columns[1:]
        text = clean_text(" ".join(text))
        _X.append(text)
        _y.append(emotion)


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
    fname = "predictions_val_13.txt"
    y_pred = [encoder.classes_[x] for x in predictions] # labels to strings
    with open(fname, "w") as outf:
        for prediction in y_pred:
            outf.write(prediction + "\t" + "\n")


def submission(dataset, models=[], lm=[], gold=[]):


    X = load_test_wassa(dataset)


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
        test_set = WordDataset(X, dummy_y, word2idx, name="wassa_test_submit",
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
        # pprint(labels)
        pprint(predicted)

        predicted_list.append(predicted)
        posteriors_list.append(posteriors)

    # pred, accuracy, f1  = ensemble_voting(predicted_list, gold, dataset)
    pred, accuracy, f1 = ensemble_posteriors(posteriors_list, gold, dataset)

    #####################################################################
    # Create submission file with the predictions3M_GU13__35_noconc_2att
    #####################################################################
    write_predictions(pred, label_encoder)
    return



gold = read_dev_gold()
models=["wassa_rnn_350_18-07-02_16:30:49", #"wassa_rnn400_repr_18-07-03_20:08:37",
        "wassa_sent250_18-07-06_14:30:42", #"wassa_sent400_18-07-06_14:41:00",
        "5M__GU_35_noconc_18-07-05_14:59:25", #"5M__GU_69_conc_18-07-06_15:13:33",
        "2M_LM_FT_GU_24__GU_57_18-07-08_01:02:16", #"wassa_2M_LM_FT_GU24_conc_GU45_18-07-03_17:53:31",
        "LM_FT_simple__GU_5_7_18-07-07_17:29:54", #"step7_noquestions_18-07-03_01:05:37",
        "2M_600_LM_FT_GU13__3_5_noconc_18-07-03_20:55:27",
        "wassa_LM_FT_2M_GU23_conc_GU69_18-07-04_18:34:12", #"wassa_LM_FT_2M_GU23_noconc_GU1214_18-07-05_00:57:46", "wassa_LM_FT_2M_GU23_conc_GU36_18-07-04_02:26:56", "wassa_LM_FT_2M_GU23_noconc_GU69_18-07-05_00:31:39",
        "3M_GU13__35_noconc_2att",
        "TWITTER_3M_GU_24__45_conc_18-07-04_16:09:59"]

lm = [False, #False,          # word2vec
      False, #False,          # sentiment
      True, #True,            # 5M simple
      True, #True,            # 2M GU 2,4
      True, #True,            # 2M simple
      True,                  # 2M GU 1,3
      True, #True, True, True,          # 2M GU 2,3
      True,                  # 3M GU 1,3
      True]                  # 3M Gu 2,4

submission(dataset="trial-v3", models=models, lm=lm, gold=gold)

    # dataset = "train", "dev", or "test"
    # they should be stored in /datasets/wassa_2018