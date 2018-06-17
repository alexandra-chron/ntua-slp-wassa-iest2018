import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import config
from config import DATA_DIR, BASE_PATH
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import clean_text
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor


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

def submission(dataset, pretrained_emb=False):

    X = load_test_wassa(dataset)

    if pretrained_emb:
        # load embeddings
        file = os.path.join(BASE_PATH, "embeddings", "word2vec_300_6_20_neg.txt")
        word2idx, idx2word, weights = load_word_vectors(file, 300)

        dummy_y = [[0] * 6] * len(X)
        #####################################################################
        # Define Dataloaders
        #####################################################################
        preprocessor = twitter_preprocessor()
        test_set = WordDataset(X, dummy_y, word2idx, name="wassa_test",
                                preprocess=preprocessor)
        sampler = SequentialSampler(test_set)

        test_loader = DataLoader(test_set, batch_size=config["batch_eval"],
                                 sampler=sampler)
        model = Classifier(embeddings=weights, out_size=6, **config).to(config.DEVICE)
        print(model)




submission(dataset="dev", pretrained_emb=False)