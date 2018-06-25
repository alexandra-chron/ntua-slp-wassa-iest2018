import os
import pickle
from collections import Counter

import nltk
import numpy
from torch.utils.data import Dataset
from tqdm import tqdm

from config import BASE_PATH


def pad_text(text, length):
    """
    Zero padding for text
    Args:
        text (numpy.array): a  1D numpy array of word_ids

    Returns:
        (numpy.array): a padded array
        :param length:
        :type length:
    """
    x = numpy.zeros(length, dtype='int32')
    if text.size < length:
        x = numpy.pad(text, (0, length - text.size % length), 'constant')
    elif text.size > length:
        x = text[0:length]
    elif text.size == length:
        x = text

    return x


def vectorize(text, word2idx, max_length=None):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length (): the maximum length of the input sequences
    Returns: zero-padded list of ids
    """

    if max_length is not None:
        words = numpy.zeros(max_length).astype(int)
        text = text[:max_length]
    else:
        words = numpy.zeros(len(text)).astype(int)

    for i, token in enumerate(text):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            words[i] = word2idx["<unk>"]

    return words


class Vocab(object):
    """
    The Vocab Class, holds the vocabulary of a corpus and
    mappings from tokens to indices and vice versa.
    """

    def __init__(self):
        self.PAD = "<pad>"
        self.SOS = "<sos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"
        self.TRG = "[#triggerword#]"

        self.vocab = Counter()

        self.tok2id = dict()
        self.id2tok = dict()

        self.size = 0

    def read_tokens(self, tokens):
        self.vocab.update(tokens)

    def trim(self, size):
        self.tok2id = dict()
        self.id2tok = dict()
        self.build(size)

    def __add_token(self, token):
        index = len(self.tok2id)
        self.tok2id[token] = index
        self.id2tok[index] = token

    def build(self, size=None):
        self.__add_token(self.PAD)
        self.__add_token(self.SOS)
        self.__add_token(self.EOS)
        self.__add_token(self.UNK)
        self.__add_token(self.TRG)

        for w, k in self.vocab.most_common(size):
            self.__add_token(w)

        self.size = len(self)

    def __len__(self):
        return len(self.tok2id)


class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionality such as
    caching.
    """

    def __init__(self, name=None):
        """

        Args:
            name (str): the name of the dataset. It is needed for caching.
                if it is None then caching is disabled.
        """
        self.name = name

    def prepare(self, data):
        raise NotImplementedError

    @staticmethod
    def __check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def __get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache", "{}.p".format(self.name))

    def __write_cache(self, data):
        self.__check_cache()

        cache_file = self.__get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def prepared_data(self, data):
        """
        Caches the output of the `prepare` method from children classes.
        All children classes should implement a `prepare` method.
        """
        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.prepare(data)

        # using cache
        cache_file = self.__get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.prepare(data)
            self.__write_cache(data)
            return data


class WordDataset(BaseDataset):

    def __init__(self, X, y, word2idx, max_length=0,
                 name=None, preprocess=None):
        """
        Args:
            X (): list of training samples
            y (): list of training labels
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
        """

        super().__init__(name)

        # set properties
        self.data = X
        self.labels = y
        self.word2idx = word2idx

        # define preprocess method
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = lambda x: [_x.lower().split() for _x in x]

        # load prepared data. Utilize the `prepared_data()` method
        # from the parent BaseDataset class, which wraps around
        # the `self.prepared_data()` caching its output
        self.data = self.prepared_data(X)

        # define max_length. If not passed then use dataset's max length
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

        self.dataset_statistics()

    def prepare(self, data):
        desc = "PreProcessing dataset {}...".format(self.name)
        data = [self.preprocess(x) for x in tqdm(data, desc=desc)]
        return data

    def dataset_statistics(self):
        words = Counter()
        for x in self.data:
            words.update(x)
        unks = {w: v for w, v in words.items() if w not in self.word2idx}
        # unks = sorted(unks.items(), key=lambda x: x[1], reverse=True)
        total_words = sum(words.values())
        total_unks = sum(unks.values())

        print("Total words: {}, Total unks:{} ({:.2f}%)".format(
            total_words, total_unks, total_unks * 100 / total_words))

        print("Unique words: {}, Unique unks:{} ({:.2f}%)".format(
            len(words), len(unks), len(unks) * 100 / len(words)))

        # label statistics
        print("Labels statistics:")
        counts = Counter(self.labels)
        stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
                 for k, v in sorted(counts.items())}
        print(stats)
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, label = self.data[index], self.labels[index]

        sample = vectorize(sample, self.word2idx, self.max_length)

        length = min(len(self.data[index]), self.max_length)

        return sample, label, length


class LangModelDataset(BaseDataset):
    def __init__(self, data, max_length=0,
                 name=None, preprocess=None,
                 vocab=None, vocab_size=None):
        """

        Args:
            file (str): the file containing the text data.
            seq_len (int): sequence length -
            if stateful==True: refers to size of backpropagation through time.
            if stateful==False: refers to max length of the sentences.
            the dataset will be split to small sequences of bptt size.
            vocab (Vocab): a vocab instance.
        """
        super().__init__(name)

        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = self.tokenize

        # step 1 - tokenize the dataset
        self.data, self.vocab = self.prepared_data(data)

        # step 2 - build the vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab.build(vocab_size)

        # step 3 - define max_length
        # If not passed then use dataset's max length
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    def tokenize(self, text):
        return text.lower().split()

    def prepare(self, data):
        _data = []
        _vocab = Vocab()
        desc = "PreProcessing dataset {}...".format(self.name)
        for line in tqdm(data, desc=desc):
            tokens = self.preprocess(line)
            _vocab.read_tokens(tokens)
            tokens.append(_vocab.EOS)
            _data.append(tokens)
        return _data, _vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        inputs = self.data[index]
        targets = inputs[1:]

        inputs = vectorize(inputs, self.vocab.tok2id, self.max_length)
        targets = vectorize(targets, self.vocab.tok2id, self.max_length)

        length = min(len(self.data[index]), self.max_length)

        return inputs, targets, length
