import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt

import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts import emoticons
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
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

        for w, k in self.vocab.most_common(size):
            self.__add_token(w)

        self.size = len(self)

    def __len__(self):
        return len(self.tok2id)


class LangModelSampler(Sampler):
    """
    Samples elements per chunk. Suitable for Language Models.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size, batch):
        """
        Define how to construct batches

        Given a list of sequences, organize the sequences in each batch
        in such a way, so that each RNN gets the proper (next) sequence.

        For example, given the following sequence and with batch=2:
        ┌ a b c d e ┐
        │ f g h i j │
        │ k l m n o │
        │ p q r s t │
        │ u v w x y │
        └ z - - - - ┘

        the batches will be:
        ┌ a b c d e ┐    ┌ f g h i j ┐    ┌ k l m n o ┐
        └ p q r s t ┘    └ u v w x y ┘    └ z - - - - ┘

        Args:
            size (int): number of sequences
            batch (int): batch size
        """
        self.size = size
        self.batch = batch

        # split the corpus in chunks of size `corpus_seqs / batch_size`
        self.chunks = numpy.array_split(numpy.arange(self.size), batch)

    def get_batch(self, index):
        """
        Fill each batch with the i-th sequence from each chunk.
        If the batch size does not evenly divides the chunks,
        then some chunks will have one less sequence, so the last batch
        will have fewer samples.
        Args:
            index (int):

        Returns:

        """
        batch = []
        for chunk in self.chunks:
            if index < chunk.size:
                batch.append(chunk[index])
        return batch

    def batches(self):
        for i in range(self.chunks[0].size):
            yield self.get_batch(i)

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return self.size


class LangModelDataset(Dataset):
    def __init__(self, X, seq_len=0,
                 tokenize=None, vocab=None, name=None, vocab_size=None, stateful=True):

        self.X = X
        self.vocab = Vocab()
        self.name = name

        if tokenize is not None:
            self.tokenize = tokenize

        # step 1 - tokenize the dataset
        data = []
        for i, line in enumerate(self.X):
            tokens = self.tokenize(line)
            self.vocab.read_tokens(tokens)

            if stateful:
                data.extend(tokens + ['<eos>'])
            else:
                data.append(tokens + ['<eos>'])

        # step 2 - build the vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab.build(vocab_size)

        # step 4 - create the inputs and their corresponding targets
        if stateful:
            # split the input data to chunks of size bptt
            if seq_len <= 0:
                raise ValueError("seq_len has to be > 0 when stateful=True")

            self.seq_len = seq_len
            self.inputs = [data[i:i + self.seq_len]
                           for i in range(0, len(data), self.seq_len)]

            data.append(self.vocab.PAD)
            self.targets = [data[i:i + self.seq_len]
                            for i in range(1, len(data), self.seq_len)]
        else:
            if seq_len == 0:
                self.seq_len = max([len(x) for x in data])

            else:
                self.seq_len = seq_len

            # todo: draw inputs and targets dynamically
            self.inputs = [d[:self.seq_len] for d in data]
            self.targets = [(d + [self.vocab.PAD])[1:self.seq_len + 1]
                            for d in data]

        self.data = data

    @staticmethod
    def tokenize(text):
        return text.split()



    def debug(self):
        lengths = [len(x) for x in self.data]
        plt.hist(lengths, density=1, bins=20)
        plt.axvline(self.seq_len, color='k', linestyle='dashed', linewidth=1)
        plt.show()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        data = vectorize(self.inputs[index],
                         self.vocab.tok2id,
                         self.seq_len)
        targets = vectorize(self.targets[index],
                            self.vocab.tok2id,
                            self.seq_len)

        return data, targets, len(self.inputs[index])


class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionality such as
    caching.
    """

    def __init__(self, X, y,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """

        Args:
            X (): List of training samples
            y (): List of training labels
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            label_transformer (LabelTransformer):
        """
        self.data = X
        self.labels = y
        self.name = name
        self.label_transformer = label_transformer

        if preprocess is not None:
            self.preprocess = preprocess

        self.data = self.load_preprocessed_data()

        self.set_max_length(max_length)

        if verbose:
            self.dataset_statistics()

    def set_max_length(self, max_length):
        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    def dataset_statistics(self):
        raise NotImplementedError

    def preprocess(self, name, X):
        """
        Preprocessing pipeline
        Args:
            X (list): list of training examples

        Returns: list of processed examples

        """
        raise NotImplementedError

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache",
                            "preprocessed_{}.p".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_preprocessed_data(self):

        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self._write_cache(data)
            return data


class WordDataset(BaseDataset):

    def __init__(self, X, y, word2idx,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            X (): list of training samples
            y (): list of training labels
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
            label_transformer (LabelTransformer):
        """
        self.word2idx = word2idx

        BaseDataset.__init__(self, X, y, max_length, name, label_transformer,
                             verbose, preprocess)

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
        if isinstance(self.labels[0], float):
            print("Mean:{:.4f}, Std:{:.4f}".format(numpy.mean(self.labels),
                                                   numpy.std(self.labels)))
        else:
            try:
                counts = Counter(self.labels)
                stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
                         for k, v in sorted(counts.items())}
                print(stats)
            except:
                print("Not implemented for mclf")
        print()

    def preprocess(self, name, dataset):
        preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time',
                       'date', 'number'],
            annotate={"hashtag", "elongated", "allcaps", "repeated",
                      'emphasis',
                      'censored'},
            all_caps_tag="wrap",
            fix_text=True,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        ).pre_process_doc

        desc = "PreProcessing dataset {}...".format(name)

        data = [preprocessor(x) for x in tqdm(dataset, desc=desc)]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training sample
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.
        """
        sample, label = self.data[index], self.labels[index]

        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx, self.max_length)

        if self.label_transformer is not None:
            label = self.label_transformer.transform(label)

        if isinstance(label, (list, tuple)):
            label = numpy.array(label)

        return sample, label, len(self.data[index]), index

