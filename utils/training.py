import datetime
import math
import os
import sys
import time

import numpy
import torch
from sklearn.utils import compute_class_weight

from config import BASE_PATH


def trim_hidden(inputs, hidden):
    """
    In the case where the last batch is smaller than batch_size,
    we will need to keep only the first N hidden states,
    where N is the number of samples in the last batch
    Args:
        inputs: the inputs in the last batch size
        hidden: the hidden state produced by the penultimate batch

    Returns: hidden
        the trimmed hidden state

    """
    batch_size = inputs.size(0)

    if isinstance(hidden, tuple):
        hidden_size = hidden[0].size(1)
    else:
        hidden_size = hidden.size(1)

    # do nothing
    if batch_size == hidden_size:
        return hidden

    # trim the hidden state to the remaining samples in the batch
    if isinstance(hidden, tuple):
        hidden = (hidden[0][:, :batch_size, :].contiguous(),
                  hidden[1][:, :batch_size, :].contiguous())
    else:
        hidden = hidden[:, :batch_size, :].contiguous()

    return hidden


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(v.detach() for v in h)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def progress(loss, epoch, batch, batch_size, dataset_size, start, interval,
             ppl=False):
    batches = math.ceil(float(dataset_size) / batch_size)

    if batch % interval != 0 and batch != batches:
        return

    count = batch * batch_size
    bar_len = 30
    filled_len = int(round(bar_len * count / float(dataset_size)))

    log_bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if ppl:
        log_losses = 'Loss:{:.2f}, PPL:{:.2f}'.format(loss, math.exp(loss))
    else:
        log_losses = 'Loss:{:.2f}'.format(loss)
    log_epoch = 'Epoch {}'.format(epoch)
    log_batches = 'Batch {}/{}'.format(batch, batches)
    time_info = 'Time {}'.format(timeSince(start, batch / batches))
    _progress_str = "\r \r [{}] {}, {} - {} - {}".format(log_bar,
                                                         log_epoch,
                                                         log_batches,
                                                         log_losses,
                                                         time_info)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def progress_multitask(loss, epoch, batch, batch_size, dataset_size, interval,
                       mode):
    batches = math.ceil(float(dataset_size) / batch_size)

    if batch % interval != 0 and batch != batches:
        return

    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    log_bar = '=' * filled_len + '-' * (bar_len - filled_len)

    log_losses = ' CE: {:.4f}, PPL: {:.4f}'.format(loss, math.exp(loss))
    log_epoch = ' Epoch {}'.format(epoch)
    log_batches = ' Batch {}/{}'.format(batch, batches)
    _progress_str = "\r \r {} [{}] ({}) {} ... {}".format(log_epoch,
                                                          log_bar,
                                                          mode,
                                                          log_batches,
                                                          log_losses, )
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def epoch_summary(dataset, loss):
    msg = "\t{:7s}: Avg Loss = {:.4f},\tAvg Perplexity = {:.4f}"
    print(msg.format(dataset, loss, math.exp(loss)))


def get_class_labels(y):
    """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive',
                                    'neutral', 'positive', ...]
    :return: sorted unique class labels
    """
    return numpy.unique(y)


def get_class_weights(y):
    """
    Returns the normalized weights for each class
    based on the frequencies of the samples
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    weights = compute_class_weight('balanced', numpy.unique(y), y)

    d = {c: w for c, w in zip(numpy.unique(y), weights)}

    return d


def class_weigths(targets, to_pytorch=False):
    w = get_class_weights(targets)
    labels = get_class_labels(targets)
    if to_pytorch:
        return torch.FloatTensor([w[l] for l in sorted(labels)])
    return labels


def sort_by_lengths(lengths):
    """
    Sort batch data and labels by length.
    Useful for variable length inputs, for utilizing PackedSequences
    Args:
        lengths (neural.Tensor): tensor containing the lengths for the data

    Returns:
        - sorted lengths Tensor
        - sort (callable) which will sort a given iterable
            according to lengths
        - unsort (callable) which will revert a given iterable to its
            original order

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    _, original_idx = sorted_idx.sort(0, descending=True)
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

    if lengths.data.is_cuda:
        reverse_idx = reverse_idx.cuda()

    sorted_lengths = sorted_lengths[reverse_idx]

    def sort(iterable):
        if len(iterable.shape) > 1:
            return iterable[sorted_idx.data][reverse_idx]
        else:
            return iterable

    def unsort(iterable):
        if len(iterable.shape) > 1:
            return iterable[reverse_idx][original_idx][reverse_idx]
        else:
            return iterable

    return sorted_lengths, sort, unsort


def debug_s2s_batch(outputs, labels, datasource):
    values, indices = outputs.contiguous().data.cpu().max(-1)
    _outputs = indices.data.cpu().numpy().tolist()
    _labels = labels.data.cpu().numpy().tolist()

    for src, trg in zip(_labels, _outputs):
        src = src[:src.index(0)]
        try:
            trg = trg[:trg.index(datasource.dataset.trg_vocab.tok2id["<eos>"])]
        except:
            pass
        print(" ".join([datasource.dataset.src_vocab.id2tok[x] for x in src]))
        print(" ".join([datasource.dataset.trg_vocab.id2tok[x] for x in trg]))
        print()


def save_checkpoint(name, model, optimizer, vocab, path=None, timestamp=False):
    """
    Save a trained model, along with its optimizer, in order to be able to
    resume training
    Args:
        name (str): the name of the model
        path (str): the directory, in which to save the checkpoints
        model ():
        optimizer ():
        timestamp (bool): whether to keep only one model (latest), or keep every
            checkpoint

    Returns:

    """
    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    if timestamp:
        model_fname = "{}_{}.pt".format(name, now)
    else:
        model_fname = "{}.pt".format(name)

    if path is None:
        path = os.path.join(BASE_PATH, "checkpoints")

    # save pytorch model
    torch.save([model, optimizer, vocab], os.path.join(path, model_fname))


def load_checkpoint(name, path=None):
    """
    Load a trained model, along with its optimizer
    Args:
        name (str): the name of the model
        path (str): the directory, in which the model is saved

    Returns:
        model, optimizer

    """
    if path is None:
        path = os.path.join(BASE_PATH, "checkpoints")

    model_fname = os.path.join(path, "{}.pt".format(name))

    with open(model_fname, 'rb') as f:
        model, optimizer, vocab = torch.load(f)

    return model, optimizer, vocab
