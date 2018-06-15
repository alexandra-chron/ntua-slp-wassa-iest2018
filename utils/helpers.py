import math
import sys
import os
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle

def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def progress_lm(loss, epoch, batch, batch_size, dataset_size, interval):
    batches = math.ceil(float(dataset_size) / batch_size)

    if batch % interval != 0 and batch != batches:
        return

    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    log_bar = '=' * filled_len + '-' * (bar_len - filled_len)

    log_losses = ' Loss: {:.2f}, PPL: {:.2f}'.format(loss, math.exp(loss))
    log_epoch = ' Epoch {}'.format(epoch)
    log_batches = ' Batch {}/{}'.format(batch, batches)
    _progress_str = "\r \r [{}] {} - {} ... {}".format(log_bar,
                                                       log_epoch,
                                                       log_batches,
                                                       log_losses)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


#
# def progress_multitask(loss, epoch, batch, batch_size, dataset_size, interval,
#                        mode):
#     batches = math.ceil(float(dataset_size) / batch_size)
#
#     if batch % interval != 0 and batch != batches:
#         return
#
#     count = batch * batch_size
#     bar_len = 40
#     filled_len = int(round(bar_len * count / float(dataset_size)))
#
#     log_bar = '=' * filled_len + '-' * (bar_len - filled_len)
#
#     log_losses = ' CE: {:.4f}, PPL: {:.4f}'.format(loss, math.exp(loss))
#     log_epoch = ' Epoch {}'.format(epoch)
#     log_batches = ' Batch {}/{}'.format(batch, batches)
#     _progress_str = "\r \r {} [{}] ({}) {} ... {}".format(log_epoch,
#                                                           log_bar,
#                                                           mode,
#                                                           log_batches,
#                                                           log_losses, )
#     sys.stdout.write(_progress_str)
#     sys.stdout.flush()
#
#     if batch == batches:
#         print()


def epoch_summary_lm(dataset, loss):
    msg = "\t{:7s}: Avg Loss = {:.4f},\tAvg Perplexity = {:.4f}"
    print(msg.format(dataset, loss, math.exp(loss)))


def packed_targets(targets, lengths, batch_first=False):
    targets = pack_padded_sequence(targets, list(lengths),
                                   batch_first=batch_first)
    targets, lengths = pad_packed_sequence(targets, batch_first=batch_first)
    return targets, lengths

def file_cache_name(file):
    head, tail = os.path.split(file)
    filename, ext = os.path.splitext(tail)
    return os.path.join(head, filename + ".p")
