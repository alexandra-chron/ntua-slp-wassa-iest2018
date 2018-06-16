import time

import numpy
import torch
from torch.nn.utils import clip_grad_norm_

from utils.training import sort_by_lengths, progress


def process_batch(model, batch, device, criterion):
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

    # Loss calculation and backpropagation
    loss = criterion(outputs, labels)

    return loss, outputs, labels


def train_clf(epoch, model, data_source, criterion,
              optimizer, device, clip=1, log_interval=5):
    model.train()
    running_loss = 0.0

    start_time = time.time()

    for i_batch, batch in enumerate(data_source, 1):
        optimizer.zero_grad()

        loss, _, _ = process_batch(model, batch, device, criterion)

        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()

        progress(loss=loss.item(),
                 epoch=epoch,
                 batch=i_batch,
                 batch_size=data_source.batch_size,
                 dataset_size=len(data_source.dataset),
                 interval=log_interval,
                 start=start_time)

    return running_loss / i_batch


def eval_clf(model, data_source, criterion, device):
    model.eval()
    total_loss = 0

    posteriors = []
    labels = []

    with torch.no_grad():
        for i_batch, batch in enumerate(data_source, 1):
            loss, posts, y = process_batch(model, batch, device, criterion)
            total_loss += loss.item()

            posteriors.append(posts)
            labels.append(y)

    posteriors = torch.cat(posteriors, dim=0)
    predicted = numpy.argmax(posteriors, 1)
    labels = numpy.array(torch.cat(labels, dim=0))

    return total_loss / i_batch, labels, predicted
