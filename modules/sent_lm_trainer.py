import torch
from torch.nn.utils import clip_grad_norm_

from utils.training import sort_by_lengths, progress


def process_batch(model, batch, device, criterion, ntokens):
    # read batch
    batch = list(map(lambda x: x.to(device), batch))
    inputs, targets, lengths = batch

    # sort batch
    lengths, sort, unsort = sort_by_lengths(lengths)
    inputs = sort(inputs)

    # feed data to model
    outputs, _ = model(inputs, None, lengths)

    # Loss calculation and backpropagation
    loss = criterion(outputs.contiguous().view(-1, ntokens),
                     targets.contiguous().view(-1))

    return loss


def train_sent_lm(epoch, model, data_source, ntokens, criterion, batch_size,
                  optimizer, device, clip=1, log_interval=5):
    model.train()
    running_loss = 0.0

    for i_batch, batch in enumerate(data_source, 1):
        optimizer.zero_grad()

        loss = process_batch(model, batch, device, criterion, ntokens)

        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()

        progress(loss=loss.item(),
                 epoch=epoch,
                 batch=i_batch,
                 batch_size=batch_size,
                 dataset_size=len(data_source.dataset),
                 interval=log_interval)

    return running_loss / i_batch


def eval_sent_lm(model, data_source, ntokens, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i_batch, batch in enumerate(data_source, 1):
            loss = process_batch(model, batch, device, criterion, ntokens)
            total_loss += loss.item()

    return total_loss / i_batch
