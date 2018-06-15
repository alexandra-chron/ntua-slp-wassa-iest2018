"""
Model for sentiment classification (positive,negative,neutral)
for Semeval2017 TaskA
"""
import math
import os
import time

import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

# from dataloaders.get_files import load_data_from_dir
from logger.experiment import Experiment, Metric
from config import DEVICE, DATA_DIR
from model.params import SEMEVAL_2017
from modules.nn.dataloading import LangModelDataset
from modules.nn.models import LangModel
from modules.sent_lm_trainer import train_sent_lm, eval_sent_lm

from utils.training import save_checkpoint, epoch_summary

config = SEMEVAL_2017
os.path.join(DATA_DIR, 'semeval_2017_4A')

# train = load_data_from_dir(os.path.join(DATA_DIR, 'semeval_2017_4A'))
# X = [obs[1] for obs in train]
# y = [obs[0] for obs in train]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.05,
#                                                     stratify=y,
#                                                     random_state=12)
with open("preprocessedtrain.txt", 'rb') as f:
    X_train = torch.load(f)
with open("preprocessedtest.txt", 'rb') as f:
    X_test = torch.load(f)

X_train = [' '.join(x) for x in X_train]
X_test = [' '.join(x) for x in X_test]
# word2idx, idx2word, embeddings = load_embeddings(config)
####################################################################
# Data Loading and Preprocessing
####################################################################

def tokenize(text):
    return text.lower().split()

datasets={}

train_set = LangModelDataset(X_train, tokenize=tokenize,
                             seq_len=50, name='train_set',
                             stateful=False)

val_set = LangModelDataset(X_test, tokenize=tokenize,
                           seq_len=train_set.seq_len,
                           name='test_set',
                           vocab=train_set.vocab,
                           stateful=False)

# define a dataloader, which handles the way a dataset will be loaded,
# like batching, shuffling and so on ...
train_loader = DataLoader(train_set, batch_size=config["batch_train"],
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=config["batch_eval"],
                        shuffle=True, num_workers=4)
ntokens = len(train_set.vocab)

####################################################################
# Training Pipeline
####################################################################
print("Vocab:", ntokens)
print("Datasets: train={}, val={}".format(len(train_set), len(val_set)))

model = LangModel(ntokens, **config).to(DEVICE)
print(model)

########################################################################
# Loss function and optimizer
########################################################################

loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, amsgrad=True)
# scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5)
scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="loss", tags=["train", "val"],
                             vis_type="line"))
experiment.add_metric(Metric(name="ppl", tags=["train", "val"],
                             vis_type="line"))

best_loss = None


for epoch in range(config["epochs"]):
    start_time = time.time()

    scheduler.step()

    avg_loss = train_sent_lm(epoch, model, train_loader, ntokens,
                             loss_function, config["batch_train"], optimizer,
                             DEVICE, config["clip_norm"])
    avg_val_loss = eval_sent_lm(model, val_loader, ntokens, loss_function,
                                DEVICE)

    ############################################################
    # epoch summary
    ############################################################
    lr = scheduler.optimizer.param_groups[0]['lr']
    duration = (time.time() - start_time)
    print("\tTime:{:5.2f}s, LR:{}".format(duration, lr))

    experiment.metrics["loss"].append(tag="train", value=avg_loss)
    experiment.metrics["ppl"].append(tag="train", value=math.exp(avg_loss))

    experiment.metrics["loss"].append(tag="val", value=avg_val_loss)
    experiment.metrics["ppl"].append(tag="val", value=math.exp(avg_val_loss))

    epoch_summary("train", avg_loss)
    epoch_summary("val", avg_val_loss)

    # after updating all the values, refresh the plots
    experiment.update_plots()

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")
        save_checkpoint(config["name"], model, optimizer)
        best_loss = avg_val_loss

    print()
