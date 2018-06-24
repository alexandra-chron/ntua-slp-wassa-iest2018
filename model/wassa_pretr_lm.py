import os
import pickle

import datetime

import torch
from pyrsos.logger.experiment import Experiment, Metric
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import BASE_PATH, DEVICE
from model.params import WASSA_WITH_PRETR_LM, ConfLangModel
from model.pipelines import train_clf, eval_clf
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import load_wassa
from utils.early_stopping import Early_stopping
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, load_checkpoint, epoch_summary, save_checkpoint_pre_lm, load_checkpoint_pre_lm

################################################################################
pretr_model, pretr_optimizer, pretr_vocab, loss, acc = load_checkpoint("wassa_pretr_clf_18-06-20_01:09:06")

finetune = "all"
# finetune = {None, embed, all}

unfreeze = 10
# at which epoch the fine-tuning starts

file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
_, _, weights = load_word_vectors(file, 300)

# load dataset
config = WASSA_WITH_PRETR_LM
config_lm = ConfLangModel

# Attention size needs to be equal to RNN size for Transfer Learning
if config['encoder_size'] != config_lm['rnn_size']:
    config['encoder_size'] = config_lm['rnn_size']
    print("Classifier RNN size needs to be equal to LM RNN size!")

X_train, X_test, y_train, y_test = load_wassa()

# 3 - convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Load Pretrained LM
pretr_model, pretr_optimizer, pretr_vocab, loss, acc = \
    load_checkpoint("twitter700K_18-06-20_23:59:37")
pretr_model.to(DEVICE)

# # Force target task to use pretrained vocab
word2idx = pretr_vocab.tok2id
idx2word = pretr_vocab.id2tok
ntokens = pretr_vocab.size

#####################################################################
# Define Dataloaders
#####################################################################

preprocessor = twitter_preprocessor()
train_set = WordDataset(X_train, y_train, word2idx, name="wassa_train_pretr_lm",
                        preprocess=preprocessor)
test_set = WordDataset(X_test, y_test, word2idx, name="wassa_test_pretr_lm",
                       preprocess=preprocessor)
train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_set, config["batch_eval"])

classes = label_encoder.classes_.size

# Define model, without pretrained embeddings
model = Classifier(embeddings=weights, out_size=classes, **config).to(DEVICE)

#############################################################################
# Transfer Learning (target takes source weights,except for linear layer)
#############################################################################
model.embedding = pretr_model.embedding
model.encoder = pretr_model.encoder

#############################################################################
# Fine tune either: No layer, only embedding layer, all layers
#############################################################################
if finetune is None:
    for param in model.parameters():
        param.requires_grad = False
elif finetune == "embed":
    model.embedding.requires_grad = False

print(model)

weights = class_weigths(train_set.labels, to_pytorch=True)
weights = weights.to(DEVICE)
criterion = CrossEntropyLoss(weight=weights)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(parameters, amsgrad=True)


#############################################################################
# Training Pipeline
#############################################################################
def acc(y, y_hat):
    return accuracy_score(y, y_hat)


def f1_macro(y, y_hat):
    return f1_score(y, y_hat, average='macro')


def f1_micro(y, y_hat):
    return f1_score(y, y_hat, average='micro')


def unfreeze_module(module, optimizer):
    for _param in module.parameters():
        _param.requires_grad = True

    optimizer.add_param_group(
        {'params': list(
            module.parameters())}
    )


#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="f1_macro_with_pretr_lm", tags=["train", "val"],
                             vis_type="line"))

experiment.add_metric(Metric(name="acc_with_pretr_lm", tags=["train", "val"],
                             vis_type="line"))

experiment.add_metric(Metric(name="loss_with_pretr_lm", tags=["train", "val"],
                             vis_type="line"))
best_loss = None
early_stopping = Early_stopping("max", config["patience"])

now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

for epoch in range(1, config["epochs"] + 1):
    # train the model for one epoch
    train_clf(epoch, model, train_loader, criterion, optimizer, DEVICE)
    print()
    # evaluate the performance of the model, on both data sets
    avg_train_loss, y, y_pred = eval_clf(model, train_loader, criterion,
                                         DEVICE)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1_macro={:.4f}, "
          "f1_micro={:.4f}".format(avg_train_loss,
                                    acc(y, y_pred),
                                    f1_macro(y, y_pred),  f1_micro(y, y_pred)))
    acc_train = acc(y, y_pred)
    f1_macro_train = f1_macro(y, y_pred)

    avg_val_loss, y, y_pred = eval_clf(model, test_loader, criterion,
                                       DEVICE)
    print("\tVal: loss={:.4f}, acc={:.4f}, f1_macro={:.4f}, "
          "f1_micro={:.4f}".format(avg_val_loss,
                                    acc(y, y_pred),
                                    f1_macro(y, y_pred),  f1_micro(y, y_pred)))
    acc_val = acc(y, y_pred)
    f1_macro_val = f1_macro(y, y_pred)

    ###############################################################
    # Unfreezing the model after X epochs
    ###############################################################
    if unfreeze > 0:
        if epoch == unfreeze:
            print("Unfreeze transfer-learning model...")
            unfreeze_module(model.encoder, optimizer)
            unfreeze_module(model.attention, optimizer)

    ###############################################################
    # Early Stopping
    ###############################################################
    if early_stopping.stop(f1_macro_val):
        print("Early Stopping....")
        break

    experiment.metrics["f1_macro_with_pretr_lm"].append(tag="train", value=f1_macro_train)
    experiment.metrics["f1_macro_with_pretr_lm"].append(tag="val", value=f1_macro_val)

    experiment.metrics["loss_with_pretr_lm"].append(tag="train", value=avg_train_loss)
    experiment.metrics["loss_with_pretr_lm"].append(tag="val", value=avg_val_loss)

    experiment.metrics["acc_with_pretr_lm"].append(tag="train", value=acc_train)
    experiment.metrics["acc_with_pretr_lm"].append(tag="val", value=acc_val)

    epoch_summary("train", avg_train_loss)
    epoch_summary("val", avg_val_loss)

    # after updating all the values, refresh the plots
    experiment.update_plots()

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")

        save_checkpoint_pre_lm("{}_{}".format("wassa_pretr_lm", now), model,
                               optimizer, word2idx=word2idx, idx2word=idx2word,
                               loss=avg_val_loss, acc=acc(y, y_pred),
                               timestamp=False)
        best_loss = avg_val_loss
