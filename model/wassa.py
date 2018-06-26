import math
import os
import pickle

import datetime

from pyrsos.logger.experiment import Experiment, Metric
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import BASE_PATH, DEVICE
from model.params import WASSA_CONF, WASSA_BASELINE, SEMEVAL_2017
from model.pipelines import train_clf, eval_clf
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import load_wassa
from utils.early_stopping import Early_stopping
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, save_checkpoint, load_checkpoint, epoch_summary, save_checkpoint_with_f1, \
    load_checkpoint_with_f1

pretrained_classifier = True

# load embeddings
file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
word2idx, idx2word, weights = load_word_vectors(file, 300)

# load dataset
config = WASSA_CONF
config_sentiment = SEMEVAL_2017

if pretrained_classifier:
    if config['encoder_size'] != config_sentiment['encoder_size']:
        config['encoder_size'] = config_sentiment['encoder_size']
        config['encoder_bidirectional'] = config_sentiment['encoder_bidirectional']
        print("Classifier RNN size needs to be equal to Sentiment RNN size! (TL)")

X_train, X_test, y_train, y_test = load_wassa()

# 3 - convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
with open("../submissions/label_encoder.pkl", "wb") as r:
    pickle.dump(label_encoder, r)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

#####################################################################
# Define Dataloaders
#####################################################################
preprocessor = twitter_preprocessor()
# preprocessor = None

train_set = WordDataset(X_train, y_train, word2idx, name="wassa_train_35",
                        max_length=35,
                        preprocess=preprocessor)
test_set = WordDataset(X_test, y_test, word2idx, name="wassa_test_35",
                       max_length=35,
                       preprocess=preprocessor)
train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_set, config["batch_eval"])

classes = label_encoder.classes_.size
model = Classifier(embeddings=weights, out_size=classes, **config).to(DEVICE)

weights = class_weigths(train_set.labels, to_pytorch=True)
weights = weights.to(DEVICE)
criterion = CrossEntropyLoss(weight=weights)

if pretrained_classifier:
    pretr_model, pretr_optimizer, pretr_vocab, loss, acc = \
        load_checkpoint("sentiment_baseline")
    pretr_model.to(DEVICE)
    pretr_model.output = model.output
    model = pretr_model

    name = "wassa_pretr_clf_with_baseline"
else:
    name = "wassa_rnn_600_bidir_batch32_adam_0.05"

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(parameters, amsgrad=True)

print(model)

#############################################################################
# Training Pipeline
#############################################################################


def acc(y, y_hat):
    return accuracy_score(y, y_hat)


def f1_macro(y, y_hat):
    return f1_score(y, y_hat, average='macro')


def f1_micro(y, y_hat):
    return f1_score(y, y_hat, average='micro')


#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="f1_macro_" + name, tags=["train", "val"],
                             vis_type="line"))
experiment.add_metric(Metric(name="loss_" + name, tags=["train", "val"],
                             vis_type="line"))
experiment.add_metric(Metric(name="acc_" + name, tags=["train", "val"],
                             vis_type="line"))
best_loss = None
early_stopping = Early_stopping("min", config["patience"])

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

    #############################################
    # Early Stopping
    #############################################
    if early_stopping.stop(avg_val_loss):
        print("Early Stopping....")
        break
    experiment.metrics["f1_macro_" + name].append(tag="train", value=f1_macro_train)
    experiment.metrics["f1_macro_" + name].append(tag="val", value=f1_macro_val)

    experiment.metrics["loss_" + name].append(tag="train", value=avg_train_loss)
    experiment.metrics["loss_" + name].append(tag="val", value=avg_val_loss)

    experiment.metrics["acc_" + name].append(tag="train", value=acc_train)
    experiment.metrics["acc_" + name].append(tag="val", value=acc_val)

    # epoch_summary("train", avg_train_loss)
    # epoch_summary("val", avg_val_loss)

    # after updating all the values, refresh the plots
    experiment.update_plots()

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")
        save_checkpoint_with_f1("{}_{}".format(name, now), model,
                        optimizer, loss=avg_val_loss, acc=acc_val, f1=f1_macro_val,
                        timestamp=False)
        best_loss = avg_val_loss
