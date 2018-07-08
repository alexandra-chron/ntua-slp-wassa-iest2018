import os
import pickle

import datetime

import torch
from pyrsos.logger.experiment import Experiment, Metric
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import BASE_PATH, DEVICE
from model.params import WASSA_WITH_PRETR_LM, ConfLangModel
from model.pipelines import train_clf, eval_clf
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier, Classifier_extralinear
from utils.dataloaders import load_wassa
from utils.early_stopping import Early_stopping
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, load_checkpoint, epoch_summary, save_checkpoint_pre_lm, \
    load_checkpoint_pre_lm, load_checkpoint_with_f1

################################################################################
# pretr_model, pretr_optimizer, pretr_vocab, loss, acc, f1 = \
#     load_checkpoint_with_f1("wassa_baseline!_18-06-25_15:00:10")

extral = False

discr = False
d = 0.6

unfreeze = True
freeze = {"embed": True,
          "hidden": True}

unfreeze_epoque = {"embed": 6,
                   "hidden": 4}

# at which epoch the fine-tuning starts
name = "wassa_LM_FT_2M_GU24_noconc_GU46_discr"

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
    load_checkpoint("LM_FT_2M_GU_2_4_18-07-02_13:05:51")
pretr_model.to(DEVICE)

# # Force target task to use pretrained vocab
word2idx = pretr_vocab.tok2id
idx2word = pretr_vocab.id2tok
ntokens = pretr_vocab.size

# Bug: need to rename fu!#%ing triggerword
# because in the vocab built in pretrained LM we didn't name it correctly

# word2idx['<triggerword>'] = word2idx.pop('[#triggerword#]')
# idx2word[4] = '<triggerword>'

#####################################################################
# Define Dataloaders
#####################################################################

preprocessor = twitter_preprocessor()
# preprocessor = None
if preprocessor is None:
    train_name = "train_simple_split_{}".format(name)
    val_name = "valid_simple_split_{}".format(name)
else:
    train_name = "train_ekphrasis_{}".format(name)
    val_name = "valid_ekphrasis_{}".format(name)

train_set = WordDataset(X_train, y_train, word2idx, name=train_name,
                        preprocess=preprocessor)
test_set = WordDataset(X_test, y_test, word2idx, name=val_name,
                       preprocess=preprocessor)

train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_set, config["batch_eval"])

classes = label_encoder.classes_.size

# Define model, without pretrained embeddings
if extral:
    model = Classifier_extralinear(embeddings=weights, out_size=classes,
                                   concat_repr=False, **config).to(DEVICE)
else:
    model = Classifier(embeddings=weights, out_size=classes,
                       concat_repr=False, **config).to(DEVICE)

#############################################################################
# Transfer Learning (target takes source weights,except for linear layer)
#############################################################################
model.embedding = pretr_model.embedding
model.encoder = pretr_model.encoder
# model.embedding.noise.mean = 0.2
# model.encoder.drop_rnn.p = 0.3

model.embedding.noise.mean = 0.2
model.embedding.dropout.p = 0.4
model.encoder.drop_rnn.p = 0.4
#############################################################################
# Fine tune either: No layer, only embedding layer, all layers
#############################################################################

if freeze["embed"]:
    for param in model.embedding.parameters():
        param.requires_grad = False
if freeze["hidden"]:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.attention.parameters():
        param.requires_grad = False

print(model)

weights = class_weigths(train_set.labels, to_pytorch=True)
weights = weights.to(DEVICE)
criterion = CrossEntropyLoss(weight=weights)
parameters = filter(lambda p: p.requires_grad is True, model.parameters())

if discr:
    #############################################################################
    # Discriminative Fine-Tuning (smaller lr as the layer becomes more general)
    #############################################################################
    embed_params = filter(lambda p: p.requires_grad is True, model.embedding.parameters())
    encoder_params = filter(lambda p: p.requires_grad is True, model.encoder.parameters())
    attention_params = filter(lambda p: p.requires_grad is True, model.attention.parameters())
    output_params = filter(lambda p: p.requires_grad is True, model.output.parameters())
    if extral:
        extralinear_params = filter(lambda p: p.requires_grad is True, model.extralinear.parameters())

        optimizer = Adam([
                        # {'params': embed_params, 'lr': 1e-3*0.1*0.1},
                        # {'params': encoder_params, 'lr': 1e-3*0.1},
                        # {'params': attention_params, 'lr': 1e-3*0.1},
                        {'params': output_params, 'lr': 0.00146},
                        {'params': extralinear_params, 'lr': 0.00146}
                        ], lr=0.00146, amsgrad=True)
    else:
        optimizer = Adam([
                        # {'params': embed_params, 'lr': 1e-3*0.1*0.1},
                        # {'params': encoder_params, 'lr': 1e-3*0.1},
                        # {'params': attention_params, 'lr': 1e-3*0.1},
                        {'params': output_params, 'lr': 0.00146}
                        ], lr=0.00146, amsgrad=True)

else:
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


def unfreeze_module(module, optimizer, lr):
    for param in module.parameters():
        param.requires_grad = True
    my_dict = {'params':
               module.parameters(), "lr": lr}
    optimizer.add_param_group(my_dict)


#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="f1_macro_" + name, tags=["train", "val"],
                             vis_type="line"))

experiment.add_metric(Metric(name="acc_" + name, tags=["train", "val"],
                             vis_type="line"))

experiment.add_metric(Metric(name="loss_" + name, tags=["train", "val"],
                             vis_type="line"))
best_loss = None
early_stopping = Early_stopping("min", config["patience"])

now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

for epoch in range(1, config["epochs"] + 1):

    # scheduler.step()

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
    if unfreeze:
        if epoch == unfreeze_epoque["embed"]:
            if discr:
                lr = 0.00146 * d * d
            else:
                lr = 1e-3
            print("Unfreezing embedding layer...")
            unfreeze_module(model.embedding, optimizer, lr)
        if epoch == unfreeze_epoque["hidden"]:
            print("Unfreezing encoder...")
            if discr:
                lr = 0.00146 * d
            else:
                lr = 1e-3
            unfreeze_module(model.encoder, optimizer, lr)
            unfreeze_module(model.attention, optimizer, lr)

    ###############################################################
    # Early Stopping
    ###############################################################
    if early_stopping.stop(avg_val_loss):
        print("Early Stopping....")
        break

    # lr = scheduler.optimizer.param_groups[0]['lr']
    # print("\tLR:{}".format(lr))

    experiment.metrics["f1_macro_" + name].append(tag="train", value=f1_macro_train)
    experiment.metrics["f1_macro_" + name].append(tag="val", value=f1_macro_val)

    experiment.metrics["loss_" + name].append(tag="train", value=avg_train_loss)
    experiment.metrics["loss_" + name].append(tag="val", value=avg_val_loss)

    experiment.metrics["acc_" + name].append(tag="train", value=acc_train)
    experiment.metrics["acc_" + name].append(tag="val", value=acc_val)

    epoch_summary("train", avg_train_loss)
    epoch_summary("val", avg_val_loss)

    # after updating all the values, refresh the plots
    experiment.update_plots()

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")

        save_checkpoint_pre_lm("{}_{}".format(name, now), model,
                               optimizer, word2idx=word2idx, idx2word=idx2word,
                               loss=avg_val_loss, acc=acc(y, y_pred), f1=f1_macro_val,
                               timestamp=False)
        best_loss = avg_val_loss
