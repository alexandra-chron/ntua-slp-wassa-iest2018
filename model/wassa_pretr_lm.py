import os
import pickle

import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import BASE_PATH, DEVICE
from model.params import WASSA_WITH_PRETR_LM
from model.pipelines import train_clf, eval_clf
from modules.neural.attention import SelfAttention
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import load_wassa
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, save_checkpoint, load_checkpoint

pretrained = True
# len(word2idx) = 804870
file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
_, _, weights = load_word_vectors(file, 300)

# load dataset
config = WASSA_WITH_PRETR_LM
X_train, X_test, y_train, y_test = load_wassa()

# 3 - convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Load Pretrained LM
pretr_model, pretr_optimizer, pretr_vocab = load_checkpoint("twitter100K_18-06-17_18:22:44")
pretr_model.to(DEVICE)

# Force target task to use pretrained vocab
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

# Transfer Learning, force target task to inherit parent task weights
# except for linear layer (pretrained was LM, target is 6 classes classifier)


model.embedding = pretr_model.embedding
model.encoder = pretr_model.encoder
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


def f1(y, y_hat):
    return f1_score(y, y_hat, average='macro')


best_loss = None
now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

for epoch in range(1, config["epochs"] + 1):
    # train the model for one epoch
    train_clf(epoch, model, train_loader, criterion, optimizer, DEVICE)
    print()
    # evaluate the performance of the model, on both data sets
    avg_train_loss, y, y_pred = eval_clf(model, train_loader, criterion,
                                         DEVICE)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_train_loss,
                                                               acc(y, y_pred),
                                                               f1(y, y_pred)))

    avg_val_loss, y, y_pred = eval_clf(model, test_loader, criterion,
                                       DEVICE)
    print("\tTest:  loss={:.4f}, acc={:.4f}, f1={:.4f}".format(avg_val_loss,
                                                               acc(y, y_pred),
                                                               f1(y, y_pred)))
    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")
        save_checkpoint("{}_{}".format("wassa", now), model, optimizer, loss=avg_val_loss, acc=acc(y, y_pred),
                        timestamp=False)
        best_loss = avg_val_loss
