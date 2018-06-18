import os

import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import DATA_DIR, BASE_PATH, DEVICE
from model.params import SEMEVAL_2017
from model.pipelines import train_clf, eval_clf
from modules.neural.dataloading import WordDataset
from modules.neural.models import Classifier
from utils.dataloaders import load_data_from_dir
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, save_checkpoint

# load embeddings
file = os.path.join(BASE_PATH, "embeddings", "ntua_twitter_300.txt")
word2idx, idx2word, weights = load_word_vectors(file, 300)

# load dataset
config = SEMEVAL_2017
train = load_data_from_dir(os.path.join(DATA_DIR, 'semeval_2017_4A'))
X = [obs[1] for obs in train]
y = [obs[0] for obs in train]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=0)

# 3 - convert labels from strings to integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

#####################################################################
# Define Dataloaders
#####################################################################
preprocessor = twitter_preprocessor()
train_set = WordDataset(X_train, y_train, word2idx, name="sent_train",
                        preprocess=preprocessor)
test_set = WordDataset(X_test, y_test, word2idx, name="sent_test",
                       preprocess=preprocessor)
train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_set, config["batch_eval"])

classes = label_encoder.classes_.size
model = Classifier(embeddings=weights, out_size=classes, **config).to(DEVICE)
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
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")
        save_checkpoint("{}_{}".format("sentiment", now),
                        model, optimizer, timestamp=False)
        best_loss = avg_val_loss
