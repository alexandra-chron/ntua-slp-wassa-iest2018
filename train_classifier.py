"""
Model for sentiment classification (positive,negative,neutral)
for
"""
import os

import torch
from sklearn.model_selection import train_test_split

from utils.dataloaders import load_wassa
from config import DEVICE, DATA_DIR
from logger.training import LabelTransformer
from model.params import WASSA_2018
from modules.neural.models import LangModel, ModelWrapper
from utils.train import get_pretrained, load_embeddings, load_datasets

config = WASSA_2018
os.path.join(DATA_DIR, 'wassa_2018')

train = load_wassa(os.path.join(DATA_DIR, 'wassa_2018'))
X = [obs[1] for obs in train]
y = [obs[0] for obs in train]

X_train, X_dev, y_train, y_dev = train_test_split(X, y,
                                                    test_size=0.05,
                                                    stratify=y,
                                                    random_state=22)
datasets = {
    "train": (X_train, y_train),
    "dev": (X_dev, y_dev)
            }
label_map = {label: idx for idx, label in
             enumerate(sorted(list(set(y_train))))}
inv_label_map = {v: k for k, v in label_map.items()}
transformer = LabelTransformer(label_map, inv_label_map)

pretrained = None
pretrained_models = None
pretrained_config = None

if pretrained is not None:
    pretrained_models = get_pretrained(pretrained)

else:
    word2idx, idx2word, embeddings = load_embeddings(config)

########################################################################
# DATASET
# construct the pytorch Datasets and Dataloaders
########################################################################
loaders = load_datasets(datasets,
                        train_batch_size=config["batch_train"],
                        eval_batch_size=config["batch_eval"],
                        token_type=config["token_type"],
                        word2idx=word2idx,
                        label_transformer=transformer)

classes = len(set(loaders["train"].dataset.labels))
out_size = classes
num_embeddings = None
finetune = True

model = ModelWrapper(embeddings=embeddings,
                     out_size=out_size,
                     num_embeddings=num_embeddings,
                     pretrained=pretrained_models,
                     finetune=finetune,
                     **config)
model.to(DEVICE)
print(model)