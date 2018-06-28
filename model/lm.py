import datetime
import math
import os
import numpy as np
from pyrsos.logger.experiment import Experiment, Metric
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import BASE_PATH, DEVICE, DATA_DIR
from model.params import WASSA_CONF, ConfLangModel
from model.pipelines import train_clf, eval_clf, train_sent_lm, eval_sent_lm
from modules.neural.dataloading import WordDataset, LangModelDataset
from modules.neural.models import Classifier, LangModel
from utils.dataloaders import load_wassa, sentence_dataset
from utils.early_stopping import Early_stopping
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocessor
from utils.training import class_weigths, epoch_summary, save_checkpoint

# load dataset
config = ConfLangModel
dataset = 'emotion2M'
name = 'emotion_with_2M'

data = sentence_dataset(os.path.join(DATA_DIR, dataset, "emotion_final.txt"))
y = np.zeros(len(data))

train_data, val_data, _, _ = train_test_split(data, y,
                                              test_size=0.2,
                                              random_state=13)
# train_data = train_data[:1000]
# val_data = val_data[:100]
#####################################################################
# Define Dataloaders
#####################################################################

# Prosoxh! to emotion dataset einai hdh PREPROCESSED me ekphrasis!

# preprocessor = twitter_preprocessor()
preprocessor = None
if preprocessor is None:
    train_name = "train_simple_split_{}".format(dataset)
    val_name = "valid_simple_split_{}".format(dataset)
else:
    train_name = "train_ekphrasis_{}".format(dataset)
    val_name = "valid_ekphrasis_{}".format(dataset)

train_set = LangModelDataset(train_data, name=train_name,
                             max_length=config["max_length"],
                             vocab_size=50000, preprocess=preprocessor)
val_set = LangModelDataset(val_data, name=val_name,
                           max_length=train_set.max_length,
                           vocab=train_set.vocab, preprocess=preprocessor)

train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_set, config["batch_eval"])

####################################################################
# Training Pipeline
####################################################################
ntokens = len(train_set.vocab)
print("Vocab:", ntokens)
print("Datasets: train={}, val={}".format(len(train_set), len(val_set)))

model = LangModel(ntokens, **config).to(DEVICE)
print(model)

loss_function = CrossEntropyLoss(ignore_index=0)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(parameters, amsgrad=True)
scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="loss_lm_" + name, tags=["train", "val"],
                             vis_type="line"))
experiment.add_metric(Metric(name="ppl_lm_" + name, tags=["train", "val"],
                             vis_type="line"))
early_stopping = Early_stopping("min", config["patience"]) # metric = val_loss

best_loss = None

now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

for epoch in range(config["epochs"]):

    scheduler.step()

    avg_loss = train_sent_lm(epoch, model, train_loader, ntokens,
                             loss_function, config["batch_train"], optimizer,
                             DEVICE, config["clip"])
    avg_val_loss = eval_sent_lm(model, val_loader, ntokens, loss_function,
                                DEVICE)

    lr = scheduler.optimizer.param_groups[0]['lr']
    print("\tLR:{}".format(lr))

    #############################################
    # Early Stopping
    #############################################
    if early_stopping.stop(avg_val_loss):
        print("Early Stopping....")
        break

    experiment.metrics["loss_lm_" + name].append(tag="train", value=avg_loss)
    experiment.metrics["ppl_lm_" + name].append(tag="train", value=math.exp(avg_loss))

    experiment.metrics["loss_lm_" + name].append(tag="val", value=avg_val_loss)
    experiment.metrics["ppl_lm_" + name].append(tag="val", value=math.exp(avg_val_loss))

    ############################################################
    # epoch summary
    ############################################################
    epoch_summary("train", avg_loss)
    epoch_summary("val", avg_val_loss)

    # after updating all the values, refresh the plots
    experiment.update_plots()

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or avg_val_loss < best_loss:
        print("saving checkpoint...")
        save_checkpoint("{}_{}".format(name, now), model, optimizer,
                        train_set.vocab,
                        loss=avg_val_loss, timestamp=False)
        best_loss = avg_val_loss

    print()
