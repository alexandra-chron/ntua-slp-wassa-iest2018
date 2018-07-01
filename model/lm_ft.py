import math
from pyrsos.logger.experiment import Experiment, Metric
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import DEVICE
from model.params import ConfLangModelFT
from model.pipelines import train_sent_lm, eval_sent_lm
from modules.neural.dataloading import LangModelDataset
from utils.dataloaders import load_wassa
from utils.early_stopping import Early_stopping
from utils.nlp import twitter_preprocessor
from utils.training import epoch_summary, save_checkpoint, load_checkpoint

# load dataset

config = ConfLangModelFT
name = 'LM_FT_GU_3_6'
dataset = 'wassa'

unfreeze = True
freeze = {"embed": True,
          "hidden": True}

unfreeze_epoque = {"embed": 6,
                   "hidden": 3}

# Load Pretrained LM
pretr_model, pretr_optimizer, pretr_vocab, loss, acc = \
    load_checkpoint("emotion2M/emotion_with_2M_18-06-28_18:04:54")
pretr_model.to(DEVICE)

# Load wassa
train_data, val_data, _, _ = load_wassa()
#####################################################################
# Define Dataloaders
#####################################################################

preprocessor = twitter_preprocessor()
if preprocessor is None:
    train_name = "train_simple_split_{}".format(dataset)
    val_name = "valid_simple_split_{}".format(dataset)
else:
    train_name = "train_ekphrasis_{}".format(dataset)
    val_name = "valid_ekphrasis_{}".format(dataset)

word2idx = pretr_vocab.tok2id
idx2word = pretr_vocab.id2tok
word2idx['<triggerword>'] = word2idx.pop('[#triggerword#]')
idx2word[4] = '<triggerword>'

train_set = LangModelDataset(train_data, name=train_name,
                             max_length=config["max_length"],
                             vocab=pretr_vocab, preprocess=preprocessor)
val_set = LangModelDataset(val_data, name=val_name,
                           max_length=train_set.max_length,
                           vocab=pretr_vocab, preprocess=preprocessor)

train_loader = DataLoader(train_set, config["batch_train"], shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_set, config["batch_eval"])

####################################################################
# Training Pipeline
####################################################################
ntokens = len(train_set.vocab)
print("Vocab:", ntokens)
print("Datasets: train={}, val={}".format(len(train_set), len(val_set)))

#############################################################################
# Transfer Learning
#############################################################################
model = pretr_model
print(model)

#############################################################################
# Fine tune either: No layer, only embedding layer, all layers
#############################################################################

if freeze["embed"]:
    for param in model.embedding.parameters():
        param.requires_grad = False
if freeze["hidden"]:
    for param in model.encoder.parameters():
        param.requires_grad = False


loss_function = CrossEntropyLoss(ignore_index=0)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(parameters, amsgrad=True)
# scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)


def unfreeze_module(module, optimizer):
    for param in module.parameters():
        param.requires_grad = True
    my_dict = {'params': list(
               module.parameters())}
    optimizer.add_param_group(my_dict)


#############################################################
# Experiment
#############################################################
experiment = Experiment(config["name"], hparams=config)
experiment.add_metric(Metric(name="loss_lm_" + name, tags=["train", "val"],
                             vis_type="line"))
experiment.add_metric(Metric(name="ppl_lm_" + name, tags=["train", "val"],
                             vis_type="line"))
early_stopping = Early_stopping("min", config["patience"])  # metric = val_loss

best_loss = None

# now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

for epoch in range(config["epochs"]):

    # scheduler.step()

    avg_loss = train_sent_lm(epoch, model, train_loader, ntokens,
                             loss_function, config["batch_train"], optimizer,
                             DEVICE, config["clip"])
    avg_val_loss = eval_sent_lm(model, val_loader, ntokens, loss_function,
                                DEVICE)

    # lr = scheduler.optimizer.param_groups[0]['lr']
    # print("\tLR:{}".format(lr))

    ###############################################################
    # Unfreezing the model after X epochs
    ###############################################################
    if unfreeze:
        if epoch == unfreeze_epoque["embed"]:
            print("Unfreezing embedding layer...")
            unfreeze_module(model.embedding, optimizer)
        if epoch == unfreeze_epoque["hidden"]:
            print("Unfreezing encoder...")
            unfreeze_module(model.encoder, optimizer)

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
        save_checkpoint("{}".format(name), model, optimizer,
                        train_set.vocab,
                        loss=avg_val_loss, timestamp=True)
        best_loss = avg_val_loss

    print()
