**Overview**

This repository contains the source code of the models submitted by NTUA-SLP team in WASSA 2018 
http://implicitemotions.wassa2018.com/

The results of the competition can be found here:
https://competitions.codalab.org/competitions/19214#results

**Documentation**

In order to make our codebase more accessible and easier to extend, we provide an overview of the structure of our project. 

`datasets` : contains the datasets for the pretraining(2 options, one for LM and one for classifier training)

`embeddings`: the word embedding files used should be put here (i.e. word2vec)

`model`: scripts for running wassa classifier (wassa.py) SE17 Task4 classifier (sentiment.py), language model (lm.py)

`modules`: the source code of the PyTorch deep-learning models and the baseline models.

`submissions`: contains the script to test trained model and create submission file for Wassa

`utils`: contains helper functions

**Bibliography**

Some papers on which this repo is based:

https://arxiv.org/abs/1801.06146

http://arxiv.org/abs/1708.02182

http://arxiv.org/abs/1708.00524
