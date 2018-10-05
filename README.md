# Overview
This repository contains the source code of the models submitted by NTUA-SLP team in IEST of WASSA 2018 at EMNLP 2018. 
The model is described in the paper: https://arxiv.org/abs/1809.00717

Citation:
```
@article{chronopoulou2018ntua,
  title={NTUA-SLP at IEST 2018: Ensemble of Neural Transfer Methods for Implicit Emotion Classification},
  author={Chronopoulou, Alexandra and Margatina, Aikaterini and Baziotis, Christos and Potamianos, Alexandros},
  journal={arXiv preprint arXiv:1809.00717},
  year={2018}
}
```

# Quick Notes
There are 3 options for Transfer Learning in this model:

**First: Pretrain a LM and transfer its weights to the target-task classifier**

1) Pretrain the LM using ```models/lm.py```
2) Fine-tune the LM on your own dataset using ```models/lm_ft.py```
3) Train the classifier using ```wassa_pretr_lm.py``` (which transfers the weights of the pretrained LM to a classifier and adds Self-Attention and a task-specific linear layer)

**Second: Pretrain a classifier on a different dataset and transfer its weights to the target-task classifier**

1) Pretrain a classifier using ```models/sentiment.py```
2) Train the final classifier by using ```wassa.py``` and setting ```pretrained_classifier = True``` and providing the correspondent config file.

**Third: Use pretrained word vectors and transfer their weights to the embedding layer of a classifier**
- To do this, simply run ```wassa.py``` and make sure to provide the correspondent word2idx, idx2word and weights of the pretrained word vectors (word2vec, GloVe, fastText).
# Documentation

In order to make our codebase more accessible and easier to extend, we provide an overview of the structure of our project. 

`datasets` : contains the datasets for the pretraining :
- ```twitter100K/``` contains unlabeled data used for pretraining an LM
- ```semeval2017A/``` and ```wassa_2018/``` contain the labeled datasets used for SemEval17 Task4A and WASSA IEST 2018 respectively

`embeddings`: the pretrained word vectors used should be put here (i.e. word2vec, GloVe).

`model`: scripts for running:
- IEST classifier ```wassa.py```
- SE17 Task4 classifier ```sentiment.py```
- language model ```lm.py```.

`modules`: the source code of the PyTorch deep-learning models and the baseline models.

`submissions`: contains the script to test trained model and create submission file for WASSA.

`utils`: contains helper functions.

**Bibliography**

A few relevant and very important papers to our work are presented below:

https://arxiv.org/abs/1801.06146

http://arxiv.org/abs/1708.02182

http://arxiv.org/abs/1708.00524
