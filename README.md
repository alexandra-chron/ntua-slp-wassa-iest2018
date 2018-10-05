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

**Quick Notes**

If you just want to run the model (which is a slightly alternative version of ULMFiT for Twitter):
1) Pretrain the LM using ```models/lm.py```
2) Fine-tune the LM on your own dataset using ```models/lm_ft.py```
3) Train the classifier using ```wassa_pretr_lm.py``` (which transfers the weights of the pretrained LM to a classifier and adds Self-Attention and a task-specific linear layer)

# Documentation

In order to make our codebase more accessible and easier to extend, we provide an overview of the structure of our project. 

`datasets` : contains the datasets for the pretraining(2 options, one for LM and one for classifier training).

`embeddings`: the word embedding files used should be put here (i.e. word2vec).

`model`: scripts for running wassa classifier (wassa.py) SE17 Task4 classifier (sentiment.py), language model (lm.py).

`modules`: the source code of the PyTorch deep-learning models and the baseline models.

`submissions`: contains the script to test trained model and create submission file for WASSA.

`utils`: contains helper functions.

**Bibliography**

Some papers on which this repo is based:

https://arxiv.org/abs/1801.06146

http://arxiv.org/abs/1708.02182

http://arxiv.org/abs/1708.00524
