# Overview
This repository contains the source code of the models submitted by NTUA-SLP team in IEST of WASSA 2018 at EMNLP 2018. 
The model is described in the paper: http://aclweb.org/anthology/W18-6209

Citation:
```
@InProceedings{W18-6209,
  author = 	"Chronopoulou, Alexandra
		and Margatina, Aikaterini
		and Baziotis, Christos
		and Potamianos, Alexandros",
  title = 	"NTUA-SLP at IEST 2018: Ensemble of Neural Transfer Methods for Implicit Emotion Classification",
  booktitle = 	"Proceedings of the 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"57--64",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/W18-6209"
}
```
# Implicit Emotion Classification Task
Task: Classify twitter messages in one of **six emotion categories** (happy, sad, fear, anger, surprise, disgust) **without** the emotion word. 
A typical tweet in this dataset has the following form:

 ```I'm \[#TARGETWORD#\] because I love you, I love you and I hate you.```  (correct label: **angry**) 

# Our approach
We use an ensemble of 3 different Transfer Learning approaches:

Before you proceed, ```pip install -r ./requirements.txt```

**First) Pretrain a LSTM-based language model (LM) and transfer it to a target-task classification model:**

<img src="https://github.com/alexandra-chron/ntua-slp-wassa-iest2018/blob/master/ulmfit.png" width="380">

```cd model/```

You can skip steps 1 and 2 (time-consuming) and use pretrained and fine-tuned LM, which should be put under  ```checkpoints/```

Download it here: [pretrained + fine-tuned LM](https://drive.google.com/file/d/1zrP5coIKlCyLSf-0rLYRog_WYJw9yT5e/view?usp=sharing)

1) (can be skipped) Pretrain the LM: ```python lm.py```
2) (can be skipped) Fine-tune the LM on your own (target) dataset: ```python lm_ft.py```

3) Train the classification model: ```python wassa_pretr_lm.py``` (initializes the weights of the embedding and hidden layer with the LM and adds a Self-Attention mechanism and a classification layer)

*This follows to a great degree ULMFiT by Howard and Ruder.*

**Second) Pretrain a LSTM-based attentive classification model on a different dataset and transfer its feature extractor to the target-task classification model:**

<img src="https://github.com/alexandra-chron/ntua-slp-wassa-iest2018/blob/master/pre_cls.png" width="300">


1) Pretrain a classifier: ```python sentiment.py```
2) Train the final classifier: ```python wassa.py``` (set ```pretrained_classifier = True``` and provide the correspondent config file.)

**Third) Use pretrained word vectors to initialize the embedding layer of a classification model:**
-  ```python wassa.py``` (set ```pretrained_classifier = False``` and provide the correspondent word2idx, idx2word and weights of the pretrained word vectors (word2vec, GloVe, fastText)).

# Quick Notes
Our pretrained word embeddings are available here: [ntua_twitter_300.txt](https://drive.google.com/file/d/1b-w7xf0d4zFmVoe9kipBHUwfoefFvU2t/view)

# Documentation

In order to make our codebase more accessible and easier to extend, we provide an overview of the structure of our project. 

`datasets` : contains the datasets for the pretraining :
- ```twitter100K/``` contains unlabeled data used for pretraining an LM
- ```semeval2017A/``` and ```wassa_2018/``` contain the labeled datasets used for SemEval17 Task4A and WASSA IEST 2018 respectively

`embeddings`: pretrained word2vec embeddings should be put here.


`model`: scripts for running:
- IEST classifier ```wassa.py```
- SE17 Task4 classifier ```sentiment.py```
- language model ```lm.py```.

`modules`: the source code of the PyTorch deep-learning models and the baseline models.

`submissions`: contains the script to test trained model and create submission file for WASSA.

`utils`: contains helper functions.

**Bibliography**

A few relevant and very important papers to our work are presented below:

```Universal Language Model Fine-tuning for Text Classification``` https://arxiv.org/abs/1801.06146

```Regularizing and Optimizing LSTM Language Models``` https://arxiv.org/abs/1708.02182

```Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm``` http://arxiv.org/abs/1708.00524
