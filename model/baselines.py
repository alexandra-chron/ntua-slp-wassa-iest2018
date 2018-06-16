import os

from config import BASE_PATH
from modules.sklearn.models import eval_clf, nbow_model, bow_model
from utils.dataloaders import load_wassa
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocess

#############################################################
# Load Data
#############################################################
preprocessor = twitter_preprocess()

X_train, X_test, y_train, y_test = load_wassa()

# X_train = X_train[:1000]
# y_train = y_train[:1000]
# X_test = X_test[:1000]
# y_test = y_test[:1000]

X_train = preprocessor("wassa_train", X_train)
X_test = preprocessor("wassa_test", X_test)

#############################################################
# Bag-of-Words
# #############################################################
bow_clf = bow_model("clf", max_features=30000)
bow_clf.fit(X_train, y_train)
y_pred = bow_clf.predict(X_test)
bow_results = eval_clf(y_pred, y_test)

print("\n" + "#" * 40)
print("Bag-of-Words")
print("#" * 40)
for k, v in bow_results.items():
    print("{}:{:.4f}".format(k, v))

#############################################################
# Neural Bag-of-Words
#############################################################

file = os.path.join(BASE_PATH, "embeddings", "word2vec_300_6_20_neg.txt")
word2idx, idx2word, weights = load_word_vectors(file, 300)

nbow_clf = nbow_model("clf", weights, word2idx)
nbow_clf.fit(X_train, y_train)
y_pred = nbow_clf.predict(X_test)
nbow_results = eval_clf(y_pred, y_test)

print("\n" + "#" * 40)
print("Neural Bag-of-Words")
print("#" * 40)
for k, v in nbow_results.items():
    print("{}:{:.4f}".format(k, v))
