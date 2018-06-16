from pprint import pprint

from modules.sklearn.models import bow_model, eval_clf
from utils.dataloaders import load_wassa
from utils.nlp import twitter_preprocess

bow_clf = bow_model("clf")

preprocessor = twitter_preprocess()

X_train, X_test, y_train, y_test = load_wassa()

X_train = preprocessor("wassa_train", X_train)
X_test = preprocessor("wassa_test", X_test)

bow_clf.fit(X_train, y_train)

y_pred = bow_clf.predict(X_test)
pprint(eval_clf(y_pred, y_test))
