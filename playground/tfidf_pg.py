import torch
import pandas as pd
from sklearn import preprocessing
from iml_group_proj.features.simple import get_simple_features

from iml_group_proj.evaluation import evaluate_many
from iml_group_proj.train_models import train_models
from iml_group_proj.features.common.data import load

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

IS_BOW = False # Bag-of-words or TfIdf

classes, train, test = load()
train = train.sample(frac=0.5)
X_train, X_test = get_simple_features(train, test, bow=IS_BOW, only_title=False)
X_train = X_train.toarray()
X_test = X_test.toarray()

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test = le.transform(test["class"])

# Training Phase
models = [
        # (MLPClassifier(random_state=1, max_iter=100, verbose=True), {"hidden_layer_sizes":[(100, 100), (200, 200)]}, 'MLP_100'),
        (MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=(100, 100), verbose=True, early_stopping=True), None, 'MLP_100'),
        # (GaussianNB(), {},'NaiveBayes'),
        # (SVC(), {"gamma": ["auto"], "kernel": ["rbf", "sigmoid", "poly"]},'SVM'),
        # (SVC, {"gamma": "auto", "kernel": "rbf"},'SVM'),
        ]

trained_models = train_models(models, X_train, y_train)

# Hacky way to remove the sypnosis
print("Bag-of-words" if IS_BOW else "Tf-idf")
result_df = evaluate_many(trained_models, X_train, y_train, X_test, y_test)
print(result_df.head(20))
