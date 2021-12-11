from re import A
import torch
import pandas as pd
from sklearn import preprocessing
from iml_group_proj.features.simple import get_simple_features

from iml_group_proj.evaluation import evaluate_many
from iml_group_proj.features.common.data import load

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

IS_BOW = True # Bag-of-words or TfIdf

classes, train, test = load()
train = train.sample(frac=0.25)
X_train, X_test = get_simple_features(train, test, bow=IS_BOW)


le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test = le.transform(test["class"])

# Training Phase
models = [
        (SVC(gamma="auto", probability=True), 'SVM'),
        (GaussianNB(), 'NaiveBayes'),
        (MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=(100, 100)), 'MLP_100'),
        ]

# Hacky way to remove the sypnosis
print("Bag-of-words" if IS_BOW else "Tf-idf")
result_df = evaluate_many(models, X_train.to_array(), y_train, X_test.to_array(), y_test)
print(result_df.head(20))
