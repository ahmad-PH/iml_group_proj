import torch
import pandas as pd
from sklearn import preprocessing
from iml_group_proj.features.bert import get_BERT_features

from iml_group_proj.evaluation import evaluate_many
from iml_group_proj.features.common.data import load

classes, train, test = load()
train = train.sample(frac=0.25)
X_train, X_test = get_BERT_features(train, test)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test = le.transform(test["class"])

# Training Phase
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
models = [
        # (MLPClassifier(random_state=1, max_iter=10, hidden_layer_sizes=(100, 100)), 'MLP_10'),
        (MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=(100, 100)), 'MLP_100'),
        # (MLPClassifier(random_state=1, max_iter=150, hidden_layer_sizes=(100, 100)), 'MLP_150'),
        ]

# Hacky way to remove the sypnosis
result_df = evaluate_many(models, X_train[:, :X_train.shape[1]//2], y_train, X_test[:, :X_test.shape[1]//2], y_test)
result_df2 = evaluate_many(models, X_train, y_train, X_test, y_test)
result_df2["name"] = result_df2["name"].apply(lambda x: x +"__with_sypnosis")

result_df_full = pd.concat([result_df, result_df2], axis=0)

print(result_df_full.head(20))

result_df_full["params"] = str({"hidden_layer_sizes":(100, 100), "max_iter": 100})
result_df_full.to_csv("_output/bert_result.csv")
