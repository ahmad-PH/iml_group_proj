import time
import pandas as pd
from sklearn import preprocessing
from iml_group_proj.features.bert import get_BERT_features

from iml_group_proj.evaluation import evaluate_many
from iml_group_proj.features.common.data import load
from iml_group_proj.models.mlp import BASIC_MLP
from iml_group_proj.train_models import train_models


classes, train, test = load()
train = train.sample(frac=0.25)
X_train, X_test = get_BERT_features(train, test)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train["class"])
y_test = le.transform(test["class"])

# Training Phase
models = [
        (BASIC_MLP, 'MLP_100'),
        # (MLPClassifier(random_state=1, max_iter=10, hidden_layer_sizes=(100, 100)), 'MLP_10'),
        # (MLPClassifier(random_state=1, max_iter=150, hidden_layer_sizes=(100, 100)), 'MLP_150'),
        ]

# Hacky way to remove the sypnosis
trained_models_1 = train_models(models, X_train[:, :X_train.shape[1]//2], y_train, {"sypnosis": False})
result_df_1 = evaluate_many(trained_models_1, X_train[:, :X_train.shape[1]//2], y_train, X_test[:, :X_test.shape[1]//2], y_test)

trained_models_2 = train_models(models, X_train, y_train, {"sypnosis": True})
result_df_2 = evaluate_many(trained_models_2, X_train, y_train, X_test, y_test)

# Full Evaluation
result_df_full = pd.concat([result_df_1, result_df_2], axis=0)

print(result_df_full.head(20))
result_df_full.to_csv(f'_output/bert_result_{time.strftime("%Y%m%d-%H%M%S")}.csv')
