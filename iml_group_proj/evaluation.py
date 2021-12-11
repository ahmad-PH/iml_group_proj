import functools
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from iml_group_proj.features.common.config import RANDOM_STATE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def evaluate_many(models, X_train, y_train, X_test, y_test):
    results = []
    for (model, params, name) in models:
        clf = HalvingGridSearchCV(model, params, random_state=RANDOM_STATE, cv=1).fit(X_train, y_train)
        y_train_pred = clf.predict_proba(X_train)
        y_test_pred = clf.predict_proba(X_test)

        train_row = evaluate(y_train_pred, y_train, name)
        train_row["is_train"] = True
        train_row["params"] = clf.best_params_

        test_row = evaluate(y_test_pred, y_test, name)
        test_row["is_train"] = False
        test_row["params"] = clf.best_params_

        results.append(train_row)
        results.append(test_row)

    return pd.DataFrame(results)
    
def precision_recall_f1(label, prediction) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(label, np.argmax(prediction, axis=1), average="macro")

    return {
            "precision": p,
            "recall": r,
            "f1_score": f
    }

def accuracy(label, prediction) -> Dict[str, float]:
    return {
            "accuracuy": accuracy_score(label, np.argmax(prediction, axis=1))
            }

def evaluate(
        prediction,
        label,
        name: str = "Untitled",
        flows: List[Callable] = [accuracy, precision_recall_f1]
        ):

    flows = flows + [lambda a, b: {"name": name}]
    results = [f(label, prediction) for f in flows]

    return functools.reduce(
        lambda a,b: {**a, **b}, # Adding two dicts together.
        results
        )


