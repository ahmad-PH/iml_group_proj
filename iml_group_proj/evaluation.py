import functools
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def evaluate_many(models_with_names, X_train, y_train, X_test, y_test):
    results = []
    for (model, name) in models_with_names:
        clf = model.fit(X_train, y_train)
        y_train_pred = clf.predict_proba(X_train)
        y_test_pred = clf.predict_proba(X_test)

        train_row = evaluate(y_train_pred, y_train, name+"__train")
        test_row = evaluate(y_test_pred, y_test, name+"__test")

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


