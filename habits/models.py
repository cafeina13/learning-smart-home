"""Trained habit models, all scored exactly like the baseline.

Each model outputs a score per hour; the decision threshold is chosen on the
validation weeks to maximise F1 (the same rule the baseline uses), then fixed
and applied to the test weeks.

logistic regression   linear, features standardised, classes balanced
random forest         200 trees, leaves of at least 20 hours (to avoid
                      memorising rare hours), classes balanced per tree

Classes are balanced because most hours have no use (a washing machine starts
in ~2% of hours): without it, "never" would look like a good answer.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from habits.features import build_features

# scikit-learn's random forest triggers a harmless version-mismatch warning from its
# parallel helper on every fit; it does not affect results.
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")


def best_threshold(y_val, scores) -> float:
    """The score cut-off with the best F1 on the validation weeks."""
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    return float(max(candidates, key=lambda th: f1_score(y_val, scores >= th, zero_division=0)))


def make_model(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0)
    raise ValueError(name)


def fit_and_predict(name: str, task) -> tuple[np.ndarray, dict]:
    """Train on the task's train hours, tune the threshold on its validation hours,
    return 0/1 predictions for its test hours (and the fitted model's details)."""
    X = build_features(task.labels, task.other_starts)
    X_train, X_val, X_test = X.loc[task.train.index], X.loc[task.val.index], X.loc[task.test.index]
    if task.train.nunique() < 2:  # never (or always) used in training: nothing to learn
        return np.zeros(len(X_test), dtype=int), {}
    model = make_model(name).fit(X_train, task.train)
    threshold = best_threshold(task.val, model.predict_proba(X_val)[:, 1])
    predictions = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    details = {"threshold": threshold}
    if name == "random_forest":
        details["importances"] = dict(zip(X.columns, model.feature_importances_))
    return predictions, details
