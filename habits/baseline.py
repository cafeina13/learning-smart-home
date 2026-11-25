"""The baseline every trained model must beat: hour-of-week frequency.

For each of the 168 hours of the week (Monday 00:00 ... Sunday 23:00) it learns
how often the appliance was switched on in that hour during training. It
predicts "will be used" when that frequency is above a threshold, and the
threshold is chosen on the validation weeks to maximise F1.

It knows nothing about recent events, only the clock. If a trained model
cannot beat this, the routine is purely a matter of time of day and the
baseline is the honest choice.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek.to_numpy() * 24 + index.hour.to_numpy()


class HourOfWeekBaseline:
    def __init__(self):
        self.freq = np.zeros(168)
        self.threshold = 0.5

    def fit(self, train: pd.Series) -> "HourOfWeekBaseline":
        how = hour_of_week(train.index)
        sums = np.bincount(how, weights=train.to_numpy(), minlength=168)
        counts = np.bincount(how, minlength=168)
        self.freq = np.divide(sums, counts, out=np.zeros(168), where=counts > 0)
        return self

    def scores(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.freq[hour_of_week(index)]

    def tune_threshold(self, val: pd.Series) -> "HourOfWeekBaseline":
        s = self.scores(val.index)
        candidates = np.unique(np.concatenate([s, [0.5]]))
        best = max(candidates, key=lambda th: f1_score(val, s >= th, zero_division=0))
        self.threshold = float(best)
        return self

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (self.scores(index) >= self.threshold).astype(int)


def evaluate(y_true, y_pred) -> dict:
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "positive_rate": float(np.mean(y_true)),
    }
