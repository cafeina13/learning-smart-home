"""The paper's forecasters, its feature selection and its hourly ensemble (scikit-learn versions).

Forecasters (Yaprakdal 2023, Section 2), each predicting the next hour's load from
the flattened 144 x features window:

    MLR    multiple linear regression                                      Model1
    ANN1   one hidden layer of 12 units, tanh, linear output                Model1
    ANN2   five hidden layers (50, 40, 30, 20, 10), tanh, linear output     Model1
    SVR    nu-SVR, cost 0.08, nu 0.5 (RBF kernel; the paper does not say)   Model2
    KNN    k = 1, Euclidean                                                 Model2
    Tree   regression tree; leaf size tuned by inner cross-validation
           (the paper used Bayesian optimisation)                           Model2

Model1: one forecaster for every hour. Model2: one forecaster per target hour of
the day (24), as the paper does for SVR, KNN and the tree. The paper's CNNs and
BiLSTM need PyTorch and are added separately.

The neural networks learn a z-scored target (the paper z-scores everything but the
calendar); predictions are returned in watts.

Feature selection (paper, Algorithm 1): starting from all features, repeatedly drop
the feature whose removal gives the lowest ANN validation MAE, until 6 remain;
then try every non-empty subset of those 6 (63) and keep the best.

Hourly ensemble (the paper's contribution): for each hour of the day, non-negative
weights that sum to 1 over the forecasters, fitted on the validation predictions.
"""

import itertools
import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor

from energy.data import Scaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FORECASTERS = ["MLR", "ANN1", "ANN2", "SVR", "KNN", "Tree"]
DEEP = ["CNN1", "CNN2", "RNN1", "RNN2", "GRU", "CNN-GRU"]  # energy/deep.py (PyTorch)
PER_HOUR = {"SVR", "KNN", "Tree"}  # the paper's Model2


def _make(name: str, seed: int = 0):
    if name == "MLR":
        return LinearRegression()
    if name == "ANN1":
        return MLPRegressor(hidden_layer_sizes=(12,), activation="tanh", early_stopping=True,
                            max_iter=500, random_state=seed)
    if name == "ANN2":
        return MLPRegressor(hidden_layer_sizes=(50, 40, 30, 20, 10), activation="tanh", early_stopping=True,
                            max_iter=500, random_state=seed)
    if name == "SVR":
        return NuSVR(C=0.08, nu=0.5)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=1)
    if name == "Tree":
        return GridSearchCV(DecisionTreeRegressor(random_state=seed),
                            {"min_samples_leaf": [5, 10, 20, 50, 100]}, cv=KFold(3), scoring="neg_mean_absolute_error")
    raise ValueError(name)


def fit_predict(name: str, X_train, y_train, hours_train, X_eval, hours_eval, features: list[str],
                loss: str = "squared") -> np.ndarray:
    """Train `name` on (X_train, y_train) and predict X_eval, in watts. X: (n, 144, features).
    `loss` applies to the networks only ("squared" as in the papers, or "absolute")."""
    scaler = Scaler(features).fit(X_train)
    A = scaler.transform(X_train).reshape(len(X_train), -1)
    B = scaler.transform(X_eval).reshape(len(X_eval), -1)
    mu, sd = float(y_train.mean()), float(y_train.std()) or 1.0
    target = (y_train - mu) / sd
    if name in DEEP:
        from energy.deep import fit_predict_deep  # PyTorch is only needed for these
        return fit_predict_deep(name, scaler.transform(X_train), target, scaler.transform(X_eval), loss=loss) * sd + mu
    if name not in PER_HOUR:
        return _make(name).fit(A, target).predict(B) * sd + mu
    out = np.full(len(B), mu, dtype=float)
    for h in range(24):
        tr, ev = hours_train == h, hours_eval == h
        if ev.any() and tr.sum() >= 5:
            out[ev] = _make(name).fit(A[tr], target[tr]).predict(B[ev]) * sd + mu
    return out


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mape(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def select_features(features: list[str], X_tr, y_tr, h_tr, X_val, y_val, h_val, keep: int = 6):
    """The paper's backward-eliminated exhaustive selection, scored by ANN1 validation MAE."""
    def score(subset):
        idx = [features.index(f) for f in subset]
        p = fit_predict("ANN1", X_tr[:, :, idx], y_tr, h_tr, X_val[:, :, idx], h_val, list(subset))
        return mae(y_val, p)

    remaining, trace = list(features), []
    while len(remaining) > keep:
        scored = {f: score([g for g in remaining if g != f]) for f in remaining}
        drop = min(scored, key=scored.get)
        trace.append(("drop", drop, scored[drop]))
        remaining.remove(drop)
    best, best_score = None, np.inf
    for r in range(1, len(remaining) + 1):
        for subset in itertools.combinations(remaining, r):
            s = score(list(subset))
            if s < best_score:
                best, best_score = list(subset), s
    trace.append(("best", best, best_score))
    return best, trace


def hourly_weights(predictions: np.ndarray, y: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """(24, n_forecasters) non-negative weights summing to 1, fitted per hour on validation data."""
    k = predictions.shape[1]
    W = np.full((24, k), 1.0 / k)
    for h in range(24):
        m = hours == h
        if m.sum() >= k:
            w, _ = nnls(predictions[m], y[m])
            if w.sum() > 0:
                W[h] = w / w.sum()
    return W


def apply_weights(predictions: np.ndarray, hours: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nk->n", predictions, W[hours])
