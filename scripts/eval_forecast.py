"""Reproduce Yaprakdal (2023) on single homes: hour-ahead load forecasting on REFIT.

    .venv\\Scripts\\python.exe scripts\\eval_forecast.py            # all 20 homes
    .venv\\Scripts\\python.exe scripts\\eval_forecast.py 6 17       # some homes
    .venv\\Scripts\\python.exe scripts\\eval_forecast.py --deep     # add the CNNs and BiLSTMs

--deep (needs PyTorch and a finished normal run, for the selected features): the
paper's two CNNs and two BiLSTMs, plus the GRU and CNN-GRU hybrid of Yaprakdal
(2024), on the time-ordered protocol with each home's selected features, alongside
the classical forecasters, and an hourly ensemble of all twelve. Written to data/results/forecast_deep.csv. (Both protocols for all four
networks on 20 homes would take about 20 hours on a laptop CPU.)

Two protocols, both run on every home:

paper      as in the paper's Section 3.1: the days are cut into five equal folds
           (in time order); fold 1 is the development fold (feature selection and
           ensemble weights, with the models trained on folds 2-5); then four
           tests: test on fold k (k = 2..5), train on the other four. Mean and
           standard deviation over the four tests.
time       a strict time-ordered split: develop on the first 60% (train) and next
           20% (validation); retrain on the first 80%; test once on the last 20%.
           Nothing after a test hour is ever used to predict it (in the paper's
           protocol, folds after the test fold are in the training data).

For each: the six forecasters, the ANN pair average (as in the paper's tables),
the hourly ensemble, with all features and with the selected ones; and the
baselines: the paper's mean and random, plus "same as last hour" (persistence)
and "same hour yesterday", which the paper does not report.

Writes data/results/forecast.csv (one row per home x protocol x feature set x
forecaster x test) and data/results/forecast_features.csv (selected features).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.paths import RESULTS as PATHS_RESULTS  # noqa: E402

from energy.data import FEATURES, house_hours, weather, windows  # noqa: E402
from energy.forecasters import (DEEP, FORECASTERS, apply_weights, fit_predict, hourly_weights, mae,  # noqa: E402
                                mape, select_features)
from habits.prepare import load_aggregate  # noqa: E402

RESULTS = PATHS_RESULTS  # data/results, or data/results/ukdale


def baselines(X, y_train, y_test, rng) -> dict[str, np.ndarray]:
    return {
        "Mean": np.full(len(X), y_train.mean()),
        "Random": rng.choice(y_train, size=len(X)),
        "Persistence (last hour)": X[:, -1, 0],
        "Same hour yesterday": X[:, -24, 0],
    }


def run_models(X, y, hours, features, sel, train, val, test, W=None, rng=None, forecasters=FORECASTERS):
    """Predictions on `test` for every forecaster and ensemble; ensemble weights from `val` if W is None."""
    idx = [features.index(f) for f in sel]
    Xs = X[:, :, idx]
    preds = {name: fit_predict(name, Xs[train], y[train], hours[train], Xs[test], hours[test], sel)
             for name in forecasters}
    P = np.column_stack([preds[n] for n in forecasters])
    if W is None:
        Pv = np.column_stack([fit_predict(n, Xs[train], y[train], hours[train], Xs[val], hours[val], sel)
                              for n in forecasters])
        W = hourly_weights(Pv, y[val], hours[val])
    for a, b, pair in (("ANN1", "ANN2", "ANN1&2"), ("CNN1", "CNN2", "CNN1&2"), ("RNN1", "RNN2", "RNN1&2")):
        if a in preds and b in preds:
            preds[pair] = (preds[a] + preds[b]) / 2
    preds["Hourly ensemble"] = apply_weights(P, hours[test], W)
    return preds, W


def evaluate_house_deep(house: int, aggregate, wx, rng, selected: list[str]) -> list[dict]:
    """Time-ordered protocol, selected features, the classical forecasters plus the four networks."""
    df = house_hours(house, aggregate, wx)
    X, y, when = windows(df, FEATURES)
    hours = when.hour.to_numpy()
    n = len(X)
    a, b = int(n * 0.6), int(n * 0.8)
    tr, va, te, trva = np.arange(a), np.arange(a, b), np.arange(b, n), np.arange(b)
    allf = FORECASTERS + DEEP
    _, W = run_models(X, y, hours, FEATURES, selected, tr, va, va, forecasters=allf)
    preds, _ = run_models(X, y, hours, FEATURES, selected, trva, None, te, W=W, forecasters=allf)
    preds.update(baselines(X[te], y[trva], y[te], rng))
    return [{"house": house, "protocol": "time", "features": "selected", "test": 1, "model": name,
             "mae_w": mae(y[te], p), "mape": mape(y[te], p)} for name, p in preds.items()]


def evaluate_house(house: int, aggregate, wx, rng) -> tuple[list[dict], list[dict]]:
    df = house_hours(house, aggregate, wx)
    X, y, when = windows(df, FEATURES)
    hours = when.hour.to_numpy()
    n = len(X)
    rows, chosen = [], []

    def record(protocol, fset, test_no, preds, y_test):
        for name, p in preds.items():
            rows.append({"house": house, "protocol": protocol, "features": fset, "test": test_no,
                         "model": name, "mae_w": mae(y_test, p), "mape": mape(y_test, p)})

    # --- time-ordered protocol -------------------------------------------------------------
    a, b = int(n * 0.6), int(n * 0.8)
    tr, va, te, trva = np.arange(a), np.arange(a, b), np.arange(b, n), np.arange(b)
    sel, trace = select_features(FEATURES, X[tr], y[tr], hours[tr], X[va], y[va], hours[va])
    chosen.append({"house": house, "protocol": "time", "selected": ",".join(sel),
                   "dropped": ",".join(str(t[1]) for t in trace if t[0] == "drop")})
    for fset, feats in (("all", FEATURES), ("selected", sel)):
        _, W = run_models(X, y, hours, FEATURES, feats, tr, va, va)  # weights from validation
        preds, _ = run_models(X, y, hours, FEATURES, feats, trva, None, te, W=W)  # retrain on 80%, test once
        preds.update(baselines(X[te], y[trva], y[te], rng))
        record("time", fset, 1, preds, y[te])

    # --- the paper's protocol: five day-folds, fold 1 for development ------------------------
    days = pd.Index(when.normalize())
    unique_days = days.unique()
    fold_of_day = np.repeat(np.arange(5), int(np.ceil(len(unique_days) / 5)))[:len(unique_days)]
    fold = pd.Series(fold_of_day, index=unique_days).reindex(days).to_numpy()
    dev_val, dev_train = np.flatnonzero(fold == 0), np.flatnonzero(fold != 0)
    sel_p, trace_p = select_features(FEATURES, X[dev_train], y[dev_train], hours[dev_train],
                                     X[dev_val], y[dev_val], hours[dev_val])
    chosen.append({"house": house, "protocol": "paper", "selected": ",".join(sel_p),
                   "dropped": ",".join(str(t[1]) for t in trace_p if t[0] == "drop")})
    for fset, feats in (("all", FEATURES), ("selected", sel_p)):
        _, W = run_models(X, y, hours, FEATURES, feats, dev_train, dev_val, dev_val)
        for k in range(1, 5):
            test_idx, train_idx = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
            preds, _ = run_models(X, y, hours, FEATURES, feats, train_idx, None, test_idx, W=W)
            preds.update(baselines(X[test_idx], y[train_idx], y[test_idx], rng))
            record("paper", fset, k, preds, y[test_idx])
    return rows, chosen


def main_deep(houses: list[int]) -> None:
    import torch
    torch.set_num_threads(8)
    chosen = pd.read_csv(RESULTS / "forecast_features.csv")
    aggregate, wx, rng = load_aggregate(), weather(), np.random.default_rng(0)
    rows = []
    for house in houses:
        sel = chosen[(chosen["house"] == house) & (chosen["protocol"] == "time")]["selected"]
        if sel.empty:
            print(f"house {house:>2}: no selected features yet (run without --deep first)")
            continue
        t = time.perf_counter()
        rows += evaluate_house_deep(house, aggregate, wx, rng, sel.iloc[0].split(","))
        print(f"house {house:>2}: {time.perf_counter() - t:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "forecast_deep.csv", index=False)
    df = pd.DataFrame(rows)
    per_house = df.groupby(["model", "house"])[["mae_w", "mape"]].mean().reset_index()
    pd.set_option("display.width", 200)
    print(f"\nTime-ordered protocol, selected features, with the networks: median over {df['house'].nunique()} homes")
    print(per_house.groupby("model")[["mae_w", "mape"]].median().sort_values("mape").round(2).to_string())


def main(houses: list[int]) -> None:
    aggregate, wx, rng = load_aggregate(), weather(), np.random.default_rng(0)
    rows, chosen = [], []
    for house in houses:
        t = time.perf_counter()
        r, c = evaluate_house(house, aggregate, wx, rng)
        rows += r
        chosen += c
        print(f"house {house:>2}: {time.perf_counter() - t:.0f} s", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS / "forecast.csv", index=False)  # saved as it goes
        pd.DataFrame(chosen).to_csv(RESULTS / "forecast_features.csv", index=False)
    summarise(pd.DataFrame(rows))


def summarise(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    per_house = df.groupby(["protocol", "features", "model", "house"])[["mae_w", "mape"]].mean().reset_index()
    for protocol in ("paper", "time"):
        t = per_house[per_house["protocol"] == protocol]
        table = t.pivot_table(index="model", columns="features", values=["mape", "mae_w"], aggfunc="median")
        table = table.sort_values(("mape", "selected"))
        print(f"\nProtocol '{protocol}': median over {t['house'].nunique()} homes")
        print(table.round(2).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    houses = [int(a) for a in args if a != "--deep"] or sorted(load_aggregate()["house"].unique())  # this dataset's homes
    (main_deep if "--deep" in args else main)(houses)
