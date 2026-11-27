"""Weekly habit finding: which data to train on, and which model, on the test weeks of every home.

    python scripts/eval_weekly_models.py [--part regimes|models|all] [--models a,b] [--tag T]
                                         [--max-homes N] [--folds K]    (CAPSTONE_DATASET=...; Kaggle)

The same examples, decisions and scoring as scripts/eval_weekly_habits.py (habits/weekly.py), with
every input group (A, B, C1-C4, D). Two questions:

1. Regimes (CatBoost): which homes' data should the model learn from?
     own      the home's own weeks before its test period only
     other    the other homes only, as a brand-new home would have
     hybrid   the other homes plus the home's own weeks
   Every regime learns only from weeks before the tested home's test period, from every home: no
   future week is learned from (habits/weekly.py: time_folds).
   A home's own habits need its own data; shared data may catch patterns one home has too few
   examples of. If the hybrid beats both, a running home should learn from both.
2. Models (other-homes regime), two families of tree ensembles and two models outside trees:
     random forest (bagging); XGBoost, LightGBM, CatBoost (boosting: the three of Gad et al.);
     a linear model (ridge regression) and a neural network (MLP) on scaled inputs, with empty
     values set to 0 plus a "has this appliance type" flag and the type one-hot encoded.
   CatBoost reads the appliance type as a category, the others as a number or one-hot.

Thresholds: chosen without the test weeks. own and hybrid: on the tested home's last fifth of its
learning weeks, by a model that did not see those weeks; other: on a fifth of the training homes.
Writes data/results/weekly_models[_TAG].csv (per task and system) and every prediction with its
threshold in weekly_models[_TAG]_predictions.parquet, and prints the comparisons.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from eval_weekly_habits import evaluate, save_predictions  # noqa: E402
from habits.paths import DATASET, RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402
from habits.weekly import (SUGGEST, best_threshold, columns_for, cut_of, features, home_data,  # noqa: E402
                           time_folds)

warnings.filterwarnings("ignore", category=RuntimeWarning)
ALL_INPUTS = ("A", "B", "C1", "C2", "C3", "C4", "D")
MODELS = ["random_forest", "xgboost", "lightgbm", "catboost", "linear", "mlp"]
MAX_TRAIN_ROWS = 600_000
MAX_TRAIN_ROWS_SLOW = 300_000  # the random forest and the MLP


def make(name: str):
    """A regressor for the share of days with a switch-on; None if its library is missing."""
    try:
        if name == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=-1,
                                         random_state=0)
        if name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", random_state=0, n_jobs=-1)
        if name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        if name == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=400, depth=6, learning_rate=0.08, loss_function="RMSE",
                                     random_seed=0, verbose=0, allow_writing_files=False, cat_features=["type_code"])
        if name == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        if name == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=60, random_state=0)
    except ImportError:
        return None
    raise ValueError(name)


class Prepared:
    """The input table each model needs: trees read it as it is (NaN allowed); the linear model and the
    MLP get NaN as 0 plus a flag per input with empty values, the type one-hot, and scaled columns."""

    def __init__(self, name: str, cols: list[str], frame: pd.DataFrame, fit_rows: np.ndarray):
        self.name, self.cols = name, cols
        self.scaled = name in ("linear", "mlp")
        if self.scaled:
            sub = frame.iloc[fit_rows]
            self.flags = [c for c in cols if c != "type_code" and sub[c].isna().any()]
            self.types = sorted(frame["type_code"].unique())
            X = self._raw(sub)
            self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-9

    def _raw(self, sub: pd.DataFrame) -> np.ndarray:
        num = sub[[c for c in self.cols if c != "type_code"]].to_numpy(dtype=float)
        flags = sub[self.flags].isna().to_numpy(dtype=float)
        onehot = (sub["type_code"].to_numpy()[:, None] == np.array(self.types)[None, :]).astype(float)
        return np.hstack([np.nan_to_num(num), flags, onehot])

    def X(self, sub: pd.DataFrame):
        if not self.scaled:
            return sub[self.cols]
        return (self._raw(sub) - self.mean) / self.std


def fit_predict(name: str, frame: pd.DataFrame, cols: list[str], fit_rows: np.ndarray, pred_rows: np.ndarray,
                seed: int = 0) -> np.ndarray | None:
    cap = MAX_TRAIN_ROWS_SLOW if name in ("random_forest", "mlp") else MAX_TRAIN_ROWS
    fit_rows = np.random.default_rng(seed).permutation(fit_rows)[:cap]
    cols = [c for c in cols if frame.iloc[fit_rows][c].notna().any()]
    model = make(name)
    if model is None:
        return None
    prep = Prepared(name, cols, frame, fit_rows)
    model.fit(prep.X(frame.iloc[fit_rows]), frame.iloc[fit_rows]["label"].to_numpy())
    return np.clip(model.predict(prep.X(frame.iloc[pred_rows])), 0, 1)


def late_rows(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Of these rows, the ones in each home's last fifth of weeks (for choosing the threshold)."""
    if len(rows) == 0:
        return rows
    sub = frame.iloc[rows]
    edge = sub.groupby("house")["week"].transform(lambda w: w.quantile(0.8)).to_numpy()
    return rows[sub["week"].to_numpy() >= edge]


def run(name: str, regime: str, frame: pd.DataFrame, homes: dict, folds: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Predictions and thresholds for every row. A model learns only from weeks before its tested homes'
    test period, from every home (habits/weekly.py: time_folds), so no future week is ever learned from.

      own     each home alone, its weeks before its own test period
      other   the other homes' weeks before the fold's earliest test start
      hybrid  the same, plus the tested homes' own weeks before it
    The threshold is chosen without the test weeks: own and hybrid on the last fifth of the tested
    homes' own learning weeks (by a model that did not see them); other on a fifth of the training homes.
    """
    cols = columns_for(frame, ALL_INPUTS)
    houses = frame["house"].to_numpy()
    weeks = frame["week"].to_numpy()
    labels = frame["label"].to_numpy()
    pred, thr = np.full(len(frame), np.nan), np.full(len(frame), SUGGEST)
    if regime == "own":
        jobs = []
        for h, home in homes.items():
            tested = np.flatnonzero(houses == h)
            jobs.append((tested, tested[weeks[tested] < np.datetime64(cut_of(home))], None))
    else:
        jobs = [(f.tested, f.other_past if regime == "other" else np.r_[f.other_past, f.own_past], f)
                for f in time_folds(frame, homes, folds)]
    for k, (tested, fit, fold) in enumerate(jobs):
        if len(fit) == 0:
            continue
        if regime == "own":
            held = late_rows(frame, fit)
        elif regime == "hybrid":
            held = late_rows(frame, fold.own_past)
        else:
            held = np.array([], int)
        if len(held) == 0:  # a fifth of the training homes, unseen by the threshold model
            train_homes = np.unique(houses[fit])
            if len(train_homes) >= 2:
                inner = np.random.default_rng(k).choice(train_homes, max(1, round(0.2 * len(train_homes))),
                                                        replace=False)
                held = fit[np.isin(houses[fit], inner)]
        early = np.setdiff1d(fit, held)
        if len(early) and len(held):
            guess = fit_predict(name, frame, cols, early, held, k)
            if guess is None:
                return None
            thr[tested] = best_threshold(guess, labels[held])
        p = fit_predict(name, frame, cols, fit, tested, k)
        if p is None:
            return None
        pred[tested] = p
    return pred, thr


def compare(results: pd.DataFrame, part: str, systems: list[str]) -> None:
    sub = results[results["system"].isin(systems)]
    wide = sub.pivot_table(index=["house", "column"], columns="system", values="f1")
    print(f"\n{part}: median over {len(wide)} tasks (test weeks)")
    print(sub.groupby("system")[["precision", "recall", "f1"]].median().round(3)
          .sort_values("f1", ascending=False).to_string())
    best = wide.median().idxmax()
    for other in wide.columns:
        if other == best:
            continue
        d = (wide[best] - wide[other]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        print(f"  {best} vs {other}: F1 higher on {(d > 0).sum()}, lower on {(d < 0).sum()} of {len(d)} "
              f"(two-sided Wilcoxon p = {p:.1e})")


def main(max_homes: int | None = None, folds: int = 5, part: str = "all", models: list[str] | None = None,
         tag: str = "") -> None:
    """part: "regimes" (CatBoost in the three regimes), "models" (the models in the other-homes regime)
    or "all". models: which models for the "models" part. tag: names the output files, so parallel
    Kaggle jobs do not overwrite each other."""
    uses, covered, appliances = load_prepared()
    homes = {}
    for house in sorted(appliances["house"].unique())[:max_homes]:
        h = home_data(house, uses, covered, appliances)
        if h is not None and h.tasks:
            homes[house] = h
    all_types = sorted({t for h in homes.values() for t in h.types.values()})
    frame = pd.concat([features(h, all_types) for h in homes.values()], ignore_index=True)
    frame["type_code"] = pd.Categorical(frame["type"], categories=all_types).codes.astype(int)
    print(f"Weekly habit finding, {DATASET}: {len(homes)} homes, {frame.groupby(['house', 'column']).ngroups} "
          f"tasks, {len(frame):,} examples, {len(all_types)} appliance types", flush=True)

    predictions, thresholds = {}, {}
    if part in ("regimes", "all"):
        for regime in ("own", "other", "hybrid"):
            out = run("catboost", regime, frame, homes, folds)
            if out is None:
                print("  catboost not installed: the regimes need it (run on Kaggle)")
                break
            predictions[f"catboost {regime}"], thresholds[f"catboost {regime}"] = out
            print(f"  regime {regime}: done", flush=True)
    if part in ("models", "all"):
        for name in models or MODELS:
            if f"{name} other" in predictions:
                continue
            out = run(name, "other", frame, homes, folds)
            if out is None:
                print(f"  {name}: not installed, skipped")
                continue
            predictions[f"{name} other"], thresholds[f"{name} other"] = out
            print(f"  model {name}: done", flush=True)

    results = evaluate(frame, homes, uses, appliances, predictions, thresholds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"weekly_models{'_' + tag if tag else ''}"
    results.to_csv(RESULTS / f"{stem}.csv", index=False)
    save_predictions(frame, predictions, thresholds, RESULTS / f"{stem}_predictions.parquet")
    print("\nthresholds chosen (mean over homes): " +
          ", ".join(f"{n} {np.nanmean(t):.2f}" for n, t in thresholds.items()))
    regimes = [n for n in predictions if n in ("catboost own", "catboost other", "catboost hybrid")]
    if part in ("regimes", "all") and regimes:
        compare(results, "Regimes (CatBoost)", [*regimes, "today", "rolling", "rate"])
    if part in ("models", "all"):
        compare(results, "Models (other-homes regime)", [n for n in predictions if n.endswith(" other")])


if __name__ == "__main__":
    args = sys.argv[1:]

    def value(flag: str, default: str) -> str:
        return args[args.index(flag) + 1] if flag in args else default

    main(int(value("--max-homes", "0")) or None, int(value("--folds", "5")), value("--part", "all"),
         value("--models", "").split(",") if "--models" in args else None, value("--tag", ""))
