"""Weekly habit finding: which data to train on, and which model, on the test weeks of every home.

    python scripts/eval_weekly_models.py [--max-homes N] [--folds K]     (CAPSTONE_DATASET=ukdale or casas; Kaggle)

The same examples, decisions and scoring as scripts/eval_weekly_habits.py (habits/weekly.py), with
every input group (A, B, C1-C4, D). Two questions:

1. Regimes (CatBoost): which homes' data should the model learn from?
     own      the home's own weeks before its test period only
     other    the other homes only (every week), as a brand-new home would have
     hybrid   the other homes plus the home's own weeks before its test period
   A home's own habits need its own data; shared data may catch patterns one home has too few
   examples of. If the hybrid beats both, a running home should learn from both.
2. Models (hybrid regime), two families of tree ensembles and two models outside trees:
     random forest (bagging); XGBoost, LightGBM, CatBoost (boosting: the three of Gad et al.);
     a linear model (ridge regression) and a neural network (MLP) on scaled inputs, with empty
     values set to 0 plus a "has this appliance type" flag and the type one-hot encoded.
   CatBoost reads the appliance type as a category, the others as a number or one-hot.

Thresholds: chosen without the test weeks. own and hybrid: on the tested home's last fifth of its
learning weeks, by a model that did not see those weeks; other: on a fifth of the training homes.
Writes data/results/weekly_models.csv (per task, system and part) and prints the comparisons.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from eval_weekly_habits import TEST_SHARE, evaluate  # noqa: E402
from habits.paths import DATASET, RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402
from habits.weekly import SUGGEST, best_threshold, columns_for, features, home_data  # noqa: E402

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


def learning_weeks(frame: pd.DataFrame, homes: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per row: is the week before the home's test period (learning), and in the learning period's last fifth."""
    learn, late = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    for house, idx in frame.groupby("house").indices.items():
        days = homes[house].days
        cut = pd.Timestamp(days[int(len(days) * (1 - TEST_SHARE))])
        late_cut = pd.Timestamp(days[int(len(days) * (1 - TEST_SHARE) * 0.8)])
        weeks = frame["week"].to_numpy()[idx]
        learn[idx] = weeks < cut
        late[idx] = (weeks >= late_cut) & (weeks < cut)
    return learn, late


def run(name: str, regime: str, frame: pd.DataFrame, homes: dict, folds: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Predictions and thresholds for every row, each home predicted without its own test weeks."""
    cols = columns_for(frame, ALL_INPUTS)
    houses = frame["house"].to_numpy()
    learn, late = learning_weeks(frame, homes)
    pred, thr = np.full(len(frame), np.nan), np.full(len(frame), SUGGEST)
    if regime == "own":
        groups = [np.flatnonzero(houses == h) for h in np.unique(houses)]
        splits = [(np.array([], int), g) for g in groups]
    else:
        splits = list(GroupKFold(n_splits=min(folds, len(homes))).split(frame, groups=houses))
    for k, (others, tested) in enumerate(splits):
        own_learn = tested[learn[tested]]
        fit = own_learn if regime == "own" else others if regime == "other" else np.r_[others, own_learn]
        if len(fit) == 0:
            continue
        # threshold, without the test weeks
        if regime == "other":
            train_homes = np.unique(houses[others])
            inner = np.random.default_rng(k).choice(train_homes, max(1, round(0.2 * len(train_homes))), replace=False)
            held = others[np.isin(houses[others], inner)]
            guess = fit_predict(name, frame, cols, others[~np.isin(houses[others], inner)], held, k)
            labels = frame["label"].to_numpy()[held]
        else:
            held = tested[late[tested]]
            early = np.setdiff1d(fit, held)
            guess = fit_predict(name, frame, cols, early, held, k) if len(early) and len(held) else None
            labels = frame["label"].to_numpy()[held]
        if guess is None and make(name) is None:
            return None
        if guess is not None and len(guess):
            thr[tested] = best_threshold(guess, labels)
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


def main(max_homes: int | None = None, folds: int = 5) -> None:
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
          f"tasks, {len(frame):,} examples", flush=True)

    predictions, thresholds, parts = {}, {}, {}
    for regime in ("own", "other", "hybrid"):
        out = run("catboost", regime, frame, homes, folds)
        if out is None:
            print("  catboost not installed: the regimes need it (run on Kaggle)")
            break
        predictions[f"catboost {regime}"], thresholds[f"catboost {regime}"] = out
        parts[f"catboost {regime}"] = "regimes"
        print(f"  regime {regime}: done", flush=True)
    for name in MODELS:
        if f"{name} hybrid" in predictions:
            parts[f"{name} hybrid"] = "models"
            continue
        out = run(name, "hybrid", frame, homes, folds)
        if out is None:
            print(f"  {name}: not installed, skipped")
            continue
        predictions[f"{name} hybrid"], thresholds[f"{name} hybrid"] = out
        parts[f"{name} hybrid"] = "models"
        print(f"  model {name}: done", flush=True)

    results = evaluate(frame, homes, uses, appliances, predictions, thresholds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "weekly_models.csv", index=False)
    print("\nthresholds chosen (mean over homes): " +
          ", ".join(f"{n} {np.mean(t):.2f}" for n, t in thresholds.items()))
    regimes = [n for n, p in parts.items() if n.startswith("catboost ")]
    if regimes:
        compare(results, "Regimes (CatBoost)", [*regimes, "today", "rolling", "rate"])
    compare(results, "Models (hybrid regime)", [n for n in predictions if n.endswith(" hybrid")])


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[args.index("--max-homes") + 1]) if "--max-homes" in args else None,
         int(args[args.index("--folds") + 1]) if "--folds" in args else 5)
