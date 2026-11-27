"""Weekly habit finding: a model (CatBoost) against counting, on the test weeks of every home.

    python scripts/eval_weekly_habits.py [--max-homes N] [--folds K]     (CAPSTONE_DATASET=ukdale or casas; Kaggle)

Every week of each home's test period (its last 20% of days), each system decides which slots are
habits for that week, per appliance and kind of day (habits/weekly.py):

  today        counting over the first 80% of days, once (habits/routines.py), as the home does now
  rolling      counting over the last 3 weeks, every week (no model: is recency alone enough?)
  rate         the last 3 weeks' rate in the slot with the model's thresholds (no model)
  model <set>  CatBoost's predicted rate: a new habit at the threshold or more, kept while at 0.6 x it,
               the threshold chosen on a fifth of the training homes (never on the tested home);
               for each set of inputs of the ablation: A+B+D, then adding C1, C2, C3, C4

The model is trained across homes: every home is predicted by a model trained on the OTHER homes,
and only on their weeks before the tested home's test period (habits/weekly.py: time_folds; REFIT's
homes were recorded at the same time, so another home's same weeks would leak the season being
predicted). Scored on what really happened that week:

  precision  share of habit-days on which the habit happened (a use started in the habit's window)
  recall     share of the real switch-ons that fell inside a habit window
  F1         their balance, per task (home x appliance); medians over tasks, paired Wilcoxon

Then SHAP on the full model: how much each input group (A, B, C1-C4, D) drives the predictions.
Writes data/results/weekly_habits.csv (per task and system), weekly_habits_shap.csv, and every
prediction with its threshold in weekly_habits_predictions.parquet.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from habits.paths import DATASET, RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402
from habits.routines import WEEKDAYS, WEEKEND, find_routines  # noqa: E402
from habits.weekly import (SLOTS, SUGGEST, TEST_SHARE, WEEKS_BACK, best_threshold, columns_for, cut_of,  # noqa: E402
                           decide, features, group_of_column, home_data, time_folds)

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices where a home lacks a type
ABLATION = {"A+B+D": ("A", "B", "D"), "+C1": ("A", "B", "D", "C1"), "+C1+C2": ("A", "B", "D", "C1", "C2"),
            "+C1+C2+C3": ("A", "B", "D", "C1", "C2", "C3"), "all (+C4)": ("A", "B", "D", "C1", "C2", "C3", "C4")}
MAX_TRAIN_ROWS = 600_000


def make_regressor():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, depth=6, learning_rate=0.08, loss_function="RMSE",
                                 random_seed=0, verbose=0, allow_writing_files=False), "catboost"
    except ImportError:  # the laptop's quick check: scikit-learn's histogram boosting instead
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0), "hist_gb"


def runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous habit slots as (start, end) windows."""
    out, s = [], None
    for i, on in enumerate(row):
        if on and s is None:
            s = i
        if not on and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(row)))
    return out


def score(habit: np.ndarray, actual: np.ndarray) -> tuple[int, int, int, int]:
    """(habit-days held, habit-days, real starts inside habits, real starts) for one week and kind.
    habit: 48 bool; actual: days x 48 bool, the starts of that week's days of that kind."""
    held = days = 0
    for a, b in runs(habit):
        days += len(actual)
        held += int(actual[:, a:b].any(axis=1).sum())
    return held, days, int(actual[:, habit].sum()), int(actual.sum())


def counted(uses: pd.DataFrame, days, appliance: str, min_days: int) -> np.ndarray:
    """2 kinds x 48: the slots of the routines counting finds on these days."""
    out = np.zeros((2, SLOTS), dtype=bool)
    for r in find_routines(uses, days, appliance, min_days=min_days, kinds=(WEEKDAYS, WEEKEND)):
        out[0 if r.weekdays == WEEKDAYS else 1, r.start_slot:r.start_slot + r.n_slots] = True
    return out


def evaluate(frame: pd.DataFrame, homes: dict, uses: pd.DataFrame, appliances: pd.DataFrame,
             predictions: dict[str, np.ndarray], thresholds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per task and system: precision, recall and F1 summed over the task's test weeks."""
    rows = []
    names = appliances.set_index(["house", "column"])["appliance"]
    for (house, col), g in frame.groupby(["house", "column"], sort=False):
        home = homes[house]
        cut = cut_of(home)
        weeks = sorted(g["week"].unique())
        test_weeks = [w for w in weeks if w >= cut]
        if not test_weeks:
            continue
        u = uses[(uses["house"] == house) & (uses["column"] == col)]
        learn_days = [d for d in home.days if pd.Timestamp(d) < cut]
        today = counted(u[u["start"] < cut], learn_days, names[(house, col)], min_days=8)
        m = home.starts[col]
        totals = {}
        previous = {name: np.zeros((2, SLOTS), dtype=bool) for name in ["rate", *predictions]}
        for w in weeks:
            in_week = home.day_week == int((w - home.weeks[0]).days // 7)
            sel = g[g["week"] == w]
            decided = {}
            for name, values in [("rate", sel["own_3w"].to_numpy()), *((n, p[sel.index]) for n, p in predictions.items())]:
                grid = np.zeros((2, SLOTS))
                grid[sel["kind"].to_numpy(), sel["slot"].to_numpy()] = np.nan_to_num(values)
                suggest = thresholds[name][sel.index[0]] if name in thresholds else SUGGEST
                decided[name] = decide(grid, previous[name], suggest)
                previous[name] = decided[name]
            if w not in test_weeks:
                continue
            window_days = [d for d in home.days if w - pd.Timedelta(days=7 * WEEKS_BACK) <= pd.Timestamp(d) < w]
            rolling = counted(u[u["start"].dt.date.isin(set(window_days))], window_days, names[(house, col)],
                              min_days=4)
            for kind in sorted(sel["kind"].unique()):
                actual = m[in_week & (home.day_kind == kind)]
                for name, habit in [("today", today[kind]), ("rolling", rolling[kind]),
                                    *((n, d[kind]) for n, d in decided.items())]:
                    t = totals.setdefault(name, np.zeros(4))
                    t += score(habit, actual)
        for name, (held, days, inside, starts) in totals.items():
            precision = held / days if days else np.nan
            recall = inside / starts if starts else np.nan
            f1 = (2 * precision * recall / (precision + recall)
                  if precision == precision and recall == recall and precision + recall > 0 else 0.0)
            rows.append({"house": house, "column": col, "appliance": names[(house, col)], "system": name,
                         "precision": precision, "recall": recall, "f1": f1, "habit_days": days})
    return pd.DataFrame(rows)


def save_predictions(frame: pd.DataFrame, predictions: dict, thresholds: dict, path: Path) -> None:
    """Every prediction and threshold with its example's keys, to check results without rerunning."""
    out = frame[["house", "column", "week", "kind", "slot", "label"]].copy()
    for name, p in predictions.items():
        out[f"pred {name}"] = p.astype("float32")
        out[f"threshold {name}"] = np.asarray(thresholds[name], dtype="float32")
    out.to_parquet(path, index=False)


def main(max_homes: int | None = None, folds: int = 5) -> None:
    uses, covered, appliances = load_prepared()
    homes = {}
    for house in sorted(appliances["house"].unique())[:max_homes]:
        h = home_data(house, uses, covered, appliances)
        if h is not None and h.tasks:
            homes[house] = h
    all_types = sorted({t for h in homes.values() for t in h.types.values()})
    frame = pd.concat([features(h, all_types) for h in homes.values()], ignore_index=True)
    frame["type_code"] = pd.Categorical(frame["type"], categories=all_types).codes
    print(f"Weekly habit finding, {DATASET}: {len(homes)} homes, {frame.groupby(['house', 'column']).ngroups} tasks, "
          f"{len(frame):,} examples, {len(all_types)} appliance types", flush=True)

    predictions = {name: np.full(len(frame), np.nan) for name in ABLATION}
    thresholds = {name: np.full(len(frame), SUGGEST) for name in ABLATION}
    houses = frame["house"].to_numpy()
    shap_rows, kind = [], None
    for fold, f in enumerate(time_folds(frame, homes, folds)):
        # Learn only from the other homes' weeks before this fold's earliest test start (no future weeks).
        train, test = f.other_past, f.tested
        if len(train) == 0:
            print(f"  fold {fold + 1}: no earlier weeks of other homes to learn from; skipped", flush=True)
            continue
        train = np.random.default_rng(fold).permutation(train)[:MAX_TRAIN_ROWS]
        for name, groups in ABLATION.items():
            cols = columns_for(frame, groups)
            cols = [c for c in cols if frame.iloc[train][c].notna().any()]  # types no training home has
            # The threshold: chosen on a fifth of the TRAINING homes, by a model that did not see them.
            train_homes = np.unique(houses[train])
            if len(train_homes) >= 2:
                inner = np.random.default_rng(fold).choice(train_homes, max(1, round(0.2 * len(train_homes))),
                                                           replace=False)
                held = np.isin(houses[train], inner)
                inner_cols = [c for c in cols if frame.iloc[train[~held]][c].notna().any()]
                model, kind = make_regressor()
                model.fit(frame.iloc[train[~held]][inner_cols], frame.iloc[train[~held]]["label"])
                guess = np.clip(model.predict(frame.iloc[train[held]][inner_cols]), 0, 1)
                thresholds[name][test] = best_threshold(guess, frame.iloc[train[held]]["label"].to_numpy())
            model, kind = make_regressor()
            model.fit(frame.iloc[train][cols], frame.iloc[train]["label"])
            predictions[name][test] = np.clip(model.predict(frame.iloc[test][cols]), 0, 1)
            if fold == 0 and name == "all (+C4)":
                shap_rows = explain(model, frame.iloc[test][cols])
        print(f"  fold {fold + 1}: {len(np.unique(houses[test]))} homes predicted", flush=True)

    results = evaluate(frame, homes, uses, appliances, {f"model {n}": p for n, p in predictions.items()},
                       {f"model {n}": t for n, t in thresholds.items()})
    print("  thresholds chosen (mean over homes): " +
          ", ".join(f"{n} {np.mean(t):.2f}" for n, t in thresholds.items()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "weekly_habits.csv", index=False)
    save_predictions(frame, predictions, thresholds, RESULTS / "weekly_habits_predictions.parquet")
    wide = results.pivot_table(index=["house", "column"], columns="system", values="f1")
    print(f"\n{kind}; median over {len(wide)} tasks, test weeks only:")
    summary = results.groupby("system")[["precision", "recall", "f1"]].median().round(3)
    print(summary.sort_values("f1", ascending=False).to_string())
    best = f"model {max(ABLATION, key=lambda n: wide[f'model {n}'].median())}"
    for other in ["today", "rolling", "rate", *(f"model {n}" for n in ABLATION)]:
        if other == best or other not in wide:
            continue
        d = (wide[best] - wide[other]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        print(f"  {best} vs {other}: F1 higher on {(d > 0).sum()}, lower on {(d < 0).sum()} of {len(d)} tasks "
              f"(two-sided Wilcoxon p = {p:.1e})")
    if len(shap_rows):
        shap_rows.to_csv(RESULTS / "weekly_habits_shap.csv", index=False)
        print("\nSHAP, full model (fold 1): share of the prediction explained by each input group")
        print((shap_rows.groupby("group")["share"].sum() * 100).round(1).sort_values(ascending=False).to_string())
        print("  top inputs:", ", ".join(shap_rows.nlargest(10, "share")["input"]))


def explain(model, X: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        return pd.DataFrame()
    sample = X.sample(min(n, len(X)), random_state=0)
    values = shap.TreeExplainer(model).shap_values(sample)
    magnitude = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    share = magnitude / magnitude.sum()
    return pd.DataFrame({"input": share.index, "group": [group_of_column(c) for c in share.index],
                         "share": share.to_numpy()})


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[args.index("--max-homes") + 1]) if "--max-homes" in args else None,
         int(args[args.index("--folds") + 1]) if "--folds" in args else 5)
