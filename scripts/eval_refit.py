"""Evaluate the habit models on REFIT (all houses), on later weeks never seen in training.

    .venv\\Scripts\\python.exe scripts\\eval_refit.py

Needs scripts\\prepare_refit.py to have run. Writes the per-task results to
data/results/habit_models.csv and prints a summary per appliance type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from habits.paths import RESULTS as PATHS_RESULTS  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from habits.baseline import HourOfWeekBaseline, evaluate  # noqa: E402
from habits.evaluate import appliance_type, tasks  # noqa: E402
from habits.models import fit_and_predict  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402

RESULTS = PATHS_RESULTS  # data/results, or data/results/ukdale


MODELS = ["hour_of_week", "logistic_regression", "random_forest"]


def main(extra: list[str] | None = None) -> None:
    """extra: more models from habits/models.py, e.g. ["lightgbm", "catboost"] (on Kaggle)."""
    global MODELS
    MODELS = MODELS + (extra or [])
    uses, covered, appliances = load_prepared()
    rows, importances = [], []
    for t in tasks(uses, covered, appliances):
        for name in MODELS:
            if name == "hour_of_week":
                predictions = HourOfWeekBaseline().fit(t.train).tune_threshold(t.val).predict(t.test.index)
            else:
                predictions, details = fit_and_predict(name, t)
                if "importances" in details:
                    importances.append(details["importances"])
            scores = evaluate(t.test, predictions)
            rows.append({"house": t.house, "column": t.column, "appliance": t.appliance,
                         "type": appliance_type(t.appliance),
                         "category": t.category, "model": name, "test_hours": len(t.test),
                         **{k: round(v, 3) for k, v in scores.items()}})
        print(".", end="", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "habit_models.csv", index=False)

    pd.set_option("display.width", 200)
    n_tasks = df[["house", "column"]].drop_duplicates().shape[0]
    print(f"\n{n_tasks} house x appliance tasks, {df['house'].nunique()} houses; F1 on the last 20% of each house\n")
    wide = df.pivot_table(index=["house", "column", "appliance", "type"], columns="model", values="f1").reset_index()
    by_type = wide.groupby("type").agg(tasks=("hour_of_week", "size"),
                                       **{m: (m, "median") for m in MODELS}).round(3)
    by_type = by_type.sort_values("random_forest", ascending=False)
    print("Median F1 per appliance type:")
    print(by_type.to_string())
    print("\nAll tasks:")
    for m in MODELS:
        print(f"  {m:<20} median F1 {wide[m].median():.3f}   mean F1 {wide[m].mean():.3f}")
    for m in MODELS[1:]:
        better = (wide[m] > wide["hour_of_week"]).mean()
        # Paired test over tasks: is the model's F1 higher than the baseline's on the same task?
        p_value = wilcoxon(wide[m], wide["hour_of_week"], alternative="greater").pvalue
        print(f"  {m} beats the baseline on {better:.0%} of tasks (one-sided Wilcoxon p = {p_value:.1e})")
    imp = pd.DataFrame(importances).mean().sort_values(ascending=False)
    print("\nRandom forest: mean feature importance")
    print(imp.round(3).to_string())
    for m in MODELS:
        if m in ("hour_of_week", "random_forest"):
            continue
        d = wide[m] - wide["random_forest"]
        p_two = wilcoxon(wide[m], wide["random_forest"]).pvalue if (d != 0).any() else 1.0
        print(f"  {m} vs random forest: better on {(d > 0).sum()}, worse on {(d < 0).sum()} of {len(d)} tasks "
              f"(two-sided Wilcoxon p = {p_two:.1e})")


if __name__ == "__main__":
    args = sys.argv[1:]  # e.g. --models lightgbm,catboost
    main(args[args.index("--models") + 1].split(",") if "--models" in args else None)
