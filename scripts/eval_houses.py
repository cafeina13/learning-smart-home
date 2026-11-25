"""Generalisation: does a model trained on OTHER homes work in a home it has never seen?

    .venv\\Scripts\\python.exe scripts\\eval_houses.py

Leave-one-home-out, per appliance type (types present in at least MIN_HOUSES homes):

    shared model   a random forest trained on all OTHER homes with this appliance type,
                   on their first 60% only (the same calendar window as the personal
                   models, so nothing from the test period is seen), threshold tuned
                   on their validation weeks
    personal model the random forest trained on the home's own first 60% (as in
                   scripts/eval_refit.py)
    baseline       the home's own hour-of-week baseline

All three are scored on the same weeks: the held-out home's last 20%. The shared
model needs no history from the new home to be trained (it still reads the
home's recent weeks as features, as any deployed model would). The test weeks
come after 7 to 12 months of the home's history, so its first weeks are not tested.

Writes data/results/held_out_homes.csv.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.paths import RESULTS as PATHS_RESULTS  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from habits.baseline import HourOfWeekBaseline, evaluate  # noqa: E402
from habits.evaluate import appliance_type, tasks  # noqa: E402
from habits.features import build_features  # noqa: E402
from habits.models import best_threshold, make_model  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402

RESULTS = PATHS_RESULTS  # data/results, or data/results/ukdale
MIN_HOUSES = 5


def main() -> None:
    uses, covered, appliances = load_prepared()

    # Features and splits once per task, grouped by appliance type.
    by_type = defaultdict(list)
    for t in tasks(uses, covered, appliances):
        X = build_features(t.labels, t.other_starts)
        by_type[appliance_type(t.appliance)].append((t, X))

    rows = []
    for kind, items in sorted(by_type.items()):
        if len({t.house for t, _ in items}) < MIN_HOUSES:
            continue
        print(f"{kind}: {len(items)} homes", flush=True)
        for t, X in items:
            others = [(o, Xo) for o, Xo in items if o.house != t.house]
            X_train = pd.concat([Xo.loc[o.train.index] for o, Xo in others])
            y_train = pd.concat([o.train for o, _ in others])
            X_val = pd.concat([Xo.loc[o.val.index] for o, Xo in others])
            y_val = pd.concat([o.val for o, _ in others])

            shared = make_model("random_forest").fit(X_train, y_train)
            threshold = best_threshold(y_val, shared.predict_proba(X_val)[:, 1])
            shared_pred = (shared.predict_proba(X.loc[t.test.index])[:, 1] >= threshold).astype(int)

            personal_pred = np.zeros(len(t.test), dtype=int)
            if t.train.nunique() == 2:
                personal = make_model("random_forest").fit(X.loc[t.train.index], t.train)
                th = best_threshold(t.val, personal.predict_proba(X.loc[t.val.index])[:, 1])
                personal_pred = (personal.predict_proba(X.loc[t.test.index])[:, 1] >= th).astype(int)

            baseline_pred = HourOfWeekBaseline().fit(t.train).tune_threshold(t.val).predict(t.test.index)
            for name, pred in (("shared", shared_pred), ("personal", personal_pred), ("baseline", baseline_pred)):
                rows.append({"type": kind, "house": t.house, "column": t.column, "model": name,
                             **{k: round(v, 3) for k, v in evaluate(t.test, pred).items()}})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "held_out_homes.csv", index=False)

    wide = df.pivot_table(index=["type", "house", "column"], columns="model", values="f1").reset_index()
    pd.set_option("display.width", 200)
    print(f"\n{len(wide)} held-out home x appliance cases\n")
    print("Median F1 per appliance type (each home scored when held out):")
    print(wide.groupby("type").agg(homes=("shared", "size"), baseline=("baseline", "median"),
                                   personal=("personal", "median"), shared=("shared", "median")).round(3).to_string())
    print("\nAll cases:")
    for m in ("baseline", "personal", "shared"):
        print(f"  {m:<9} median F1 {wide[m].median():.3f}   mean F1 {wide[m].mean():.3f}")
    for a, b in (("shared", "baseline"), ("shared", "personal")):
        better = (wide[a] > wide[b]).mean()
        p = wilcoxon(wide[a], wide[b]).pvalue
        print(f"  {a} > {b} on {better:.0%} of cases (two-sided Wilcoxon p = {p:.1e})")


if __name__ == "__main__":
    main()
