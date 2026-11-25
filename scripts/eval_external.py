"""External test: train on REFIT (Loughborough), test on UK-DALE (London).

    python scripts/eval_external.py        (needs both datasets prepared)

The strongest generalisation test available: a different study, different
years, different meters, a different city. Per appliance type present in both:

    REFIT model   random forest trained on all 20 REFIT homes (their first 60%,
                  threshold tuned on their validation 20%), never shown UK-DALE
    personal      the UK-DALE home's own random forest (its first 60%)
    baseline      the UK-DALE home's own hour-of-week baseline

All scored on the UK-DALE home's last 20%, the same weeks. Writes
data/results/ukdale/external.csv.
"""

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.baseline import HourOfWeekBaseline, evaluate  # noqa: E402
from habits.evaluate import appliance_type, tasks  # noqa: E402
from habits.features import build_features  # noqa: E402
from habits.models import best_threshold, make_model  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results" / "ukdale"


def by_type(dataset: str):
    out = defaultdict(list)
    for t in tasks(*load_prepared(PROCESSED / dataset)):
        out[appliance_type(t.appliance)].append((t, build_features(t.labels, t.other_starts)))
    return out


def main() -> None:
    refit, ukdale = by_type("refit"), by_type("ukdale")
    rows = []
    for kind in sorted(set(refit) & set(ukdale)):
        train = refit[kind]
        X_tr = pd.concat([X.loc[t.train.index] for t, X in train])
        y_tr = pd.concat([t.train for t, _ in train])
        X_va = pd.concat([X.loc[t.val.index] for t, X in train])
        y_va = pd.concat([t.val for t, _ in train])
        model = make_model("random_forest").fit(X_tr, y_tr)
        threshold = best_threshold(y_va, model.predict_proba(X_va)[:, 1])
        print(f"{kind}: trained on {len(train)} REFIT homes, tested on {len(ukdale[kind])} UK-DALE appliances",
              flush=True)
        for t, X in ukdale[kind]:
            external = (model.predict_proba(X.loc[t.test.index])[:, 1] >= threshold).astype(int)
            personal = np.zeros(len(t.test), dtype=int)
            if t.train.nunique() == 2:
                own = make_model("random_forest").fit(X.loc[t.train.index], t.train)
                th = best_threshold(t.val, own.predict_proba(X.loc[t.val.index])[:, 1])
                personal = (own.predict_proba(X.loc[t.test.index])[:, 1] >= th).astype(int)
            baseline = HourOfWeekBaseline().fit(t.train).tune_threshold(t.val).predict(t.test.index)
            for name, pred in (("REFIT model", external), ("personal", personal), ("baseline", baseline)):
                rows.append({"type": kind, "house": t.house, "column": t.column, "appliance": t.appliance,
                             "model": name, **{k: round(v, 3) for k, v in evaluate(t.test, pred).items()}})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "external.csv", index=False)
    wide = df.pivot_table(index=["type", "house", "column"], columns="model", values="f1").reset_index()
    pd.set_option("display.width", 200)
    print(f"\n{len(wide)} UK-DALE appliances in {wide['house'].nunique()} homes; F1 on their last 20%\n")
    print(wide.groupby("type")[["baseline", "personal", "REFIT model"]].median().round(3).to_string())
    print("\nmedian F1:", {m: round(wide[m].median(), 3) for m in ("baseline", "personal", "REFIT model")})


if __name__ == "__main__":
    main()
