"""Does the split decide who wins? Every model on our question, split by time and at random.

    python scripts/eval_split_effect.py [--max-tasks N]                  (Kaggle, CPU)
    python scripts/eval_split_effect.py --models GRU                     (Kaggle, GPU)
    CAPSTONE_DATASET=ukdale python scripts/eval_split_effect.py
    CAPSTONE_DATASET=casas  python scripts/eval_split_effect.py

"Will this appliance (CASAS: activity) start in this hour?" for the same tasks as
scripts/eval_refit.py, scored with F1 on the test hours, under two splits of the
same hours (60/20/20):

    time     first 60% train, next 20% validation, last 20% test (ours)
    random   the hours dealt at random, stratified (Gad et al.'s way)

Models: the hour-of-week baseline, our random forest, Gad et al.'s LightGBM
with their inputs (habits/gad.py: hourly_inputs), and with --models GRU the
professor's GRU (the networks' claim to beat trees is tested under both splits
too).

Writes split_effect.csv (the GRU: split_effect_gru.csv) in the dataset's results
folder, one row per task x split x model as it finishes (a rerun skips finished
ones); the summary reads both files.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from habits import gad  # noqa: E402
from habits.baseline import HourOfWeekBaseline, evaluate  # noqa: E402
from habits.evaluate import appliance_type, random_split, tasks  # noqa: E402
from habits.models import fit_and_predict  # noqa: E402
from habits.paths import RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402

OUT = RESULTS / "split_effect.csv"
TREES = ["hour_of_week", "random_forest", "lightgbm_gad"]
SPLITS = ["time", "random"]


def predict(name: str, task):
    if name == "hour_of_week":
        return HourOfWeekBaseline().fit(task.train).tune_threshold(task.val).predict(task.test.index), {}
    if name == "lightgbm_gad":
        return gad.fit_and_predict_hourly(task)
    if name == "GRU":
        from habits.sequence import fit_and_predict_sequence
        return fit_and_predict_sequence("GRU", task)
    return fit_and_predict(name, task)


def main(models: list[str], max_tasks: int | None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["house", "column", "split", "model"])
    finished = set(zip(done["house"], done["column"], done["split"], done["model"]))
    for i, t in enumerate(tasks(*load_prepared())):
        if max_tasks is not None and i >= max_tasks:
            break
        for split in SPLITS:
            task = t if split == "time" else random_split(t)
            for name in models:
                if (t.house, t.column, split, name) in finished:
                    continue
                start = time.perf_counter()
                predictions, _ = predict(name, task)
                row = {"house": t.house, "column": t.column, "appliance": t.appliance,
                       "type": appliance_type(t.appliance), "split": split, "model": name,
                       "test_hours": len(task.test), "seconds": round(time.perf_counter() - start, 1),
                       **{k: round(v, 3) for k, v in evaluate(task.test, predictions).items()}}
                pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"task {i + 1}: house {t.house} {t.appliance}", flush=True)
    summary()


def summary() -> None:
    df = pd.concat([pd.read_csv(f) for f in sorted(RESULTS.glob("split_effect*.csv"))], ignore_index=True)
    wide = df.pivot_table(index=["house", "column"], columns=["split", "model"], values="f1").dropna()
    print(f"\n{len(wide)} tasks; median F1 on the test hours (mean in brackets)\n")
    models = list(dict.fromkeys(df["model"]))
    print(f"  {'model':<20}" + "".join(f"{s:>18}" for s in SPLITS))
    for m in models:
        print(f"  {m:<20}" + "".join(f"{wide[(s, m)].median():>10.3f} ({wide[(s, m)].mean():.3f})" for s in SPLITS))
    print("\nRandom split minus time split, per model (paired over tasks):")
    for m in models:
        d = wide[("random", m)] - wide[("time", m)]
        p = wilcoxon(wide[("random", m)], wide[("time", m)], alternative="greater").pvalue if d.any() else 1.0
        print(f"  {m:<20} median {d.median():+.3f}, higher on {(d > 0).sum()} of {len(d)} tasks "
              f"(one-sided Wilcoxon p = {p:.1e})")
    print("\nWho wins under each split (paired over tasks):")
    for s in SPLITS:
        for a, b in (("lightgbm_gad", "random_forest"), ("random_forest", "hour_of_week"),
                     ("GRU", "random_forest")):
            if (s, a) in wide and (s, b) in wide:
                d = wide[(s, a)] - wide[(s, b)]
                p_more = wilcoxon(wide[(s, a)], wide[(s, b)], alternative="greater").pvalue
                p_less = wilcoxon(wide[(s, a)], wide[(s, b)], alternative="less").pvalue
                print(f"  {s:<6} {a} vs {b}: better on {(d > 0).sum()}, worse on {(d < 0).sum()} "
                      f"(better p = {p_more:.1e}, worse p = {p_less:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(TREES), help="comma-separated; GRU needs a GPU")
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()
    chosen = args.models.split(",")
    if chosen == ["GRU"]:
        OUT = RESULTS / "split_effect_gru.csv"
    main(chosen, args.max_tasks)
