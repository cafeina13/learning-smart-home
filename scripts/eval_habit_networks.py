"""The professor's GRU and CNN-GRU as habit models, against the random forest and the baseline.

    python scripts/eval_habit_networks.py [--weeks N] [--epochs N] [--max-tasks N]   (GPU: Kaggle)
    CAPSTONE_DATASET=ukdale python scripts/eval_habit_networks.py

Run 1 (default): 1-week window, at most 30 epochs -> habit_networks.csv.
Run 2, fixed before seeing its results: --weeks 4 --epochs 60, so the networks
get the same 4 weeks of history as the forest's features and the GRU is not cut
off by the epoch cap -> habit_networks_4w.csv.

Same tasks, split and scoring as scripts/eval_refit.py. Writes
habit_networks.csv next to habit_models.csv, one row per task as it finishes
(a rerun skips finished tasks), then compares with habit_models.csv if present.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from habits.baseline import evaluate  # noqa: E402
from habits.evaluate import appliance_type, tasks  # noqa: E402
from habits.paths import RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402
from habits import sequence  # noqa: E402
from habits.sequence import DEVICE, NETWORKS, fit_and_predict_sequence  # noqa: E402

OUT = RESULTS / "habit_networks.csv"


def main(max_tasks: int | None) -> None:
    print(f"window {sequence.WINDOW} h, at most {sequence.MAX_EPOCHS} epochs -> {OUT.name}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["house", "column", "model"])
    finished = set(zip(done["house"], done["column"], done["model"]))
    print(f"device: {DEVICE}; {len(finished)} results already in {OUT}", flush=True)
    for i, t in enumerate(tasks(*load_prepared())):
        if max_tasks is not None and i >= max_tasks:
            break
        for name in NETWORKS:
            if (t.house, t.column, name) in finished:
                continue
            start = time.perf_counter()
            predictions, details = fit_and_predict_sequence(name, t)
            row = {"house": t.house, "column": t.column, "appliance": t.appliance,
                   "type": appliance_type(t.appliance), "category": t.category, "model": name,
                   "test_hours": len(t.test), "epochs": details.get("epochs", 0),
                   **{k: round(details[k], 5) for k in ("best_epoch", "train_loss_best", "train_loss_last",
                                                       "val_loss_best", "val_loss_last", "lr_last") if k in details},
                   **{k: round(v, 3) for k, v in evaluate(t.test, predictions).items()}}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"task {i + 1}: house {t.house} {t.appliance:<24} {name:<8} F1 {row['f1']:.3f}  "
                  f"({row['epochs']} epochs, {time.perf_counter() - start:.0f} s)", flush=True)
    compare()


def compare() -> None:
    nets = pd.read_csv(OUT)
    others = RESULTS / "habit_models.csv"
    df = pd.concat([pd.read_csv(others), nets], ignore_index=True) if others.exists() else nets
    wide = df.pivot_table(index=["house", "column"], columns="model", values="f1").dropna()
    print(f"\n{len(wide)} tasks; F1 on the last 20% of each house")
    for m in wide.columns:
        print(f"  {m:<20} median F1 {wide[m].median():.3f}   mean F1 {wide[m].mean():.3f}")
    for net in NETWORKS:
        for ref in ("hour_of_week", "random_forest"):
            if net in wide and ref in wide:
                d = wide[net] - wide[ref]
                p_more = wilcoxon(wide[net], wide[ref], alternative="greater").pvalue
                p_less = wilcoxon(wide[net], wide[ref], alternative="less").pvalue
                print(f"  {net} vs {ref}: better on {(d > 0).sum()}, worse on {(d < 0).sum()} "
                      f"(one-sided Wilcoxon: better p = {p_more:.1e}, worse p = {p_less:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=sequence.MAX_EPOCHS)
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()
    sequence.WINDOW, sequence.MAX_EPOCHS = 168 * args.weeks, args.epochs
    if args.weeks != 1:
        OUT = RESULTS / f"habit_networks_{args.weeks}w.csv"
    main(args.max_tasks)
