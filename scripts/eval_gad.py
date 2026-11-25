"""Replicate Gad et al. (2025) on their data (CASAS), then change one thing at a time.

    CAPSTONE_DATASET=casas python scripts/eval_gad.py [--homes N]   (Kaggle; needs prepare_casas.py)

Runs, per home (habits/gad.py has the details):

    paper  config 3   LightGBM, XGBoost, CatBoost   their Table 6
    paper  config 1-2 LightGBM                      their feature ablation
    time   config 3   LightGBM                      the same samples, split by time
    strict config 3   LightGBM                      split by time, only what a home knows

each with three baselines on the same test intervals: the training majority
activity, the previous interval's activity, and the 21-day history vote.

Writes data/results/casas/gad_replication.csv, one row per home x run x model
as it finishes (a rerun skips finished ones), then prints the summary.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from habits import gad  # noqa: E402
from habits.paths import DATASET, PROCESSED, RESULTS  # noqa: E402

OUT = RESULTS / "gad_replication.csv"
RUNS = [("paper", 3, gad.MODELS), ("paper", 1, ["lightgbm"]), ("paper", 2, ["lightgbm"]),
        ("time", 3, ["lightgbm"]), ("strict", 3, ["lightgbm"])]
PAPER_TABLE_6 = {"xgboost": 0.887, "catboost": 0.905, "lightgbm": 0.915}  # mean test accuracy, config 3


def main(max_homes: int | None) -> None:
    if DATASET != "casas":
        raise SystemExit("run with CAPSTONE_DATASET=casas")
    iv = pd.read_parquet(PROCESSED / "intervals.parquet")
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["house", "variant", "config"])
    finished = set(zip(done["house"], done["variant"], done["config"]))
    homes = sorted(iv["house"].unique())[:max_homes]
    for i, house in enumerate(homes):
        home = iv[iv["house"] == house]
        for variant, config, models in RUNS:
            if (house, variant, config) in finished:
                continue
            start = time.perf_counter()
            rows = gad.run_home(home, config, variant, models)
            df = pd.DataFrame([{"house": house, "variant": variant, "config": config, **r} for r in rows])
            df.round(4).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            best = df.set_index("model")["accuracy"]
            print(f"home {i + 1}/{len(homes)} hh{house} {variant:<6} config {config}: "
                  + "  ".join(f"{m} {a:.3f}" for m, a in best.items())
                  + f"  ({df['test'].iat[0]} test intervals, {time.perf_counter() - start:.0f} s)", flush=True)
    summary()


def summary() -> None:
    df = pd.read_csv(OUT)
    pd.set_option("display.width", 200)
    n = df["house"].nunique()
    print(f"\n{n} homes. Test accuracy, mean +- sd over homes (weighted F1 and macro F1 means after /)\n")
    table = (df.groupby(["variant", "config", "model"])
               .agg(accuracy=("accuracy", "mean"), sd=("accuracy", "std"), f1_weighted=("f1_weighted", "mean"),
                    f1_macro=("f1_macro", "mean"), homes=("house", "nunique")).round(3))
    print(table.to_string())
    paper = df[(df["variant"] == "paper") & (df["config"] == 3)].set_index("model")
    print("\nAgainst the paper's Table 6 (config 3, mean test accuracy):")
    for m, theirs in PAPER_TABLE_6.items():
        if m in paper.index:
            print(f"  {m:<9} paper {theirs:.3f}   ours {paper.loc[[m], 'accuracy'].mean():.3f}")
    wide = df[df["config"] == 3].pivot_table(index="house", columns=["variant", "model"], values="accuracy")
    print("\nOne change at a time (LightGBM, config 3), paired over homes:")
    for a, b in (("paper", "time"), ("time", "strict")):
        if ("lightgbm" in wide.get(a, {})) and ("lightgbm" in wide.get(b, {})):
            x, y = wide[(a, "lightgbm")].dropna(), wide[(b, "lightgbm")].dropna()
            both = x.index.intersection(y.index)
            p = wilcoxon(x[both], y[both], alternative="greater").pvalue
            print(f"  {a} -> {b}: {x[both].mean():.3f} -> {y[both].mean():.3f} (one-sided Wilcoxon p = {p:.1e})")
    for variant in ("paper", "time", "strict"):
        if (variant, "lightgbm") in wide and (variant, "history_vote") in wide:
            d = (wide[(variant, "lightgbm")] - wide[(variant, "previous_activity")]).dropna()
            print(f"  {variant}: LightGBM minus 'previous activity' baseline: mean {d.mean():+.3f}, "
                  f"better on {(d > 0).sum()} of {len(d)} homes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--homes", type=int, default=None)
    main(ap.parse_args().homes)
