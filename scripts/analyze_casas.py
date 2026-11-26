"""CASAS described, for the dataset analysis (scripts/analyze_data.py): run where the prepared CASAS tables are (Kaggle).

    CAPSTONE_DATASET=casas python scripts/analyze_casas.py

The CASAS tables are made on Kaggle (scripts/prepare_casas.py; the raw data is
236 MB). This reads them and writes a small summary, data/results/casas/
casas_summary.json, that scripts/analyze_data.py turns into the CASAS section:

    starts      when each common activity starts, weekday x hour, all homes
    per_day     how often each activity starts per recorded day (median over homes)
    same_label  per home: the share of labelled sensor intervals whose activity is
                the same as the previous labelled interval's (why Gad et al.'s
                random split, and the "previous activity" rule, score so high),
                and the median run of consecutive same-activity intervals
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.casas import NOISY  # noqa: E402
from habits.paths import DATASET, PROCESSED, RESULTS  # noqa: E402

COMMON = 6  # activities shown as heatmaps: the ones most homes have


def main() -> None:
    if DATASET != "casas":
        raise SystemExit("run with CAPSTONE_DATASET=casas")
    uses = pd.read_parquet(PROCESSED / "uses.parquet")
    apps = pd.read_parquet(PROCESSED / "appliances.parquet")
    intervals = pd.read_parquet(PROCESSED / "intervals.parquet")

    habits = apps[(apps["category"] == "habit") & ~apps["column"].isin(NOISY)]
    starts = uses.merge(habits[["house", "column"]], on=["house", "column"])
    homes = habits.groupby("column")["house"].nunique()
    common = (habits.groupby("column")["uses"].sum()[homes >= 15]).sort_values(ascending=False).index[:COMMON]
    grids = {}
    for a in common:
        s = starts.loc[starts["column"] == a, "start"]
        g = np.zeros((7, 24), dtype=int)
        np.add.at(g, (s.dt.dayofweek.to_numpy(), s.dt.hour.to_numpy()), 1)
        grids[a] = {"homes": int(homes[a]), "starts": int(len(s)), "grid": g.tolist()}

    per_day = habits[habits["column"].isin(homes[homes >= 10].index)].groupby("column")["uses_per_day"].median()

    same = []
    for house, iv in intervals.groupby("house"):
        lab = iv["activity"]
        lab = lab[lab.notna() & ~lab.isin(NOISY)].reset_index(drop=True)
        repeat = lab.eq(lab.shift(1)).iloc[1:]
        runs = (lab != lab.shift(1)).cumsum().value_counts()
        same.append({"house": int(house), "labelled": int(len(lab)), "same_as_previous": float(repeat.mean()),
                     "median_run": float(runs.median()), "mean_run": float(runs.mean())})

    out = RESULTS / "casas_summary.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"starts": grids, "per_day": per_day.round(2).to_dict(),
                               "per_day_homes": homes[per_day.index].astype(int).to_dict(), "same_label": same},
                              indent=1), encoding="utf-8")
    s = pd.DataFrame(same)
    print(f"common activities: {', '.join(common)}")
    print(f"same activity as the previous interval: median {s['same_as_previous'].median():.1%} over "
          f"{len(s)} homes (range {s['same_as_previous'].min():.1%}-{s['same_as_previous'].max():.1%}); "
          f"median run {s['median_run'].median():.0f} intervals")
    print(f"written {out}")


if __name__ == "__main__":
    main()
