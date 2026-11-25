"""Prepare UK-DALE into data/processed/ukdale/ (same tables as REFIT).

    CAPSTONE_DATASET=ukdale python scripts/prepare_ukdale.py path/to/ukdale.zip [houses...]

Afterwards every experiment script runs on UK-DALE with CAPSTONE_DATASET=ukdale.
"""

import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CAPSTONE_DATASET", "ukdale")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from habits.paths import DATASET, PROCESSED  # noqa: E402
from habits.ukdale import houses, prepare_house  # noqa: E402


def main(zip_path: str, only: list[int]) -> None:
    assert DATASET == "ukdale", "run with CAPSTONE_DATASET=ukdale"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    uses, covered, rows, aggregate = [], [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        for house in only or houses(zf):
            t = time.perf_counter()
            u, c, r, a = prepare_house(zf, house)
            uses.append(u)
            covered.append(c)
            rows += r
            aggregate.append(a)
            print(f"house {house}: {len(c)} covered hours, {len(u)} uses ({time.perf_counter() - t:.0f} s)", flush=True)
    pd.concat(uses, ignore_index=True).to_parquet(PROCESSED / "uses.parquet")
    pd.concat(covered, ignore_index=True).to_parquet(PROCESSED / "covered_hours.parquet")
    pd.DataFrame(rows).to_parquet(PROCESSED / "appliances.parquet")
    pd.concat(aggregate, ignore_index=True).to_parquet(PROCESSED / "aggregate_hourly.parquet")
    print(f"saved to {PROCESSED}")


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]])
