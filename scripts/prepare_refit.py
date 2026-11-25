"""Parse every REFIT house once and save the small tables the experiments use.

    .venv\\Scripts\\python.exe scripts\\prepare_refit.py            # all 20 houses
    .venv\\Scripts\\python.exe scripts\\prepare_refit.py 3 5        # only houses 3 and 5

Takes a few minutes for all houses; afterwards everything reads data/processed/refit/.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from habits.prepare import OUT_DIR, prepare_house  # noqa: E402
from habits.refit import HOUSES  # noqa: E402


def main(houses: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uses, covered, info, aggregate = [], [], [], []
    for house in houses:
        t = time.perf_counter()
        u, c, i, a = prepare_house(house)
        uses.append(u)
        covered.append(c)
        info += i
        aggregate.append(a)
        print(f"house {house:>2}: {len(c):>6} covered hours, {len(u):>6} uses  ({time.perf_counter() - t:.0f} s)",
              flush=True)
    pd.concat(uses, ignore_index=True).to_parquet(OUT_DIR / "uses.parquet")
    pd.concat(covered, ignore_index=True).to_parquet(OUT_DIR / "covered_hours.parquet")
    pd.DataFrame(info).to_parquet(OUT_DIR / "appliances.parquet")
    pd.concat(aggregate, ignore_index=True).to_parquet(OUT_DIR / "aggregate_hourly.parquet")
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or HOUSES)
