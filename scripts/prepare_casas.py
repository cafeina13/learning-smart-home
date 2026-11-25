"""Prepare the CASAS homes of Gad et al. (2025) from labeled_data.zip, once (run on Kaggle).

    wget -O /kaggle/temp/casas.zip "<habits.casas.URL>"
    CAPSTONE_DATASET=casas python scripts/prepare_casas.py /kaggle/temp/casas.zip

Writes data/processed/casas/: uses.parquet, covered_hours.parquet and
appliances.parquet (our pipeline's tables: every habit script then runs on
CASAS), and intervals.parquet (Gad et al.'s labelled sensor intervals).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from habits import casas  # noqa: E402
from habits.paths import DATASET, PROCESSED  # noqa: E402


def main(zip_path: str) -> None:
    if DATASET != "casas":
        raise SystemExit("run with CAPSTONE_DATASET=casas")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    uses, covered, rows, spans = [], [], [], []
    for home, events in casas.read_zip(zip_path):
        labels = casas.label_events(events)
        u, c, r = casas.tables(home, events, labels)
        s = casas.intervals(events, labels)
        s.insert(0, "house", casas.house_number(home))
        uses.append(u), covered.append(c), rows.extend(r), spans.append(s)
        labelled = s["activity"].notna() & ~s["activity"].isin(casas.NOISY)
        print(f"{home}: {len(events):>8} events, {events['time'].min():%Y-%m-%d} to {events['time'].max():%Y-%m-%d}, "
              f"{len(c) // 24} recorded days, {len(s):>7} intervals ({labelled.sum()} labelled), "
              f"{sum(x['category'] == 'habit' for x in r)} activities", flush=True)
    pd.concat(uses, ignore_index=True).to_parquet(PROCESSED / "uses.parquet")
    pd.concat(covered, ignore_index=True).to_parquet(PROCESSED / "covered_hours.parquet")
    pd.DataFrame(rows).to_parquet(PROCESSED / "appliances.parquet")
    pd.concat(spans, ignore_index=True).to_parquet(PROCESSED / "intervals.parquet")
    print(f"saved to {PROCESSED}")


if __name__ == "__main__":
    main(sys.argv[1])
