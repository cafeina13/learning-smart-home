"""Pack what Kaggle needs into one small zip: the code and the prepared REFIT tables.

    .venv\\Scripts\\python.exe scripts\\make_bundle.py

Writes data/bundle/capstone-bundle.zip (about 12 MB): every file tracked in git,
the prepared REFIT and UK-DALE tables (not the raw data), the cached PVGIS weather,
and the forecast results so far (the deep run needs each home's selected features).
Upload it to Kaggle as a private dataset named "capstone-bundle".
"""

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bundle" / "capstone-bundle.zip"
DATA = [
    "data/processed/refit/uses.parquet",
    "data/processed/refit/covered_hours.parquet",
    "data/processed/refit/appliances.parquet",
    "data/processed/refit/aggregate_hourly.parquet",
    "data/raw/pvgis/loughborough_2013_2015_1kwp.json",
    "data/results/forecast.csv",
    "data/results/forecast_features.csv",
    "data/results/habit_models.csv",  # the forest and baseline the habit networks are compared with
    "data/processed/ukdale/uses.parquet",
    "data/processed/ukdale/covered_hours.parquet",
    "data/processed/ukdale/appliances.parquet",
    "data/processed/ukdale/aggregate_hourly.parquet",
    "data/results/ukdale/habit_models.csv",
    "data/results/forecast_habits.csv",  # the trees, for the networks' comparison on the same hours
    "data/results/ukdale/forecast_features.csv",
    "data/results/ukdale/forecast_habits.csv",
]


def main() -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    missing = [f for f in DATA if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing (run the REFIT preparation and the classical forecast first): {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in tracked + DATA:
            zf.write(ROOT / f, f"capstone/{f}")
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(tracked) + len(DATA)} files)")


if __name__ == "__main__":
    main()
