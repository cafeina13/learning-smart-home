"""Turn the raw REFIT files (6.9 GB of 8-second power readings) into small tables, once.

For every house and every named appliance:

- idle_level_w: the typical draw when plugged in but not in use: the median of
  the small non-zero readings (below 15 W), if there are enough of them. Many
  REFIT appliances read exactly 0 W when idle (switched off at the wall); others
  idle at a few watts (house 5's desktop at ~9 W, its washing machine at ~3 W).
- on_threshold_w: above this the appliance counts as "in use":
  max(15 W, 3 x idle level + 10 W).
- standby_w: the AVERAGE power over all idle time, zeros included. This is
  exactly what a standby cut would save per idle hour, so it drives the
  standby-cut suggestions (a TV that is already switched off at the wall saves
  nothing, and gets no suggestion).
- uses: start, end, duration, energy of each use (habits/refit.py: extract_uses).

Plus the hours each house has trustworthy data for (covered hours), so missing
data is never mistaken for "not used".

Times are converted to local wall-clock time without a time zone, like the rest
of the project: a habit is "07:10", whatever the UTC offset that day.

Output (data/processed/refit/): uses.parquet, covered_hours.parquet,
appliances.parquet (one row per house x appliance with its thresholds and counts),
aggregate_hourly.parquet (each house's mean total power per local hour, which the
smart-start evaluation needs: the rest of the house uses solar first).
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from habits.dataset import covered_hours
from habits.paths import PROCESSED
from habits.refit import RAW_DIR, UseRules, appliance_names, category, extract_uses, house_path

OUT_DIR = PROCESSED  # data/processed/<dataset> (habits/paths.py)


def idle_level_w(values: np.ndarray, min_share: float = 0.01) -> float:
    small = values[(values > 0) & (values < 15)]
    return float(np.median(small)) if len(small) >= min_share * len(values) else 0.0


def use_rules_for(power: pd.Series, appliance_category: str = "habit") -> tuple[UseRules, float, float]:
    """(rules for extract_uses, idle level in W, average standby draw in W).

    Machines with programmes (washing machine, dishwasher, dryer) pause for many
    minutes inside one cycle (heat, wash, heat again), so pauses up to 30 minutes
    are bridged and anything under 10 minutes is not a cycle. Other appliances:
    pauses up to 2 minutes, at least 1 minute of use.
    """
    values = power.dropna().to_numpy()
    if len(values) == 0:
        return UseRules(), 0.0, 0.0
    idle = idle_level_w(values)
    threshold = max(15.0, 3 * idle + 10.0)
    idle_values = values[values < threshold]
    standby = float(idle_values.mean()) if len(idle_values) else 0.0
    if appliance_category == "shiftable":
        rules = UseRules(on_threshold_w=threshold, max_gap_s=1800.0, min_on_s=600.0)
    else:
        rules = UseRules(on_threshold_w=threshold)
    return rules, idle, standby


def read_house(house: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Raw house file, Issues rows dropped, index = UTC timestamps (kept tz-aware so
    durations stay right across daylight-saving changes)."""
    df = pd.read_csv(house_path(house, raw_dir), engine="pyarrow")
    if "Issues" in df.columns:
        df = df[df.pop("Issues") != 1]
    df.index = pd.to_datetime(df.pop("Unix"), unit="s", utc=True)
    df = df.drop(columns=[c for c in ("Time",) if c in df.columns]).astype("float32")
    return df.sort_index()


def local_naive(t: pd.Series | pd.DatetimeIndex):
    """UTC -> Europe/London wall-clock time, without a time zone."""
    if isinstance(t, pd.Series):
        return t.dt.tz_convert("Europe/London").dt.tz_localize(None)
    return t.tz_convert("Europe/London").tz_localize(None)


def aggregate_hourly(df: pd.DataFrame, house: int) -> pd.DataFrame:
    """Mean whole-house power per local hour (W)."""
    hourly = df["Aggregate"].resample("h").mean().dropna()
    hourly.index = local_naive(hourly.index)
    hourly = hourly.groupby(level=0).mean()  # the autumn clock change repeats one local hour
    return pd.DataFrame({"house": house, "hour": hourly.index, "aggregate_w": hourly.to_numpy()})


def appliance_uses(house: int, column: str, name: str, power: pd.Series, n_days: int) -> tuple[pd.DataFrame, dict]:
    """One appliance's uses and its row of the appliance table (power: UTC-indexed watts).

    Shared by REFIT and UK-DALE, so both datasets are prepared by identical rules."""
    rules, idle, standby = use_rules_for(power, category(name))
    uses = extract_uses(power, rules)
    if len(uses):  # an appliance that was never used has an empty, untyped table
        uses["start"], uses["end"] = local_naive(uses["start"]), local_naive(uses["end"])
    uses.insert(0, "appliance", name)
    uses.insert(0, "column", column)
    uses.insert(0, "house", house)
    info = {
        "house": house, "column": column, "appliance": name, "category": category(name),
        "idle_level_w": round(idle, 2), "standby_w": round(standby, 2), **asdict(rules),
        "uses": len(uses), "days": n_days, "uses_per_day": round(len(uses) / max(1, n_days), 2),
        "mean_energy_wh": round(float(uses["energy_wh"].mean()), 1) if len(uses) else 0.0,
    }
    return uses, info


def prepare_house(house: int, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    df = read_house(house, raw_dir)
    names = appliance_names(raw_dir).get(house, {})
    hours = covered_hours(local_naive(df.index))
    n_days = len({h.date() for h in hours})
    all_uses, info = [], []
    for column, name in names.items():
        if column not in df.columns:
            continue
        uses, row = appliance_uses(house, column, name, df[column], n_days)
        all_uses.append(uses)
        info.append(row)
    uses = pd.concat(all_uses, ignore_index=True) if all_uses else pd.DataFrame()
    covered = pd.DataFrame({"house": house, "hour": hours})
    return uses, covered, info, aggregate_hourly(df, house)


def load_aggregate(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "aggregate_hourly.parquet")


def load_prepared(out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(uses, covered hours, appliance table) as written by scripts/prepare_refit.py or prepare_ukdale.py."""
    return (pd.read_parquet(out_dir / "uses.parquet"),
            pd.read_parquet(out_dir / "covered_hours.parquet"),
            pd.read_parquet(out_dir / "appliances.parquet"))
