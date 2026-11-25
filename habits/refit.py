"""Load the REFIT dataset and turn appliance power into switch-on events.

REFIT (Murray et al., 2017, CC BY 4.0): 20 UK homes (numbered 1-21, no house
14), Oct 2013 - Jun 2015. Each cleaned house file, CLEAN_House<N>.csv, has one
row every 6-8 seconds:

    Time, Unix, Aggregate, Appliance1, ..., Appliance9, Issues

Power is in watts. Which appliance sits in which column differs per house; the
names are parsed from the dataset's readme (see appliance_names). Issues = 1
marks rows where the appliances add up to more than the whole house; those rows
are dropped.

A power trace becomes "uses" of the appliance:
  1. on   = power above a threshold (standby draw stays below it)
  2. gaps shorter than `max_gap_s` are bridged (a washing machine pauses mid-cycle)
  3. runs shorter than `min_on_s` are dropped (sensor blips)
Each use has a start, an end, and the energy it consumed. The start is what a
habit is made of: "kettle switched on around 07:10 on weekdays".
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "refit"
README = "CLEAN_READ_ME_081116.txt"
HOUSES = [h for h in range(1, 22) if h != 14]

# Appliances that switch themselves on and off (compressors, always-on loads).
# Their cycles are not human habits, so habit learning skips them.
AUTOMATIC = ("fridge", "freezer", "router", "network", "vivarium", "pump", "dehumidifier",
             "boiler", "modem", "server", "alarm", "smoke")  # the last five: UK-DALE's always-on devices
# Appliances whose run can be moved to another time without bother: the energy
# layer may suggest running these when the solar panel produces.
SHIFTABLE = ("washing machine", "washer dryer", "dishwasher", "tumble dryer")


@cache
def appliance_names(raw_dir: Path = RAW_DIR) -> dict[int, dict[str, str]]:
    """{house: {"Appliance1": "Fridge", ...}} parsed from the REFIT readme.

    The readme lists each house as "House N" followed by lines like
    "4.Washing Machine(1), Servis, 6065"; the name is the text before the
    first comma. Unknown appliances ("???") keep their column name.
    """
    names: dict[int, dict[str, str]] = {}
    house = None
    for line in (raw_dir / README).read_text(encoding="latin-1").splitlines():
        if m := re.match(r"House (\d+)\s*$", line.strip()):
            house = int(m.group(1))
            names[house] = {}
        elif house is not None and (m := re.match(r"([1-9])\.([^,]+)", line.strip())):
            name = m.group(2).strip()
            if "?" not in name:
                names[house][f"Appliance{m.group(1)}"] = name
    return names


def category(name: str) -> str:
    """'automatic', 'shiftable', or 'habit' (switched on by a person)."""
    lower = name.lower()
    if any(word in lower for word in AUTOMATIC):
        return "automatic"
    if any(word in lower for word in SHIFTABLE):
        return "shiftable"
    return "habit"


@dataclass
class UseRules:
    on_threshold_w: float = 15.0
    max_gap_s: float = 120.0
    min_on_s: float = 60.0


def house_path(house: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"CLEAN_House{house}.csv"


def load_house(house: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One house as a DataFrame indexed by local time, one float32 column per channel."""
    df = pd.read_csv(house_path(house, raw_dir))
    df = df[df.pop("Issues") != 1]
    df.index = pd.to_datetime(df.pop("Unix"), unit="s", utc=True).dt.tz_convert("Europe/London")
    df = df.drop(columns=["Time"]).astype("float32")
    return df.rename(columns=appliance_names(raw_dir).get(house, {}))


def extract_uses(power: pd.Series, rules: UseRules = UseRules()) -> pd.DataFrame:
    """Turn one appliance's power trace into a table of uses: start, end, duration_s, energy_wh."""
    power = power.dropna()
    if power.empty:
        return pd.DataFrame(columns=["start", "end", "duration_s", "energy_wh"])

    t = power.index
    seconds = (t - t[0]).total_seconds().to_numpy()
    on = power.to_numpy() > rules.on_threshold_w

    # Sample i "belongs" to the interval until the next sample (capped, so a
    # long data gap does not count as hours of use).
    dt = np.diff(seconds, append=seconds[-1] + 8.0)
    dt = np.minimum(dt, 30.0)

    # Find runs of consecutive "on" samples.
    edges = np.diff(on.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive

    runs = []
    for s, e in zip(starts, ends):
        if runs and seconds[s] - (seconds[runs[-1][1] - 1] + dt[runs[-1][1] - 1]) <= rules.max_gap_s:
            runs[-1][1] = e  # bridge a short pause
        else:
            runs.append([s, e])

    rows = []
    values = power.to_numpy()
    for s, e in runs:
        duration = seconds[e - 1] + dt[e - 1] - seconds[s]
        if duration < rules.min_on_s:
            continue
        energy_wh = float(np.sum(values[s:e] * dt[s:e]) / 3600.0)
        rows.append((t[s], t[e - 1], duration, energy_wh))
    return pd.DataFrame(rows, columns=["start", "end", "duration_s", "energy_wh"])
