"""Prepare UK-DALE (Kelly & Knottenbelt, 2015, CC BY 4.0) into the same tables as REFIT.

UK-DALE 2017 "disaggregated" release, ukdale.zip (3.6 GB), 5 homes in London:

    house_N/labels.dat         "<channel> <label>" per line, e.g. "10 kettle"
    house_N/channel_M.dat      "<unix time> <watts>" per line, about every 6 s
                               (channel 1 is the whole house)

The files are read straight from the zip, one channel at a time, and go through
exactly the same steps as REFIT (habits/prepare.py: per-appliance thresholds,
standby, cycle bridging), so the two datasets are prepared by identical rules.
Labels are turned into names the rest of the project understands
("dish_washer" -> "Dishwasher", "tv" -> "Television").

Run on a machine with the zip (it is meant for Kaggle, where the download is fast):

    CAPSTONE_DATASET=ukdale python scripts/prepare_ukdale.py /path/to/ukdale.zip
"""

import re
import zipfile

import pandas as pd

from habits.dataset import covered_hours
from habits.prepare import aggregate_hourly, appliance_uses, local_naive

MAINS = ("aggregate", "mains")
RENAME = {
    "dish washer": "Dishwasher", "dishwasher": "Dishwasher",
    "washing machine": "Washing Machine", "washer dryer": "Washer Dryer",
    "tumble dryer": "Tumble Dryer", "tv": "Television", "television": "Television",
    "fridge freezer": "Fridge-Freezer", "fridge": "Fridge", "freezer": "Freezer",
    "kettle": "Kettle", "toaster": "Toaster", "microwave": "Microwave",
}


def appliance_name(label: str) -> str:
    """'dish_washer' -> 'Dishwasher', 'htpc' -> 'Htpc' (unknown labels keep their own words)."""
    words = label.replace("_", " ").strip().lower()
    return RENAME.get(words, words.title())


def houses(zf: zipfile.ZipFile) -> list[int]:
    found = {int(m.group(1)) for n in zf.namelist() if (m := re.search(r"house_(\d+)/labels\.dat$", n))}
    return sorted(found)


def labels(zf: zipfile.ZipFile, house: int) -> dict[int, str]:
    path = next(n for n in zf.namelist() if n.endswith(f"house_{house}/labels.dat"))
    out = {}
    for line in zf.read(path).decode().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1].strip()
    return out


def read_channel(zf: zipfile.ZipFile, house: int, channel: int) -> pd.Series:
    """Watts indexed by UTC time."""
    path = next(n for n in zf.namelist() if n.endswith(f"house_{house}/channel_{channel}.dat"))
    with zf.open(path) as f:
        df = pd.read_csv(f, sep=" ", header=None, names=["unix", "watts"], usecols=[0, 1],
                         dtype={"unix": "int64", "watts": "float32"})
    s = pd.Series(df["watts"].to_numpy(), index=pd.to_datetime(df["unix"], unit="s", utc=True))
    return s[~s.index.duplicated()].sort_index()


def prepare_house(zf: zipfile.ZipFile, house: int, log=print):
    """(uses, covered hours, appliance rows, hourly aggregate) for one UK-DALE home."""
    names = labels(zf, house)
    mains_channel = next((c for c, l in names.items() if l.lower() in MAINS), 1)
    mains = read_channel(zf, house, mains_channel)
    hours = covered_hours(local_naive(mains.index), min_samples=300)  # 300 x ~6 s = 30 of 60 minutes
    n_days = len({h.date() for h in hours})
    agg = aggregate_hourly(pd.DataFrame({"Aggregate": mains}), house)
    all_uses, rows = [], []
    for channel, label in sorted(names.items()):
        if channel == mains_channel or label.lower() in MAINS:
            continue
        name = appliance_name(label)
        uses, row = appliance_uses(house, f"channel_{channel}", name, read_channel(zf, house, channel), n_days)
        row["label"] = label
        all_uses.append(uses)
        rows.append(row)
        log(f"  house {house} channel {channel:>2} {label:<24} -> {name:<18} {row['category']:<9} "
            f"{row['uses']:>6} uses, standby {row['standby_w']:.1f} W")
    uses = pd.concat(all_uses, ignore_index=True) if all_uses else pd.DataFrame()
    return uses, pd.DataFrame({"house": house, "hour": hours}), rows, agg
