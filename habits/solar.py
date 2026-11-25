"""Real historical solar output for the REFIT homes, from the EU's PVGIS service.

PVGIS (European Commission JRC) gives hourly PV output for any location and year,
computed from satellite irradiance (SARAH3) and ERA5 weather. We ask for a 1 kWp
system at Loughborough (where the REFIT homes are), facing south at the optimal
tilt, with 14% system losses, for 2013-2015, and scale it to the system size
being studied. Unlike the simulation's clear-sky curve, this includes the real
clouds of the very days the machines ran.

The data is fetched once and cached in data/raw/pvgis/. Times in PVGIS are UTC;
they are converted to local wall-clock time like the rest of the project.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

from habits.paths import DATASET, LOCATION, ROOT

PLACE, LAT, LON = LOCATION  # Loughborough for REFIT, London for UK-DALE (habits/paths.py)
YEARS = (2013, 2015) if DATASET == "refit" else (2012, 2017)
CACHE = ROOT / "data" / "raw" / "pvgis" / f"{PLACE.lower()}_{YEARS[0]}_{YEARS[1]}_1kwp.json"
URL = ("https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat={lat}&lon={lon}&startyear={y0}&endyear={y1}"
       "&pvcalculation=1&peakpower=1&loss=14&optimalangles=1&outputformat=json")


def fetch(cache: Path = CACHE) -> dict:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(lat=LAT, lon=LON, y0=YEARS[0], y1=YEARS[1]), timeout=120) as r:
            cache.write_bytes(r.read())
    return json.loads(cache.read_text())


def hourly_pv_w(kwp: float = 3.0, cache: Path = CACHE) -> pd.Series:
    """PV output in watts for a `kwp` system, per local hour (hour start, no time zone).

    PVGIS stamps each hour as e.g. "20140101:1011" = 10:11 UTC, the middle of
    the hour 10:00-11:00 UTC; the value is the average power over that hour.
    """
    rows = fetch(cache)["outputs"]["hourly"]
    utc = pd.to_datetime([r["time"] for r in rows], format="%Y%m%d:%H%M", utc=True).floor("h")
    local = utc.tz_convert("Europe/London").tz_localize(None)
    s = pd.Series([r["P"] * kwp for r in rows], index=local, name="pv_w")
    return s.groupby(level=0).mean()  # the autumn clock change makes one local hour appear twice
