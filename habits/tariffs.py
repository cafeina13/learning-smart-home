"""Electricity prices for cost-aware smart start: three UK tariffs from one supplier and year.

All prices are Octopus Energy's for 2019 (pence per kWh, VAT included), read from its public API
(https://api.octopus.energy/v1/products/), so the three tariffs are compared on equal terms:

  flat       Flexible Octopus (VAR-19-04-12), one price at every hour
  economy7   the same product's Economy 7 version: a night rate from 00:30 to 07:30 UTC (01:30 to
             08:30 in British summer time), the window of Octopus's smart meters, and a day rate otherwise
  agile      Agile Octopus (AGILE-18-02-21): a new price every half hour, published the day before

Export, for a home with a panel: Outgoing Octopus fixed (OUTGOING-FIX-12M-19-05-13), 5.5 p/kWh. Solar a
machine uses is solar the home does not export, so it is costed at this price, not as free.

The datasets are from 2012-2017 and Agile began in 2018, so the 2019 prices are replayed onto the
recording's calendar: each recorded day takes the 2019 day in the same week of the year and on the same
weekday. Agile's half-hourly prices are fetched once and cached in data/raw/tariffs/.

Times follow the rest of the project: UK wall-clock time without a time zone.
"""

import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from habits.paths import DATASET, ROOT

YEAR = 2019
REGION = "B" if DATASET == "refit" else "C"  # Octopus regions: B = East Midlands (REFIT), C = London (UK-DALE)
CACHE = ROOT / "data" / "raw" / "tariffs"
AGILE_URL = ("https://api.octopus.energy/v1/products/AGILE-18-02-21/electricity-tariffs/"
             "E-1R-AGILE-18-02-21-{region}/standard-unit-rates/"
             "?period_from={year}-01-01T00:00Z&period_to={next}-01-01T00:00Z&page_size=1500")

# p/kWh incl. VAT, valid from 2019-04-12 (Flexible Octopus April 2019 v1), by region.
FLAT = {"B": 15.162, "C": 14.9415}
E7_DAY = {"B": 16.884, "C": 16.9995}
E7_NIGHT = {"B": 9.4395, "C": 9.4395}
EXPORT = 5.5  # Outgoing Octopus fixed, from 2019-05-16
E7_NIGHT_UTC = (30, 450)  # minutes after midnight UTC: 00:30 to 07:30

TARIFFS = ("flat", "economy7", "agile")


def fetch_agile(region: str = REGION, year: int = YEAR, cache: Path = CACHE) -> list[dict]:
    """Every half-hourly Agile price of the year for one region, from the API once, then from the cache."""
    path = cache / f"agile_{year}_{region}.json"
    if not path.exists():
        rows, url = [], AGILE_URL.format(region=region, year=year, next=year + 1)
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                page = json.loads(r.read())
            rows += page["results"]
            url = page["next"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    return json.loads(path.read_text())


def agile_local(region: str = REGION, year: int = YEAR, cache: Path = CACHE) -> pd.Series:
    """Agile prices on a complete half-hourly grid of local wall-clock time for that year."""
    rows = fetch_agile(region, year, cache)
    utc = pd.to_datetime([r["valid_from"] for r in rows], utc=True)
    local = utc.tz_convert("Europe/London").tz_localize(None)
    prices = pd.Series([r["value_inc_vat"] for r in rows], index=local).sort_index()
    prices = prices.groupby(level=0).mean()  # the autumn clock change repeats one local hour
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="30min", inclusive="left")
    return prices.reindex(grid).ffill().bfill()  # the spring change skips one: carry the last price


def same_weekday_in(day: date, year: int = YEAR) -> date:
    """The day of `year` in the same week of the year and on the same weekday as `day`."""
    target = date(year, day.month, min(day.day, 28 if day.month == 2 else day.day))
    moved = day + timedelta(days=7 * round((target - day).days / 7))
    while moved.year < year:  # near New Year the nearest same weekday can fall outside the year
        moved += timedelta(days=7)
    while moved.year > year:
        moved -= timedelta(days=7)
    return moved


def is_e7_night(times: pd.DatetimeIndex) -> np.ndarray:
    """True for local wall-clock times inside the 00:30-07:30 UTC night window."""
    # The repeated autumn hour is read as summer time; the skipped spring hour never occurs in the data.
    aware = times.tz_localize("Europe/London", ambiguous=np.ones(len(times), bool), nonexistent="shift_forward")
    utc = aware.tz_convert("UTC")
    minutes = np.asarray(utc.hour * 60 + utc.minute)
    return (minutes >= E7_NIGHT_UTC[0]) & (minutes < E7_NIGHT_UTC[1])


class Prices:
    """Import price (p/kWh) of one tariff at any local time of the recording."""

    def __init__(self, tariff: str, region: str = REGION, cache: Path = CACHE):
        if tariff not in TARIFFS:
            raise ValueError(f"unknown tariff {tariff!r}; one of {TARIFFS}")
        self.tariff, self.region = tariff, region
        self._agile = agile_local(region, YEAR, cache) if tariff == "agile" else None
        self._offsets: dict[date, timedelta] = {}

    def at(self, times) -> np.ndarray:
        times = pd.DatetimeIndex(times)
        if self.tariff == "flat":
            return np.full(len(times), FLAT[self.region])
        if self.tariff == "economy7":
            return np.where(is_e7_night(times), E7_NIGHT[self.region], E7_DAY[self.region])
        # Agile: move each time to the same weekday of 2019, then read its half-hour's price.
        days, which = np.unique(times.normalize().to_numpy(), return_inverse=True)
        shift = np.array([self._offset(pd.Timestamp(d).date()) for d in days], dtype="timedelta64[ns]")
        moved = (times + pd.to_timedelta(shift[which.ravel()])).floor("30min")
        pos = self._agile.index.get_indexer(moved)
        return np.where(pos >= 0, self._agile.to_numpy()[np.maximum(pos, 0)], np.nan)

    def __call__(self, t) -> float:
        """The price at one time, for the live scheduler (home/scheduler.py), which asks step by step."""
        if self.tariff == "flat":
            return FLAT[self.region]
        if self.tariff == "economy7":
            utc = pd.Timestamp(t).tz_localize("Europe/London", ambiguous=True, nonexistent="shift_forward")
            minutes = utc.tz_convert("UTC").hour * 60 + utc.tz_convert("UTC").minute
            night = E7_NIGHT_UTC[0] <= minutes < E7_NIGHT_UTC[1]
            return E7_NIGHT[self.region] if night else E7_DAY[self.region]
        moved = (pd.Timestamp(t) + self._offset(pd.Timestamp(t).date())).floor("30min")
        return float(self._agile.get(moved, np.nan))

    def _offset(self, day: date) -> timedelta:
        if day not in self._offsets:
            self._offsets[day] = same_weekday_in(day) - day
        return self._offsets[day]
