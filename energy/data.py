"""The forecasting data: one row per hour per home, and the 144-hour input windows.

Yaprakdal (2023), "An Ensemble Deep-Learning-Based Model for Hour-Ahead Load
Forecasting with a Feature Selection Approach", Energies 16(1):57, forecasts the
next hour's load from the previous six days (144 hourly steps) of every feature.
Her two datasets are national/regional (Czech Republic, Australia) with calendar
and weather features. Here the same set-up is applied to single homes (REFIT),
which her paper notes are much more volatile than aggregated load.

Features per hour (her paper's kinds, our sources):
    load         the home's mean power that hour (REFIT aggregate), watts
    year, month, day, hour    calendar
    temperature  2 m air temperature at Loughborough (ERA5, via PVGIS)
    irradiance   solar irradiance on a tilted plane (SARAH3, via PVGIS)
    wind         10 m wind speed (ERA5, via PVGIS)

Preprocessing as in the paper: month, day and hour are divided by their maximum
and 1 is added; every other feature is z-scored (mean and standard deviation from
the training data only).

Hourly means above MAX_LOAD_W are treated as missing: a UK home's supply cannot
deliver that (house 3 shows a 34 kW hour, a logging glitch).
"""

import numpy as np
import pandas as pd

from habits.prepare import load_aggregate
from habits.solar import fetch

WINDOW = 144  # six days of hourly steps, as in the paper
MAX_LOAD_W = 15_000
FEATURES = ["load", "year", "month", "day", "hour", "temperature", "irradiance", "wind"]
CALENDAR = {"month": 12, "day": 31, "hour": 23}  # divided by these, plus 1


def weather() -> pd.DataFrame:
    """Hourly weather at Loughborough 2013-2015 (local wall-clock hours)."""
    rows = fetch()["outputs"]["hourly"]
    utc = pd.to_datetime([r["time"] for r in rows], format="%Y%m%d:%H%M", utc=True).floor("h")
    local = utc.tz_convert("Europe/London").tz_localize(None)
    df = pd.DataFrame({"temperature": [r["T2m"] for r in rows], "irradiance": [r["G(i)"] for r in rows],
                       "wind": [r["WS10m"] for r in rows]}, index=local)
    return df.groupby(level=0).mean()


def house_hours(house: int, aggregate: pd.DataFrame | None = None, wx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every calendar hour of the home's recording, features as columns; load is NaN where missing."""
    aggregate = load_aggregate() if aggregate is None else aggregate
    wx = weather() if wx is None else wx
    load = aggregate.loc[aggregate["house"] == house].set_index("hour")["aggregate_w"]
    load = load.where(load <= MAX_LOAD_W)
    grid = pd.date_range(load.index.min(), load.index.max(), freq="h")
    df = pd.DataFrame({"load": load.reindex(grid)}, index=grid)
    df["year"], df["month"], df["day"], df["hour"] = grid.year, grid.month, grid.day, grid.hour
    return df.join(wx.reindex(grid))


def windows(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """(X, y, target hours): X[i] is the 144 hours before target hour i (shape 144 x features),
    y[i] that hour's load. Only targets whose window and own value are complete are kept."""
    values = df[features].to_numpy(dtype="float32")
    load = df["load"].to_numpy(dtype="float32")
    complete = ~np.isnan(values).any(axis=1) & ~np.isnan(load)
    # A window is usable if all WINDOW hours before the target and the target itself are complete.
    ok = np.convolve(complete.astype(int), np.ones(WINDOW + 1, dtype=int), mode="full")[WINDOW:len(complete)]
    targets = np.flatnonzero(ok == WINDOW + 1) + WINDOW
    X = np.stack([values[t - WINDOW:t] for t in targets]) if len(targets) else np.empty((0, WINDOW, len(features)))
    return X, load[targets], df.index[targets]


class Scaler:
    """The paper's preprocessing: calendar features / max + 1; others z-scored on training data."""

    def __init__(self, features: list[str]):
        self.features = features
        self.mean, self.std = {}, {}

    def fit(self, X: np.ndarray) -> "Scaler":
        for j, f in enumerate(self.features):
            if f not in CALENDAR:
                self.mean[f] = float(X[:, :, j].mean())
                self.std[f] = float(X[:, :, j].std()) or 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j, f in enumerate(self.features):
            X[:, :, j] = X[:, :, j] / CALENDAR[f] + 1 if f in CALENDAR else (X[:, :, j] - self.mean[f]) / self.std[f]
        return X
