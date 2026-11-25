"""Features for "will this appliance be switched on in this hour?", from past hours only.

Every feature for hour h uses information from before h, so nothing from the
hour being predicted (or later) leaks in.

time         hour of day (as sine/cosine, so 23:00 is next to 00:00), day of
             week (sine/cosine), weekend flag
own history  used in the previous hour, the same hour yesterday, the same hour
             last week; uses in the last 24 h and last 7 days; hours since the
             last use (capped at one week); recent rate at this hour of the week
             (share of the last 4 weeks with a use at this hour: an adaptive
             version of the hour-of-week baseline)
household    switch-ons of the house's OTHER appliances in the previous 1 and 3
             hours: a proxy for "someone is home and awake"
data gaps    whether the previous hour had data at all (lags into a gap read 0)

Hours are laid out on a full calendar grid first, so "24 hours ago" really is
24 hours ago even when the data has gaps; only covered hours are returned.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend",
    "lag_1h", "lag_24h", "lag_168h", "uses_24h", "uses_7d", "hours_since_use", "recent_how_rate",
    "others_1h", "others_3h", "prev_hour_covered",
]


def _grid(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(hours.min(), hours.max(), freq="h")


def build_features(labels: pd.Series, other_starts: pd.Series | None = None) -> pd.DataFrame:
    """labels: 0/1 per covered hour (time index). other_starts: per-hour count of the
    house's other appliances' switch-ons (any hours; missing hours count as 0)."""
    grid = _grid(labels.index)
    y = labels.reindex(grid, fill_value=0).astype(float)
    covered = pd.Series(1.0, index=labels.index).reindex(grid, fill_value=0.0)
    others = (other_starts.reindex(grid, fill_value=0) if other_starts is not None
              else pd.Series(0.0, index=grid)).astype(float)

    f = pd.DataFrame(index=grid)
    hour, dow = grid.hour.to_numpy(), grid.dayofweek.to_numpy()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["weekend"] = (dow >= 5).astype(float)

    past = y.shift(1, fill_value=0)  # everything below is built from `past`: hours before h
    f["lag_1h"] = past
    f["lag_24h"] = y.shift(24, fill_value=0)
    f["lag_168h"] = y.shift(168, fill_value=0)
    f["uses_24h"] = past.rolling(24, min_periods=1).sum()
    f["uses_7d"] = past.rolling(168, min_periods=1).sum()

    # Hours since the last use (before h), capped at a week.
    idx = np.arange(len(y))
    last_use = pd.Series(np.where(y.to_numpy() > 0, idx, np.nan), index=grid).ffill().shift(1)
    f["hours_since_use"] = np.minimum(idx - last_use.fillna(-10_000).to_numpy(), 168.0)

    # Share of the last 4 weeks with a use at this same hour of the week.
    weekly = sum(y.shift(168 * k, fill_value=0) for k in range(1, 5))
    weekly_covered = sum(covered.shift(168 * k, fill_value=0) for k in range(1, 5))
    f["recent_how_rate"] = (weekly / weekly_covered.replace(0, np.nan)).fillna(0.0)

    f["others_1h"] = others.shift(1, fill_value=0)
    f["others_3h"] = others.shift(1, fill_value=0).rolling(3, min_periods=1).sum()
    f["prev_hour_covered"] = covered.shift(1, fill_value=0)
    return f.loc[labels.index, FEATURES]


def other_appliance_starts(house_uses: pd.DataFrame, exclude_column: str) -> pd.Series:
    """Per-hour count of switch-ons of a house's other appliances (fridges and freezers left
    out: they cycle on their own and say nothing about people). A house with motion sensors
    (CASAS, category "sensor") uses those instead: its "appliances" are annotated activities,
    which a real home would not observe."""
    if (house_uses["category"] == "sensor").any():
        others = house_uses[house_uses["category"] == "sensor"]
        return others["start"].dt.floor("h").value_counts().sort_index()
    others = house_uses[(house_uses["column"] != exclude_column) & (house_uses["category"] != "automatic")]
    return others["start"].dt.floor("h").value_counts().sort_index()
