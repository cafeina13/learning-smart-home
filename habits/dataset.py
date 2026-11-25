"""From uses to the learning problem: "is this appliance switched on in this hour?"

Every hour of the recording becomes one row per appliance, labelled 1 if at
least one use started in that hour. Hours where the house has no data at all
are left out, so a broken logger is not mistaken for "not used".

The split is by time, never shuffled: the model learns from earlier months and
is tested on later ones, which is how it would be used in a real home.
"""

import pandas as pd


def hourly_labels(uses: pd.DataFrame, covered_hours: pd.DatetimeIndex) -> pd.Series:
    """1 for each covered hour in which a use started, else 0."""
    started = uses["start"].dt.floor("h").value_counts()
    labels = pd.Series(0, index=covered_hours, dtype="int8")
    hit = started.index.intersection(covered_hours)
    labels.loc[hit] = 1
    return labels


def covered_hours(index: pd.DatetimeIndex, min_samples: int = 300) -> pd.DatetimeIndex:
    """Hours with enough samples to trust (300 x 8 s = 40 of 60 minutes)."""
    counts = pd.Series(1, index=index).groupby(index.floor("h")).size()
    return counts.index[counts >= min_samples]


def covered_days(hours, min_hours: int = 20) -> list:
    """Days with at least `min_hours` covered hours (the rest are too broken to count as a day)."""
    hours = pd.DatetimeIndex(hours)
    counts = pd.Series(1, index=hours).groupby(hours.date).size()
    return sorted(counts.index[counts >= min_hours])


def time_split(labels: pd.Series, train_frac: float = 0.6, val_frac: float = 0.2):
    """Split a time-indexed series into train / validation / test by date order."""
    n = len(labels)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return labels.iloc[:a], labels.iloc[a:b], labels.iloc[b:]
