"""The features must never see the hour they predict, or anything after it."""

import numpy as np
import pandas as pd

from habits.features import FEATURES, build_features


def labels(n_hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2014-01-06", periods=n_hours, freq="h")
    return pd.Series((rng.random(n_hours) < 0.1).astype(int), index=hours)


def test_no_feature_depends_on_the_hour_itself_or_the_future():
    y = labels()
    base = build_features(y)
    h = 400
    changed = y.copy()
    changed.iloc[h:] = 1 - changed.iloc[h:]  # flip the predicted hour and everything after it
    after = build_features(changed)
    pd.testing.assert_frame_equal(base.iloc[:h + 1], after.iloc[:h + 1])  # rows up to and including h unchanged


def test_lags_respect_gaps_in_the_data():
    y = labels()
    with_gap = y.drop(y.index[100:110])  # ten hours without data
    f = build_features(with_gap)
    after_gap = y.index[110]
    assert f.loc[after_gap, "prev_hour_covered"] == 0
    assert f.loc[after_gap, "lag_24h"] == y.loc[after_gap - pd.Timedelta(hours=24)]  # still the right hour


def test_shapes_and_simple_values():
    y = labels()
    f = build_features(y)
    assert list(f.columns) == FEATURES and len(f) == len(y)
    assert f["hours_since_use"].max() <= 168
    i = 300
    assert f["lag_1h"].iloc[i] == y.iloc[i - 1]
    assert f["uses_24h"].iloc[i] == y.iloc[i - 24:i].sum()
