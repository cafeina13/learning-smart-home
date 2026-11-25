"""The forecasting pieces: windows, preprocessing, ensemble weights, feature selection."""

import numpy as np
import pandas as pd

from energy.data import WINDOW, Scaler, windows
from energy.forecasters import apply_weights, hourly_weights, select_features


def hours_frame(n=24 * 10, gap=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-06", periods=n, freq="h")
    df = pd.DataFrame({"load": 300 + 200 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 10, n),
                       "hour": idx.hour, "temperature": rng.normal(10, 3, n)}, index=idx)
    if gap is not None:
        df.iloc[gap[0]:gap[1], 0] = np.nan
    return df


def test_window_ends_just_before_the_target_hour():
    df = hours_frame()
    X, y, when = windows(df, ["load", "hour"])
    assert X.shape[1:] == (WINDOW, 2)
    first = df.index.get_loc(when[0])
    assert first == WINDOW
    np.testing.assert_allclose(X[0][:, 0], df["load"].iloc[:WINDOW].to_numpy(), rtol=1e-6)
    assert y[0] == np.float32(df["load"].iloc[WINDOW])  # the target is not inside its own window


def test_targets_whose_window_has_a_gap_are_skipped():
    df = hours_frame(gap=(170, 172))
    _, _, when = windows(df, ["load"])
    positions = {df.index.get_loc(t) for t in when}
    assert 170 not in positions and 171 not in positions  # missing targets
    assert all(p >= 172 + WINDOW or p < 170 for p in positions)  # windows that would contain the gap


def test_scaler_follows_the_paper():
    df = hours_frame()
    X, _, _ = windows(df, ["load", "hour"])
    Z = Scaler(["load", "hour"]).fit(X).transform(X)
    np.testing.assert_allclose(Z[:, :, 1], X[:, :, 1] / 23 + 1)  # hour / max + 1
    assert abs(Z[:, :, 0].mean()) < 1e-4 and abs(Z[:, :, 0].std() - 1) < 1e-3  # z-scored


def test_hourly_weights_are_a_convex_combination_and_pick_the_better_forecaster():
    rng = np.random.default_rng(1)
    n = 24 * 30
    hours = np.tile(np.arange(24), 30)
    y = rng.normal(500, 100, n)
    good = y + rng.normal(0, 5, n)
    bad = y + rng.normal(0, 80, n)
    P = np.column_stack([good, bad])
    W = hourly_weights(P, y, hours)
    assert np.allclose(W.sum(axis=1), 1) and (W >= 0).all()
    assert (W[:, 0] > W[:, 1]).all()
    assert np.abs(apply_weights(P, hours, W) - y).mean() < np.abs(bad - y).mean()


def test_feature_selection_keeps_the_informative_feature():
    rng = np.random.default_rng(2)
    n, f = 400, 3
    X = rng.normal(size=(n, WINDOW, f)).astype("float32")
    y = (X[:, -1, 0] * 50 + 300).astype("float32")  # only feature 0's last hour matters
    hours = np.zeros(n, dtype=int)
    best, trace = select_features(["a", "b", "c"], X[:300], y[:300], hours[:300], X[300:], y[300:], hours[300:],
                                  keep=2)
    assert "a" in best and trace[0][0] == "drop"


def test_networks_build_and_train_for_any_feature_count(monkeypatch):
    import pytest
    pytest.importorskip("torch")
    import energy.deep as deep
    monkeypatch.setattr(deep, "MAX_EPOCHS", 2)  # just check they run
    rng = np.random.default_rng(3)
    for f in (1, 3, 8):
        A = rng.normal(size=(120, WINDOW, f)).astype("float32")
        target = rng.normal(size=120).astype("float32")
        for name in ("CNN1", "CNN2", "RNN1", "GRU", "CNN-GRU"):
            out = deep.fit_predict_deep(name, A, target, A[:7])
            assert out.shape == (7,) and np.isfinite(out).all()
