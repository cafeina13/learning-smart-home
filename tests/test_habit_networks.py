"""The habit networks' inputs: past hours only, on a calendar grid; and they train end to end."""

import numpy as np
import pandas as pd

from habits import sequence
from habits.dataset import time_split
from habits.evaluate import Task
from habits.sequence import WINDOW, fit_and_predict_sequence, grid_inputs


def daily_kettle(days: int = 60, gap: bool = True) -> pd.Series:
    hours = pd.date_range("2014-03-03", periods=days * 24, freq="h")
    if gap:
        hours = hours[(hours < "2014-03-20") | (hours >= "2014-03-22")]  # two days without data
    return pd.Series((hours.hour == 7).astype("int8"), index=hours)


def test_window_ends_before_the_hour():
    labels = daily_kettle()
    x, position = grid_inputs(labels, None)
    for k in (200, 500, len(labels) - 1):
        h, p = labels.index[k], position[k]
        last = x[p - 1]  # the window is rows [p - WINDOW, p)
        assert last[0] == int((h - pd.Timedelta(hours=1)).hour == 7)  # own use of hour h-1, not h
        assert np.isclose(last[3], np.sin(2 * np.pi * h.hour / 24))  # calendar of hour h itself
        assert x[p - WINDOW:p].shape == (WINDOW, sequence.N_FEATURES)


def test_gap_hours_are_marked_and_empty():
    labels = daily_kettle()
    x, position = grid_inputs(labels, None)
    first_after = labels.index.get_loc(pd.Timestamp("2014-03-22"))
    window = x[position[first_after] - 48:position[first_after]]
    assert window[:, 2].sum() == 0 and window[:, 0].sum() == 0  # 48 h before: the gap


def test_networks_learn_a_daily_habit(monkeypatch):
    monkeypatch.setattr(sequence, "MAX_EPOCHS", 3)
    labels = daily_kettle(days=40, gap=False)
    train, val, test = time_split(labels)
    task = Task(1, "Appliance1", "Kettle", "habit", labels, pd.DataFrame(), train, val, test, None)
    for name in sequence.NETWORKS:
        predictions, details = fit_and_predict_sequence(name, task)
        assert len(predictions) == len(test) and details["epochs"] >= 1


def test_a_longer_window_still_ends_before_the_hour(monkeypatch):
    monkeypatch.setattr(sequence, "WINDOW", 4 * 168)  # run 2: the forest's 4 weeks of history
    monkeypatch.setattr(sequence, "MAX_EPOCHS", 2)
    labels = daily_kettle(days=60, gap=False)
    x, position = grid_inputs(labels, None)
    k = len(labels) - 1
    assert position[k] == k + 4 * 168 and x[position[k] - 1, 0] == int(labels.index[k - 1].hour == 7)
    train, val, test = time_split(labels)
    task = Task(1, "Appliance1", "Kettle", "habit", labels, pd.DataFrame(), train, val, test, None)
    for name in sequence.NETWORKS:
        predictions, _ = fit_and_predict_sequence(name, task)
        assert len(predictions) == len(test)
