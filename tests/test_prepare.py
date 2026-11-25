"""The REFIT preparation rules, on small made-up power traces."""

import numpy as np
import pandas as pd
import pytest

from habits.prepare import idle_level_w, use_rules_for
from habits.refit import extract_uses


def trace(minutes: int, spans: list[tuple[float, float, float]], idle: float = 0.0) -> pd.Series:
    index = pd.date_range("2014-01-06", periods=minutes * 60 // 8, freq="8s", tz="UTC")
    minute = np.arange(len(index)) * 8 / 60
    power = np.full(len(index), idle, dtype="float32")
    for a, b, w in spans:
        power[(minute >= a) & (minute < b)] = w
    return pd.Series(power, index=index)


def test_idle_level_ignores_zeros_and_use():
    values = np.array([0] * 500 + [9] * 400 + [85] * 100, dtype=float)
    assert idle_level_w(values) == 9  # a desktop idling at 9 W
    assert idle_level_w(np.array([0] * 990 + [2000] * 10, dtype=float)) == 0  # a kettle


def test_standby_is_the_average_idle_draw_including_zeros():
    # Plugged in and idling at 10 W half the idle time, switched off at the wall the other half.
    p = trace(600, [(0, 60, 100.0)], idle=0.0)
    p[(p.index - p.index[0]).total_seconds() > 330 * 60] = 10.0
    rules, idle, standby = use_rules_for(p)
    assert idle == 10 and rules.on_threshold_w == 40
    assert standby == pytest.approx(10 * 270 / 540, rel=0.02)  # 10 W for 270 of 540 idle minutes


def test_a_dishwasher_cycle_with_long_pauses_is_one_use():
    # Heat 20 min, wash at low power for 25 min (below threshold), heat again 15 min.
    p = trace(180, [(10, 30, 2000.0), (55, 70, 2000.0)])
    habit_rules, _, _ = use_rules_for(p, "habit")
    machine_rules, _, _ = use_rules_for(p, "shiftable")
    assert len(extract_uses(p, habit_rules)) == 2  # a 2-minute bridge splits it
    assert len(extract_uses(p, machine_rules)) == 1  # a 30-minute bridge keeps the cycle whole


def test_short_blip_is_not_a_machine_cycle():
    p = trace(60, [(10, 15, 500.0)])  # 5 minutes
    machine_rules, _, _ = use_rules_for(p, "shiftable")
    assert extract_uses(p, machine_rules).empty  # under the 10-minute minimum
