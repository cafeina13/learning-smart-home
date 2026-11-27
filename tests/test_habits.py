"""Habit pipeline on synthetic power traces, so it is tested before REFIT is downloaded."""

import numpy as np
import pandas as pd
import pytest

from habits.baseline import HourOfWeekBaseline, evaluate
from habits.dataset import covered_hours, hourly_labels, time_split
from habits.refit import HOUSES, RAW_DIR, README, UseRules, appliance_names, category, extract_uses


@pytest.mark.skipif(not (RAW_DIR / README).exists(), reason="REFIT readme not downloaded")
def test_readme_gives_names_for_all_20_houses():
    names = appliance_names()
    assert sorted(names) == HOUSES
    assert names[3]["Appliance9"] == "Kettle"
    assert names[4]["Appliance4"] == "Washing Machine(1)"
    assert "Appliance2" not in names[12]  # "???" stays unnamed


def test_categories():
    assert category("Fridge-Freezer") == "automatic"
    assert category("Dishwasher") == "shiftable"
    assert category("Washer Dryer") == "shiftable"
    assert category("Kettle") == "habit"
    assert category("Television Site") == "habit"


def trace(start: str, minutes: int, on_spans: list[tuple[int, int, float]]) -> pd.Series:
    """8-second power samples; on_spans are (from_minute, to_minute, watts)."""
    index = pd.date_range(start, periods=minutes * 60 // 8, freq="8s", tz="Europe/London")
    power = np.full(len(index), 2.0, dtype="float32")  # standby draw
    minute = (np.arange(len(index)) * 8) / 60
    for a, b, w in on_spans:
        power[(minute >= a) & (minute < b)] = w
    return pd.Series(power, index=index)


def test_one_kettle_boil_is_one_use_with_right_energy():
    uses = extract_uses(trace("2014-01-06 07:00", 30, [(10, 13, 2000.0)]))
    assert len(uses) == 1
    use = uses.iloc[0]
    assert use["start"].hour == 7 and use["start"].minute == 10
    assert abs(use["duration_s"] - 180) <= 16
    assert abs(use["energy_wh"] - 100) < 5  # 2 kW for 3 minutes = 100 Wh


def test_short_pause_is_bridged_and_blips_are_dropped():
    # Washing machine: 20 min on, 1 min pause, 20 min on -> one use. A 16-second blip -> nothing.
    uses = extract_uses(trace("2014-01-06 09:00", 90, [(5, 25, 500.0), (26, 46, 500.0), (70, 70.27, 800.0)]),
                        UseRules(max_gap_s=120, min_on_s=60))
    assert len(uses) == 1
    assert uses.iloc[0]["duration_s"] > 40 * 60


def test_standby_never_counts_as_use():
    assert extract_uses(trace("2014-01-06 00:00", 60, [])).empty


def test_baseline_learns_a_weekday_morning_routine():
    # Six weeks of a kettle used at 07:00 on weekdays only.
    hours = pd.date_range("2014-01-06", periods=6 * 168, freq="h", tz="Europe/London")
    starts = [h + pd.Timedelta(minutes=10) for h in hours if h.hour == 7 and h.dayofweek < 5]
    uses = pd.DataFrame({"start": pd.DatetimeIndex(starts)})
    labels = hourly_labels(uses, hours)
    assert labels.sum() == 30

    train, val, test = time_split(labels)
    model = HourOfWeekBaseline().fit(train).tune_threshold(val)
    result = evaluate(test, model.predict(test.index))
    assert result["f1"] == 1.0


def test_hours_without_data_are_excluded():
    index = pd.date_range("2014-01-06 00:00", periods=450, freq="8s", tz="Europe/London")  # one hour
    gap = pd.date_range("2014-01-06 05:00", periods=100, freq="8s", tz="Europe/London")  # sparse hour
    hours = covered_hours(index.append(gap))
    assert list(hours) == [pd.Timestamp("2014-01-06 00:00", tz="Europe/London")]


def test_routines_work_without_energy_columns():
    # CASAS activities have no energy or duration: counting must still find their routines.
    from datetime import date, datetime, timedelta

    from habits.routines import WEEKDAYS, find_routines

    days = [date(2014, 1, 6) + timedelta(days=i) for i in range(28)]
    starts = [datetime.combine(d, datetime.min.time()) + timedelta(hours=7, minutes=10) for d in days if d.weekday() < 5]
    uses = pd.DataFrame({"start": pd.to_datetime(starts)})
    routines = find_routines(uses, days, "Cook_Breakfast", kinds=(WEEKDAYS,))
    assert routines and routines[0].start_slot == 14 and routines[0].energy_wh_per_use == 0.0
