"""The tariff prices: the Economy 7 window across the clock change, and 2019 days on the same weekday."""

import json
from datetime import date

import pandas as pd
import pytest

from habits.tariffs import E7_DAY, E7_NIGHT, FLAT, Prices, is_e7_night, same_weekday_in


def test_same_weekday_in_2019_keeps_the_weekday_and_season():
    for day in (date(2013, 10, 9), date(2014, 6, 30), date(2015, 2, 28), date(2016, 2, 29),
                date(2013, 12, 31), date(2014, 1, 1), date(2017, 12, 30)):
        moved = same_weekday_in(day)
        assert moved.year == 2019
        assert moved.weekday() == day.weekday()
        assert abs((moved - date(2019, day.month, min(day.day, 28))).days) <= 7


def test_economy7_night_is_0030_to_0730_utc():
    winter = pd.DatetimeIndex(["2014-01-15 00:29", "2014-01-15 00:30", "2014-01-15 07:29", "2014-01-15 07:30"])
    assert is_e7_night(winter).tolist() == [False, True, True, False]
    # In British summer time the same UTC window is 01:30-08:30 on the clock.
    summer = pd.DatetimeIndex(["2014-07-15 00:45", "2014-07-15 01:30", "2014-07-15 08:29", "2014-07-15 08:30"])
    assert is_e7_night(summer).tolist() == [False, True, True, False]


def test_flat_and_economy7_prices():
    times = pd.DatetimeIndex(["2014-01-15 03:00", "2014-01-15 12:00"])
    assert Prices("flat", "B").at(times).tolist() == [FLAT["B"]] * 2
    assert Prices("economy7", "B").at(times).tolist() == [E7_NIGHT["B"], E7_DAY["B"]]


def test_agile_reads_the_same_weekday_of_2019(tmp_path):
    # A fake cache: every half hour of 2019 priced by its hour of day, plus 100 on Wednesdays.
    utc = pd.date_range("2019-01-01", "2020-01-01", freq="30min", inclusive="left", tz="UTC")
    local = utc.tz_convert("Europe/London")
    rows = [{"valid_from": t.isoformat(), "value_inc_vat": float(l.hour + (100 if l.weekday() == 2 else 0))}
            for t, l in zip(utc, local)]
    (tmp_path / "agile_2019_B.json").write_text(json.dumps(rows))
    prices = Prices("agile", "B", cache=tmp_path)
    wednesday_6pm = pd.DatetimeIndex(["2014-06-18 18:10"])  # a Wednesday
    thursday_6pm = pd.DatetimeIndex(["2014-06-19 18:10"])
    assert prices.at(wednesday_6pm).tolist() == [118.0]
    assert prices.at(thursday_6pm).tolist() == [18.0]


def test_unknown_tariff_is_refused():
    with pytest.raises(ValueError):
        Prices("cheapest")
