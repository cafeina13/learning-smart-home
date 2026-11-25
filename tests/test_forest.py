"""The habit forest in the running home: it checks suggestions on unseen days and guards standby cuts."""

from datetime import datetime, time

import pandas as pd
import pytest

from habits.forest import Guarded, ForestGuard, HabitForest, restore_time, window_hours
from habits.routines import WEEKDAYS
from habits.suggestions import _check_with_forest, suggest_for_appliance
from habits.synthetic import household
from home.bus import InMemoryBus
from home.devices import PluggedAppliance, SimClock


def all_hours(days) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(hours=23), freq="h")


@pytest.fixture(scope="module")
def coffee_forest():
    uses, days, _ = household(weeks=16)
    return HabitForest.fit(uses["Coffee Machine"], all_hours(days)), days


def test_forest_trains_on_the_first_60_percent_and_checks_on_the_last_20(coffee_forest):
    forest, days = coffee_forest
    assert forest is not None and 0 < forest.threshold < 1
    assert forest.check_days == days[int(len(days) * 0.8):]


def test_forest_agrees_with_the_real_routine_and_not_with_a_made_up_one(coffee_forest):
    forest, _ = coffee_forest
    real = forest.agreement(WEEKDAYS, 13, 2, expect_use=True)  # 06:30-07:30, the coffee habit
    fake = forest.agreement(WEEKDAYS, 6, 2, expect_use=True)  # 03:00-04:00
    night_quiet = forest.agreement(WEEKDAYS, 0, 10, expect_use=False)  # 00:00-05:00, nobody makes coffee
    assert real >= 0.7 and fake <= 0.3 and night_quiet >= 0.9


def test_suggestions_carry_the_forests_verdict(coffee_forest):
    forest, days = coffee_forest
    uses, _, _ = household(weeks=16)
    out = suggest_for_appliance("Coffee Machine", uses["Coffee Machine"], days, forest=forest)
    weekday = next(s for s in out if s.days == WEEKDAYS)
    assert weekday.model_agreement >= 0.7 and "habit model agrees" in weekday.energy_note


class StubForest:
    """A forest that always gives the same probability (to test what the system does with it)."""
    def __init__(self, p: float, agreement: float = 1.0):
        self.p, self.threshold, self._agreement = p, 0.5, agreement

    def agreement(self, *args, **kwargs) -> float:
        return self._agreement

    def predict(self, labels, others=None) -> pd.Series:
        return pd.Series(self.p, index=labels.index)


def test_a_suggestion_the_forest_disagrees_with_is_held_back():
    uses, days, standby = household(weeks=8)
    cuts = suggest_for_appliance("Television Site", uses["Television Site"], days, standby["Television Site"])
    windows = [(0, 16)] * len(cuts)
    assert _check_with_forest(list(cuts), windows, StubForest(0, agreement=0.3)) == []
    kept = _check_with_forest(list(cuts), windows, StubForest(0, agreement=0.9))
    assert len(kept) == len(cuts) and all(s.model_agreement == 0.9 for s in kept)


def test_window_hours_and_restore_time():
    day = datetime(2014, 1, 6).date()
    assert list(window_hours(day, 14, 1).hour) == [7]  # 07:00-07:30
    assert list(window_hours(day, 46, 4).hour) == [23, 0]  # 23:00-01:00, past midnight
    hours = pd.date_range("2014-01-06 00:00", periods=24, freq="h")
    p = pd.Series(0.1, index=hours)
    start, end = datetime(2014, 1, 6, 0, 30), datetime(2014, 1, 6, 9, 0)
    assert restore_time(start, end, p, 0.5) == end  # nothing flagged: the cut runs its course
    p[pd.Timestamp("2014-01-06 06:00")] = 0.8
    assert restore_time(start, end, p, 0.5) == datetime(2014, 1, 6, 6, 0)
    p[pd.Timestamp("2014-01-06 00:00")] = 0.8
    assert restore_time(start, end, p, 0.5) == start  # flagged from the first hour: no cut tonight


def guarded_tv(p: float):
    bus, clock = InMemoryBus(), SimClock(datetime(2014, 3, 3, 0, 20))
    tv = PluggedAppliance(bus, clock, "living", "television_site", watts=90, standby_w=9)
    learn = pd.date_range("2014-01-06", "2014-03-02 23:00", freq="h")
    g = Guarded(StubForest(p), pd.Series(0, index=learn, dtype="int8"), pd.Series(dtype=float))
    guard = ForestGuard(bus, clock, {tv.id: g}, {tv.id})
    tv.publish_state()
    bus.flush()
    return bus, clock, tv, guard


@pytest.mark.parametrize("p, restored", [(0.9, True), (0.1, False)])
def test_guard_restores_a_cut_plug_only_when_a_use_looks_likely(p, restored):
    bus, clock, tv, guard = guarded_tv(p)
    tv.set("cut", source="rule:cut living/television_site weekdays 00:30-17:00")
    bus.flush()
    assert (tv.value == "off") == restored  # "off" = plugged in, standby restored; never switched on
    assert len(guard.restored) == int(restored)
    clock.advance(3600)  # next hour: the guard checks again while the power is still cut
    guard.tick()
    bus.flush()
    assert tv.value == ("off" if restored else "cut")


def test_guard_counts_the_households_switch_ons():
    bus, clock, tv, guard = guarded_tv(0.1)
    tv.set("on", source="resident")
    bus.flush()
    assert guard.starts[tv.id] == [datetime(2014, 3, 3, 0, 20)]


def test_forest_rescues_count_as_strikes_and_three_disable_the_rule():
    from home.notify import PhoneNotifier
    from home.rules import Action, Rule, TimeWindow
    from home.simulation import ArticleHome

    bus, clock = InMemoryBus(), SimClock(datetime(2014, 3, 3, 0, 0))
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=9)
    learn = pd.date_range("2014-01-06", "2014-03-02 23:00", freq="h")
    ForestGuard(bus, clock, {tv.id: Guarded(StubForest(0.9), pd.Series(0, index=learn, dtype="int8"),
                                            pd.Series(dtype=float))}, {tv.id})
    sent = []
    PhoneNotifier(bus, lambda title, message, priority, tags: sent.append((title, message)))
    rule = Rule("tv night cut", [], [Action(tv.id, "cut")], when=TimeWindow(time(1, 0), time(5, 0)), origin="learned")
    home.engine.add(rule)
    home.announce()
    for night in range(3):
        target = datetime(2014, 3, 3 + night, 0, 55)
        clock.advance((target - clock.now()).total_seconds())
        home.run(seconds=10 * 60, dt=60)  # the rule cuts at 01:00, the guard restores at once
        assert tv.value == "off"
        home.run(seconds=5 * 3600, dt=600)  # leave the window, so it can fire again tomorrow
    assert home.engine.forest_strikes[rule.name] == 3 and not rule.enabled
    assert any("the habit model stepped in 3" in m for _, m in sent)
