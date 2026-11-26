"""Explainable AI in the running home: reasons in words for the habit forest's predictions (SHAP)."""

from datetime import datetime

import pandas as pd
import pytest

from habits.explain import GROUPS, NOT_A_REASON, grouped, phrase, sentence
from habits.forest import Guarded, ForestGuard, HabitForest
from habits.routines import WEEKDAYS
from habits.synthetic import household
from home.bus import InMemoryBus
from home.devices import PluggedAppliance, SimClock
from home.events import ALERTS_TOPIC

pytest.importorskip("shap")


def all_hours(days) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(hours=23), freq="h")


@pytest.fixture(scope="module")
def kettle():
    uses, days, _ = household(weeks=16)
    hours = all_hours(days)
    return HabitForest.fit(uses["Kettle"], hours), uses["Kettle"], hours, days


def test_phrases_use_the_hours_own_values():
    row = pd.Series({"weekend": 0.0, "lag_1h": 0.0, "lag_24h": 1.0, "lag_168h": 0.0, "uses_24h": 2.0,
                     "uses_7d": 9.0, "hours_since_use": 2.0, "recent_how_rate": 0.75, "others_1h": 1.0,
                     "others_3h": 3.0})
    when = pd.Timestamp("2014-03-04 07:00")  # a Tuesday
    assert phrase("Hour of day", row, when) == "it is 07:00"
    assert phrase("Day of week", row, when) == "it is a Tuesday"
    assert phrase("Used at this hour yesterday", row, when) == "used at this hour yesterday"
    assert phrase("Used in the previous hour", row, when) == "not used in the previous hour"
    assert phrase("Hours since the last use", row, when) == "last used 2 hours ago"
    assert phrase("Share of the last 4 weeks used at this hour", row, when) == "used at this hour in 75% of the last 4 weeks"
    assert phrase("Rest of the house active, last 3 hours", row, when) == "3 other appliance switch-ons in the last 3 hours"


def test_groups_sum_paired_inputs():
    columns = [c for cols in GROUPS.values() for c in cols]
    values = [[1.0] * len(columns)]
    g = grouped(values, columns)
    assert g.loc[0, "Hour of day"] == 2.0 and g.loc[0, "Weekend"] == 1.0  # sine + cosine summed


def test_the_forest_explains_the_morning_kettle(kettle):
    forest, uses, hours, days = kettle
    labels = pd.Series(0, index=hours, dtype="int8")
    labels[labels.index.isin(uses["start"].dt.floor("h"))] = 1
    seven = pd.Timestamp(days[-3]) + pd.Timedelta(hours=7)  # a late morning the forest predicts a use for
    reasons = forest.explain_hour(labels[labels.index <= seven])
    assert 1 <= len(reasons) <= 3
    assert all(r.contribution > 0 and r.group not in NOT_A_REASON for r in reasons)
    assert sentence(reasons) == "; ".join(r.phrase for r in reasons)


def test_the_check_says_what_its_agreement_rests_on(kettle):
    forest, *_ = kettle
    why = forest.window_reasons(WEEKDAYS, 14, 1, expect_use=True)  # weekdays 07:00-07:30
    assert why and all(isinstance(w, str) for w in why)


def test_the_guard_sends_its_reasons_with_a_restore(kettle):
    forest, uses, hours, days = kettle
    bus, clock = InMemoryBus(), SimClock(datetime.combine(days[-1], datetime.min.time()).replace(hour=7, minute=5))
    plug = PluggedAppliance(bus, clock, "kitchen", "kettle", watts=2000, standby_w=1.0)
    learn = hours[hours < pd.Timestamp(days[-1])]
    labels = pd.Series(0, index=learn, dtype="int8")
    labels[labels.index.isin(uses["start"].dt.floor("h"))] = 1
    forest.threshold = 0.0  # make sure it restores, to see the alert's reasons
    guard = ForestGuard(bus, clock, {plug.id: Guarded(forest, labels, pd.Series(dtype=float))}, {plug.id})
    alerts = []
    bus.subscribe(ALERTS_TOPIC, lambda topic, payload: alerts.append(payload))
    plug.publish_state()
    plug.set("cut", source="rule:cut kitchen/kettle")
    bus.flush()
    restore = [a for a in alerts if a.get("kind") == "forest_restored"]
    assert restore and restore[0]["reasons"] and guard.reasons[0] == restore[0]["reasons"]
