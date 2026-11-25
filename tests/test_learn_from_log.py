"""Closed loop: a resident lives in the home, and the home learns its habits from its own event log."""

from datetime import datetime, time

import pytest

from habits.from_log import days_in_log, uses_from_log
from habits.resident import Resident
from habits.routines import WEEKDAYS, find_routines
from habits.suggestions import device_id, suggest_for_appliance
from home.bus import InMemoryBus
from home.devices import SimClock
from home.eventlog import EventLog
from home.simulation import ArticleHome

APPLIANCES = {"Kettle": (2000, 3), "Television Site": (90, 200), "Washing Machine": (450, 110)}


@pytest.fixture(scope="module")
def lived_in_home():
    bus = InMemoryBus()
    clock = SimClock(datetime(2026, 1, 5, 0, 0))  # a Monday
    log = EventLog()
    log.attach(bus)
    home = ArticleHome(bus, clock, report_every=600)  # sensors every 10 min keeps the log small
    switches = {device_id(a): home.add_switch(device_id(a), w, run_minutes=m) for a, (w, m) in APPLIANCES.items()}
    resident = Resident(clock, switches, seed=1)
    while clock.now() < datetime(2026, 2, 16):  # six weeks
        resident.act()
        home.step(60)
    return home, log


def test_log_holds_the_residents_uses(lived_in_home):
    _, log = lived_in_home
    kettle = uses_from_log(log, "kitchen/kettle", 2000, only_source="resident")
    assert 40 <= len(kettle) <= 120  # morning weekday + weekend + afternoon habits over six weeks
    assert (kettle["duration_s"].between(170, 190)).all()  # 3-minute boils
    assert abs(kettle["energy_wh"].iloc[0] - 100) < 5


def test_home_rediscovers_the_weekday_kettle_routine(lived_in_home):
    _, log = lived_in_home
    days = days_in_log(log)
    assert len(days) >= 40
    kettle = uses_from_log(log, "kitchen/kettle", 2000, only_source="resident")
    weekday = [r for r in find_routines(kettle, days, "Kettle") if r.weekdays == WEEKDAYS]
    assert weekday and time(6, 30) <= weekday[0].start <= time(7, 30)  # the habit is ~07:09


def test_suggestions_from_own_log(lived_in_home):
    _, log = lived_in_home
    days = days_in_log(log)
    washing = uses_from_log(log, "utility/washing_machine", 450, only_source="resident")
    suggestions = suggest_for_appliance("Washing Machine", washing, days)
    assert [s.kind for s in suggestions] == ["smart_start"]
    assert 5 in suggestions[0].days  # the Saturday wash


def test_automations_are_not_learned_back_as_habits(lived_in_home):
    home, log = lived_in_home
    # Something other than the resident switches the TV on: it must not count as the resident's habit.
    home.add_switch("living/television_site", 90).set("on", source="rule:test")
    home.run(seconds=4 * 3600, dt=60)
    all_uses = uses_from_log(log, "living/television_site", 90)
    resident_uses = uses_from_log(log, "living/television_site", 90, only_source="resident")
    assert len(all_uses) == len(resident_uses) + 1


def test_attribution_survives_a_delayed_state():
    # Over a real broker the "on" state can arrive (and be stamped) a step after the command.
    log = EventLog()
    log.record("home/kitchen/kettle/set", {"value": "on", "ts": "2026-01-05T07:10:00", "source": "resident"})
    log.record("home/kitchen/kettle/state", {"device": "kitchen/kettle", "kind": "switch", "value": "on",
                                             "ts": "2026-01-05T07:11:00", "source": "sim"})
    log.record("home/kitchen/kettle/state", {"device": "kitchen/kettle", "kind": "switch", "value": "off",
                                             "ts": "2026-01-05T07:14:00", "source": "sim"})
    uses = uses_from_log(log, "kitchen/kettle", 2000, only_source="resident")
    assert len(uses) == 1 and uses["duration_s"].iloc[0] == 180
