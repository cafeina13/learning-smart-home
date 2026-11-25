"""The real-data demo: REFIT house 6 replayed in the simulated home (skipped without the prepared data)."""

from datetime import timedelta
from pathlib import Path

import pytest

from habits.replay import ReplayResident
from home.bus import InMemoryBus
from home.devices import SimClock
from home.eventlog import EventLog
from home.simulation import ArticleHome

PREPARED = Path(__file__).resolve().parents[1] / "data" / "processed" / "refit" / "uses.parquet"
pytestmark = pytest.mark.skipif(not PREPARED.exists(), reason="REFIT data not prepared")


@pytest.fixture(scope="module")
def demo():
    from habits.demo import build_refit_demo
    return build_refit_demo()


def live_one_day(demo, smart_start=None, rules=()):
    bus, clock = InMemoryBus(), SimClock(demo.start)
    log = EventLog()
    log.attach(bus)
    home = ArticleHome(bus, clock, solar=demo.solar_w, report_every=600)
    devices = {a.device: home.add_appliance(a.device, a.control, a.watts, a.standby_w, a.run_minutes)
               for a in demo.appliances}
    for rule in rules:
        home.engine.add(rule)
    resident = ReplayResident(clock, devices, demo.replay_uses, smart_start=smart_start)
    resident.skip_to(demo.start)
    end = demo.start + timedelta(days=1)
    while clock.now() < end:
        resident.act()
        home.step(60)
    return home, log, devices


def test_suggestions_are_learned_only_from_the_first_80_percent(demo):
    assert demo.suggestions, "house 6 should get suggestions"
    assert all(u["start"].max() < demo.start for u in demo.learn_uses.values() if len(u))
    assert all(u["start"].min() >= demo.start - timedelta(hours=7) for u in demo.replay_uses.values() if len(u))
    assert {s.kind for s in demo.suggestions} >= {"standby_cut"}


def test_one_real_day_is_replayed_use_for_use(demo):
    home, log, _ = live_one_day(demo)
    day_end = demo.start + timedelta(days=1)
    expected = sum(((u["start"] >= demo.start) & (u["start"] < day_end)).sum() for u in demo.replay_uses.values())
    (commands,) = log.rows("SELECT COUNT(*) FROM messages WHERE type = 'set' AND source = 'resident' "
                           "AND value IN ('\"on\"', '\"armed\"')")[0]
    assert expected > 5 and abs(commands - expected) <= 2  # a use may start while the last one still runs
    assert home.meter["load_kwh"] > 1  # a real day's appliance energy
    assert home.solar.value >= 0


def test_accepted_cut_on_real_data_never_cuts_a_tv_in_use(demo):
    cut = next(s for s in demo.suggestions if s.kind == "standby_cut" and "TV" in s.appliance)
    home, log, devices = live_one_day(demo, rules=cut.rules)
    tv = cut.device
    rows = log.rows("SELECT type, value, source FROM messages WHERE device = ? AND type IN ('set', 'state') "
                    "ORDER BY id", (tv,))
    state, last_command = "off", None
    cuts = 0
    for msg_type, value, source in rows:
        value = value.strip('"')
        if msg_type == "set":
            last_command = (value, source)
            continue
        if value == "cut":
            cuts += 1
            # The TV may go from "on" to "cut" only because the person just switched it off.
            assert state != "on" or last_command == ("off", "resident")
        state = value
    assert cuts > 0, "the cut rule should have acted during a real day"

