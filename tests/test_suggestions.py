from datetime import datetime, time

import pandas as pd

from habits.routines import WEEKDAYS, find_quiet_window, find_routines, holds_rate
from habits.suggestions import SuggestionStore, control_of, device_id, suggest_for_appliance
from habits.synthetic import household
from home.bus import InMemoryBus
from home.devices import SimClock
from home.rules import Rule
from home.simulation import ArticleHome


def test_device_ids_guess_a_room():
    assert device_id("Kettle") == "kitchen/kettle"
    assert device_id("Television Site") == "living/television_site"
    assert device_id("Washing Machine(1)") == "utility/washing_machine_1"
    assert device_id("Electric Heater") == "home/electric_heater"


def test_finds_the_weekday_kettle_routine_and_it_holds_later():
    uses, days, _ = household(weeks=16)
    kettle = uses["Kettle"]
    train_days, test_days = days[:70], days[70:]
    train = kettle[kettle["start"] < pd.Timestamp(days[70])]
    test = kettle[kettle["start"] >= pd.Timestamp(days[70])]

    routines = [r for r in find_routines(train, train_days, "Kettle") if r.weekdays == WEEKDAYS]
    assert len(routines) == 1
    r = routines[0]
    assert time(6, 30) <= r.start <= time(7, 30)
    assert r.support >= 0.7
    assert holds_rate(r.weekdays, r.start_slot, r.n_slots, test, test_days) >= 0.6


def test_full_day_window_runs_24_hours():
    from home.rules import TimeWindow
    sunday = TimeWindow(time(0), time(0), (6,))
    assert sunday.check(datetime(2026, 10, 18, 0, 0)) and sunday.check(datetime(2026, 10, 18, 23, 59))
    assert not sunday.check(datetime(2026, 10, 19, 0, 0))
    assert sunday.describe() == "Sun all day from 00:00"


def test_tv_quiet_window_spans_the_night():
    uses, days, _ = household()
    q = find_quiet_window(uses["Television Site"], days, "Television Site", WEEKDAYS)
    assert q is not None and q.hours >= 8
    assert q.start >= time(22, 0) or q.start <= time(1, 0)  # starts late evening
    assert time(5, 0) <= q.end <= time(19, 30)  # ends next day before the evening TV


def test_control_types():
    assert control_of("Coffee Machine") == "auto"
    assert control_of("Television Site") == "manual"
    assert control_of("Kettle") == "manual"
    assert control_of("Dishwasher") == "armable"
    assert control_of("Fridge-Freezer") is None


def suggestions_by_appliance():
    uses, days, standby = household()
    return {a: suggest_for_appliance(a, u, days, standby.get(a, 0.0)) for a, u in uses.items()}


def test_the_house_never_offers_to_switch_on_a_manual_or_armable_appliance():
    for appliance, suggestions in suggestions_by_appliance().items():
        for s in suggestions:
            for rule in s.rules:
                for action in rule.actions:
                    if s.control == "manual":
                        assert action.value in ("cut", "off"), (appliance, rule.name)
                    if s.control == "armable":
                        raise AssertionError(f"smart start must not carry rules: {rule.name}")


def test_suggestion_kinds_per_appliance():
    by = suggestions_by_appliance()
    assert {s.kind for s in by["Coffee Machine"]} == {"automation"}
    assert {s.kind for s in by["Television Site"]} == {"standby_cut"}
    assert by["Kettle"] == []  # manual and no standby draw: nothing worth doing
    assert [s.kind for s in by["Dishwasher"]] == ["smart_start"]  # one suggestion, not one per day
    washing = by["Washing Machine"][0]
    assert washing.kind == "smart_start"  # needs no routine: armed whenever it is loaded
    assert washing.setting == {"default_deadline": "18:00", "next_day": True}  # loaded in the evening
    assert washing.energy_kwh_week > 0 and washing.support > 0.5  # most past runs would have gained solar


def test_irregular_machine_still_gets_smart_start_but_rare_one_does_not():
    import numpy as np
    rng = np.random.default_rng(0)
    days = [pd.Timestamp("2014-03-03") + pd.Timedelta(days=i) for i in range(84)]
    # Twice a week at random evening hours: no routine, but plenty to move to the sun.
    starts = [d + pd.Timedelta(hours=float(rng.uniform(17, 22))) for d in days if rng.random() < 2 / 7]
    uses = pd.DataFrame({"start": starts, "end": [s + pd.Timedelta(minutes=100) for s in starts],
                         "duration_s": 6000.0, "energy_wh": 1000.0})
    day_list = [d.date() for d in days]
    assert [s.kind for s in suggest_for_appliance("Dishwasher", uses, day_list)] == ["smart_start"]
    rare = uses.iloc[:3]  # three runs in twelve weeks: not worth suggesting
    assert suggest_for_appliance("Dishwasher", rare, day_list) == []


def test_standby_cut_has_a_restore_on_the_right_day():
    tv = suggestions_by_appliance()["Television Site"]
    weekday = next(s for s in tv if s.days == WEEKDAYS)
    cut, restore = weekday.rules
    assert cut.actions[0].value == "cut" and restore.actions[0].value == "off"
    assert restore.when.start == cut.when.end
    if cut.when.end <= cut.when.start:  # crosses midnight: restore is on the next day
        assert restore.when.weekdays == (1, 2, 3, 4, 5)


def test_accepted_cut_runs_in_the_home_and_spares_a_tv_in_use():
    uses, days, standby = household()
    store = SuggestionStore()
    s = next(x for x in suggest_for_appliance("Television Site", uses["Television Site"], days, 9.0)
             if x.days == WEEKDAYS)
    assert store.offer(s) and not store.offer(s)
    accepted = store.decide(s.key, accept=True)
    assert store.decide(s.key, accept=True) is None  # already decided

    bus = InMemoryBus()
    cut_rule = accepted.rules[0]
    start = datetime.combine(datetime(2026, 10, 12).date(), cut_rule.when.start)  # a Monday
    clock = SimClock(start)
    home = ArticleHome(bus, clock)
    tv = home.add_appliance(s.device, "manual", watts=90, standby_w=9, run_minutes=60)
    for rule in accepted.rules:
        home.engine.add(rule)
    tv.set("on", source="user")  # watching when the cut window opens
    home.run(seconds=30 * 60, dt=60)
    assert tv.value == "on"  # not interrupted
    home.run(seconds=40 * 60, dt=60)
    assert tv.value == "cut"  # cut once it was switched off


def test_rules_round_trip_through_json():
    for suggestions in suggestions_by_appliance().values():
        for s in suggestions:
            for rule in s.rules:
                assert Rule.from_dict(rule.to_dict()).to_dict() == rule.to_dict()


def test_newer_evidence_updates_a_pending_suggestion():
    uses, days, standby = household()
    store = SuggestionStore()
    old = suggest_for_appliance("Coffee Machine", uses["Coffee Machine"], days, learned_from="offline data")[0]
    store.offer(old)
    new = suggest_for_appliance("Coffee Machine", uses["Coffee Machine"], days, learned_from="this home's event log")[0]
    assert not store.offer(new)  # not a new suggestion...
    assert store.list()[0]["learned_from"] == "this home's event log"  # ...but the evidence is updated
    store.decide(new.key, accept=False)
    store.offer(old)
    assert store.list()[0]["learned_from"] == "this home's event log"  # decided ones are left alone
