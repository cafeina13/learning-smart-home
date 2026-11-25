from datetime import datetime, time

from home.devices import Lamp
from home.events import state_topic
from home.rules import Action, Condition, Rule, RuleEngine, TimeWindow


def report(bus, device, value):
    bus.publish(state_topic(device), {"device": device, "kind": "x", "value": value, "ts": ""})


def test_rule_fires_once_on_rising_edge(bus, clock):
    lamp = Lamp(bus, clock, "living", "lamp")
    engine = RuleEngine(bus, clock, [
        Rule("dark", [Condition("roof/solar", "<=", 1)], [Action("living/lamp", "on")]),
    ])
    report(bus, "roof/solar", 300)
    assert lamp.value == "off"
    report(bus, "roof/solar", 0)
    report(bus, "roof/solar", 0)  # still true: must not fire again
    assert lamp.value == "on"
    assert [name for _, name in engine.fired] == ["dark"]


def test_match_any_vs_all(bus, clock):
    states = {"a/x": 5, "b/y": 50}
    any_rule = Rule("r", [Condition("a/x", "<", 10), Condition("b/y", "<", 10)], [], match="any")
    all_rule = Rule("r", [Condition("a/x", "<", 10), Condition("b/y", "<", 10)], [], match="all")
    now = clock.now()
    assert any_rule.holds(states, now)
    assert not all_rule.holds(states, now)


def test_missing_reading_or_wrong_type_is_false():
    assert not Condition("a/x", "<", 10).check({})
    assert not Condition("a/x", "<", 10).check({"a/x": "on"})


def test_time_window_rule_fires_inside_window(bus, clock):
    # The shape a learned rule will have: "weekdays 07:00-07:30, switch on".
    kettle_rule = Rule("morning", [], [Action("kitchen/kettle", "on")],
                       when=TimeWindow(time(7, 0), time(7, 30), weekdays=(0, 1, 2, 3, 4)), origin="learned")
    monday_7am = datetime(2026, 10, 12, 7, 10)
    saturday_7am = datetime(2026, 10, 17, 7, 10)
    assert kettle_rule.holds({}, monday_7am)
    assert not kettle_rule.holds({}, saturday_7am)
    assert not kettle_rule.holds({}, datetime(2026, 10, 12, 8, 0))


def test_window_across_midnight_belongs_to_its_start_day():
    friday_night = TimeWindow(time(23, 30), time(6, 30), weekdays=(4,))
    assert friday_night.check(datetime(2026, 10, 16, 23, 45))  # Fri 23:45
    assert friday_night.check(datetime(2026, 10, 17, 3, 0))  # Sat 03:00, still Friday's night
    assert not friday_night.check(datetime(2026, 10, 17, 7, 0))
    assert not friday_night.check(datetime(2026, 10, 16, 3, 0))  # Fri 03:00 belongs to Thursday's night
    assert friday_night.describe() == "Fri 23:30-06:30"
    assert TimeWindow(time(7), time(7, 30), (0, 1, 2, 3, 4)).describe() == "weekdays 07:00-07:30"


def test_two_conflicting_rules_settle_instead_of_flipping(bus, clock):
    # In Packet Tracer, "battery low -> window open" and "no smoke -> window closed" held at
    # once make the gateway flip the window forever. Here each rule acts once, on its change.
    from home.simulation import ArticleHome
    home = ArticleHome(bus, clock)
    engine = home.engine
    engine.add(Rule("smoke_clear_window", [Condition("garage/smoke", "<", 2)], [Action("living/window", "closed")]))
    home.run(seconds=10 * 60, dt=60)
    before = sum(1 for t, p in bus.published if t == "home/living/window/set")
    home.battery.reading = 5  # battery low: low_energy opens the window while smoke_clear holds
    home.run(seconds=30 * 60, dt=60)
    commands = [p["value"] for t, p in bus.published if t == "home/living/window/set"][before:]
    assert commands == ["open"]  # one action, then quiet: no flipping
    assert home.window.value == "open"
