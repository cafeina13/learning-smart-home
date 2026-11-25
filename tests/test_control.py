"""Control types: what automations may and may not do, smart start, and learning from overrides."""

from datetime import datetime, time, timedelta

from home.events import ALERTS_TOPIC
from home.rules import Action, Rule, RuleEngine, TimeWindow
from home.scheduler import make_plan
from home.simulation import ArticleHome, clear_sky_solar_w


def alerts(bus, kind):
    return [p for t, p in bus.published if t == ALERTS_TOPIC and p.get("kind") == kind]


# --- manual appliances behind a smart plug ---------------------------------------------------

def test_automation_can_never_switch_a_tv_on(bus, clock):
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=9)
    tv.set("on", source="rule:bad idea")
    assert tv.value == "off"
    assert alerts(bus, "refused")[0]["device"] == "living/tv"
    tv.set("on", source="user")  # a person can
    assert tv.value == "on"


def test_cut_waits_until_the_tv_is_switched_off(bus, clock):
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=9, run_minutes=30)
    tv.set("on", source="user")
    tv.set("cut", source="rule:night")  # someone is still watching
    assert tv.value == "on" and tv.power_w() == 90
    home.run(seconds=31 * 60)  # the programme ends
    assert tv.value == "cut" and tv.power_w() == 0


def test_switching_off_applies_a_waiting_cut(bus, clock):
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=12)  # no cycle: a person switches it off
    tv.set("on", source="resident")
    tv.set("cut", source="rule:night")  # waits: someone is watching
    assert tv.value == "on"
    tv.set("off", source="resident")
    assert tv.value == "cut" and tv.power_w() == 0


def test_restore_only_acts_on_a_cut_plug(bus, clock):
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=9)
    tv.set("on", source="user")
    tv.set("off", source="rule:morning")  # "restore" must not switch off a TV in use
    assert tv.value == "on"
    assert alerts(bus, "refused") == []


def test_three_overrides_disable_the_cutting_rule(bus, clock):
    home = ArticleHome(bus, clock)
    tv = home.add_appliance("living/tv", "manual", watts=90, standby_w=9)
    rule = Rule("tv night cut", [], [Action("living/tv", "cut")],
                when=TimeWindow(time(12, 0), time(12, 30)), origin="learned")
    home.engine.add(rule)
    first_day = clock.now().date()
    for day in range(3):
        target = datetime.combine(first_day + timedelta(days=day + 1), time(11, 55))
        clock.advance((target - clock.now()).total_seconds())
        home.run(seconds=10 * 60)  # into the window: the rule cuts
        assert tv.value == "cut"
        tv.set("on", source="user")  # the person wants TV anyway: override
        tv.set("off", source="user")
        home.run(seconds=30 * 60)  # leave the window, so it can fire again tomorrow
    assert home.engine.overrides[rule.name] == 3
    assert not rule.enabled
    assert alerts(bus, "rule_disabled")[0]["rule"] == rule.name


# --- armable machines and smart start ---------------------------------------------------------

def test_machine_cannot_be_started_by_automation_unless_armed(bus, clock):
    home = ArticleHome(bus, clock)
    dishwasher = home.add_appliance("kitchen/dishwasher", "armable", watts=600, run_minutes=120)
    dishwasher.set("on", source="scheduler")
    assert dishwasher.value == "off"
    dishwasher.set("armed", source="user")  # arming needs a deadline
    assert dishwasher.value == "off"


def test_plan_prefers_solar_hours_and_meets_the_deadline():
    forecast = lambda t: clear_sky_solar_w(t, 400)
    now = datetime(2026, 10, 12, 21, 0)
    plan = make_plan("kitchen/dishwasher", now, datetime(2026, 10, 13, 18, 0), 600, 120, forecast)
    best = plan.options[0]
    assert time(10, 0) <= best.start.time() <= time(12, 0)  # the run is centred on the sunniest hours
    assert best.start + timedelta(minutes=120) <= plan.deadline
    assert plan.chosen == best.start and len(plan.options) == 3
    starts = sorted(o.start for o in plan.options)
    assert all(b - a >= timedelta(hours=1) for a, b in zip(starts, starts[1:]))


def test_night_deadline_starts_as_early_as_possible():
    forecast = lambda t: clear_sky_solar_w(t, 400)
    now = datetime(2026, 10, 12, 21, 7)
    plan = make_plan("kitchen/dishwasher", now, datetime(2026, 10, 13, 5, 0), 600, 120, forecast)
    assert plan.chosen == datetime(2026, 10, 12, 21, 15)
    assert plan.options[0].solar_kwh == 0


def test_armed_dishwasher_runs_at_the_chosen_slot(bus):
    from home.devices import SimClock
    clock = SimClock(datetime(2026, 10, 12, 21, 0))
    home = ArticleHome(bus, clock)
    dishwasher = home.add_appliance("kitchen/dishwasher", "armable", watts=600, standby_w=1, run_minutes=120)
    dishwasher.set("armed", source="user", deadline=datetime(2026, 10, 13, 18, 0).isoformat())
    assert dishwasher.value == "armed"
    plan = home.scheduler.plans["kitchen/dishwasher"]

    # The user prefers the second suggestion; picking a time past the latest start is refused.
    second = plan.options[1].start
    assert not home.scheduler.choose("kitchen/dishwasher", plan.latest_start + timedelta(minutes=15))
    assert home.scheduler.choose("kitchen/dishwasher", second)

    home.run(seconds=(second - clock.now()).total_seconds() - 600, dt=60)
    assert dishwasher.value == "armed"  # not yet
    home.run(seconds=20 * 60, dt=60)
    assert dishwasher.value == "on"
    home.run(seconds=130 * 60, dt=60)
    assert dishwasher.value == "off" and clock.now() < plan.deadline
    assert "kitchen/dishwasher" not in home.scheduler.plans


def test_load_aware_smart_start_avoids_the_hours_the_house_uses_the_sun_itself():
    from home.scheduler import make_plan
    from home.simulation import clear_sky_solar_w
    day = datetime(2026, 6, 15, 8, 0)
    sun = lambda t: clear_sky_solar_w(t, 2000.0)  # noqa: E731
    lunch = lambda t: 1800.0 if 11 <= t.hour < 14 else 100.0  # noqa: E731 (cooking at midday)
    solar_only = make_plan("kitchen/dishwasher", day, day.replace(hour=18), 2000, 120, sun)
    load_aware = make_plan("kitchen/dishwasher", day, day.replace(hour=18), 2000, 120,
                           lambda t: max(0.0, sun(t) - lunch(t)))
    runs = lambda p: (p.chosen, p.chosen + timedelta(hours=2))  # noqa: E731
    start, end = runs(solar_only)
    assert start.hour < 12 < end.hour + (end.minute > 0)  # around noon, the sunniest hours
    start, end = runs(load_aware)
    assert end <= day.replace(hour=11) or start >= day.replace(hour=14)  # around lunch, not during it
