"""The article's three scenarios, checked end to end on the in-memory bus."""

from datetime import datetime

from home.devices import SimClock
from home.events import ALERTS_TOPIC
from home.simulation import ArticleHome


def alerts(bus):
    return [p for t, p in bus.published if t == ALERTS_TOPIC]


def test_scenario1_alert_when_one_room_is_much_hotter(bus, clock):
    home = ArticleHome(bus, clock)
    home.run(seconds=120)
    assert alerts(bus) == []  # all rooms equal: no alert

    # The article "consciously increased" one sensor's temperature.
    home.physics.extra_heat_c["kitchen/temperature"] = 12.0
    home.run(seconds=120)
    found = alerts(bus)
    assert len(found) == 1  # one alert, not one per report
    assert found[0]["device"] == "kitchen/temperature"

    home.physics.extra_heat_c.clear()
    home.run(seconds=120)
    home.physics.extra_heat_c["kitchen/temperature"] = 12.0
    home.run(seconds=120)
    assert len(alerts(bus)) == 2  # a new episode raises a new alert


def test_scenario2_smoke_level_published_every_60_seconds(bus, clock):
    home = ArticleHome(bus, clock)
    received = []
    bus.subscribe("home/garage/smoke/state", lambda t, p: received.append(p))  # the article's tablet
    home.run(seconds=600)
    assert len(received) == 10
    assert all(p["kind"] == "smoke" for p in received)


def test_scenario3_smoke_opens_house_then_closes_it(bus, clock):
    home = ArticleHome(bus, clock)
    home.announce()
    levels = []
    bus.subscribe("home/garage/smoke/state", lambda t, p: levels.append(p["value"]))
    home.car.set("on")  # the article starts the car to make smoke
    home.run(seconds=15 * 60)
    assert max(levels) > 10
    assert (home.door.value, home.garage_door.value, home.window.value, home.fan.value) == ("open", "open", "open", 2)
    # Ventilating works: smoke falls while the car still runs, but stays above 2, so the house does not flap shut.
    assert 2 < home.smoke.reading < 10

    home.car.set("off")
    home.run(seconds=60 * 60)
    assert home.smoke.reading < 2
    assert (home.door.value, home.garage_door.value, home.fan.value) == ("closed", "closed", 0)


def test_scenario3_low_battery_dims_lamp_and_opens_window(bus):
    clock = SimClock(datetime(2026, 10, 12, 21, 0))  # night: no solar
    home = ArticleHome(bus, clock)
    home.announce()
    home.lamp.set("on")
    home.run(seconds=120)
    assert home.lamp.value == "dim"  # solar <= 1 already holds at night
    assert home.window.value == "open"
    assert "low_energy" in [name for _, name in home.engine.fired]


def test_battery_drains_at_night_and_charges_by_day(bus):
    clock = SimClock(datetime(2026, 10, 12, 20, 0))
    home = ArticleHome(bus, clock)
    home.coffee.set("on")
    start = home.battery.reading
    home.run(seconds=3600)
    assert home.battery.reading < start  # 1 kW coffee machine, no sun

    home.coffee.set("off")
    clock.advance(14 * 3600)  # to 11:00 next day
    low = home.battery.reading
    home.run(seconds=3600)
    assert home.battery.reading > low


def test_added_appliance_runs_its_cycle_and_is_metered(bus, clock):
    # Noon: the panel gives about 400 W, so a 500 W washing machine draws the last 100 W from the grid.
    home = ArticleHome(bus, clock)
    washer = home.add_switch("utility/washing_machine", watts=500, run_minutes=60)
    washer.set("on")
    home.run(seconds=90 * 60)
    assert washer.value == "off"  # finished its cycle by itself
    assert abs(home.meter["load_kwh"] - 0.5) < 0.02
    assert home.meter["solar_kwh"] > 0.3  # most of it came from the panel around noon
    assert home.meter["grid_kwh"] > 0  # the rest from the grid (the battery only backs lamp and coffee)
