"""The article's home, rebuilt: its devices, its three scenarios, and simple physics.

Scenario 1 (M2M temperature alert): four temperature sensors; a controller
    computes the average and raises an alert when one sensor is far above it.
    The article sends that alert to a server over UDP; here it goes to the
    "home/alerts" topic.
Scenario 2 (MQTT smoke reporting): the garage smoke sensor publishes its level
    every 60 s; any subscriber (the article's tablet, our dashboard and event
    log) receives it.
Scenario 3 (gateway with solar and battery): a solar panel charges a battery
    that powers the lamp and the coffee machine. Three hand-written rules,
    copied from the article:
      - smoke above 10: open the house door, garage door and window, fan on high
      - smoke below 2: close them again, fan off
      - solar at or below 1 or battery below 20%: dim the lamp, open the window

The physics is deliberately simple. It only has to produce believable sensor
readings for the rules to react to; the article's contribution is the
architecture, not a building model.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from home.bus import Bus
from home.devices import (
    ArmableAppliance,
    PluggedAppliance,
    Battery,
    Device,
    Fan,
    Lamp,
    Opening,
    SimClock,
    SmokeSensor,
    SolarPanel,
    Switch,
    TemperatureSensor,
)
from home.events import ALERTS_TOPIC, ROOT, device_id_from_topic
from home.rules import Action, Condition, Rule, RuleEngine
from home.scheduler import SmartStart

# Scenario 1: how far above the average (in degrees C) counts as an alert.
TEMP_ALERT_DELTA = 5.0


class TemperatureMonitor:
    """Scenario 1's controller (the article's microcontroller).

    It keeps the latest reading of each temperature sensor. When one rises more
    than TEMP_ALERT_DELTA above the average of all of them, it publishes one
    alert, and stays quiet until that sensor has come back down.
    """

    def __init__(self, bus: Bus, clock: SimClock, sensor_ids: list[str], delta: float = TEMP_ALERT_DELTA):
        self.bus = bus
        self.clock = clock
        self.sensor_ids = set(sensor_ids)
        self.delta = delta
        self.latest: dict[str, float] = {}
        self._alerting: set[str] = set()
        bus.subscribe(f"{ROOT}/+/+/state", self._on_state)

    def _on_state(self, topic: str, payload: dict) -> None:
        device = device_id_from_topic(topic)
        if device not in self.sensor_ids:
            return
        self.latest[device] = payload["value"]
        if len(self.latest) < len(self.sensor_ids):
            return  # wait until every sensor has reported once
        average = sum(self.latest.values()) / len(self.latest)
        for sensor, value in self.latest.items():
            too_hot = value > average + self.delta
            if too_hot and sensor not in self._alerting:
                self._alerting.add(sensor)
                self.bus.publish(ALERTS_TOPIC, {
                    "kind": "temperature_deviation",
                    "device": sensor,
                    "value": value,
                    "average": round(average, 2),
                    "ts": self.clock.now().isoformat(),
                })
            elif not too_hot:
                self._alerting.discard(sensor)


def article_rules() -> list[Rule]:
    # Scenario 3's rules, exactly as the article states them.
    openings = ["hall/door", "garage/door", "living/window"]
    return [
        Rule(
            name="smoke_high",
            conditions=[Condition("garage/smoke", ">", 10)],
            actions=[Action(d, "open") for d in openings] + [Action("living/fan", 2)],
        ),
        Rule(
            name="smoke_clear",
            conditions=[Condition("garage/smoke", "<", 2)],
            actions=[Action(d, "closed") for d in openings] + [Action("living/fan", 0)],
        ),
        Rule(
            name="low_energy",
            match="any",
            conditions=[Condition("roof/solar", "<=", 1), Condition("utility/battery", "<", 20)],
            actions=[Action("living/lamp", "dim"), Action("living/window", "open")],
        ),
    ]


def clear_sky_solar_w(t, peak_w: float) -> float:
    """Panel output on a clear day: a sine from 06:00 to 18:00 peaking at noon.

    Used both as the simulated panel and as the scheduler's forecast. (A real
    home would read the panel and use a weather-based forecast.)
    """
    hour = t.hour + t.minute / 60
    return max(0.0, peak_w * math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0


@dataclass
class Physics:
    # Smoke: the running car adds smoke; it clears faster with openings and the fan.
    smoke_per_min_car: float = 2.0
    smoke_clear_rate_closed: float = 0.02  # fraction per minute
    smoke_clear_rate_per_opening: float = 0.15
    smoke_clear_rate_per_fan_level: float = 0.15
    # Solar panel and battery.
    solar_peak_w: float = 400.0  # at noon; zero before 06:00 and after 18:00
    battery_capacity_wh: float = 2000.0
    # Scenario 1: indoor temperature and any extra heat injected at one sensor.
    base_temp_c: float = 21.0
    extra_heat_c: dict[str, float] = field(default_factory=dict)


class ArticleHome:
    """All devices, controllers and rules from the article, wired to one bus."""

    def __init__(self, bus: Bus, clock: SimClock, physics: Physics | None = None, report_every: float = 60.0,
                 solar: Callable[[datetime], float] | None = None):
        # solar: panel output in watts at a given time. By default the clear-sky curve;
        # the REFIT demo passes real PVGIS output for the replayed dates.
        self.bus = bus
        self.clock = clock
        self.physics = physics or Physics()
        self.solar_w = solar or (lambda t: clear_sky_solar_w(t, self.physics.solar_peak_w))
        b, c = bus, clock

        self.temps = [TemperatureSensor(b, c, room, "temperature", self.physics.base_temp_c, report_every)
                      for room in ("living", "bedroom", "kitchen", "hall")]
        self.smoke = SmokeSensor(b, c, "garage", "smoke", 0.0, report_every)
        self.car = Switch(b, c, "garage", "car", watts=0)
        self.garage_door = Opening(b, c, "garage", "door")
        self.door = Opening(b, c, "hall", "door")
        self.window = Opening(b, c, "living", "window")
        self.fan = Fan(b, c, "living", "fan")
        self.lamp = Lamp(b, c, "living", "lamp")
        self.coffee = Switch(b, c, "kitchen", "coffee_machine", watts=1000)
        self.solar = SolarPanel(b, c, "roof", "solar", 0.0, report_every)
        self.battery = Battery(b, c, "utility", "battery", 80.0, report_every)

        self.devices: list[Device] = [
            *self.temps, self.smoke, self.car, self.garage_door, self.door, self.window,
            self.fan, self.lamp, self.coffee, self.solar, self.battery,
        ]
        self.monitor = TemperatureMonitor(bus, clock, [t.id for t in self.temps])
        self.engine = RuleEngine(bus, clock, article_rules())
        self.scheduler = SmartStart(bus, clock, [], self.solar_w)
        # Energy meters, in kWh since the simulation started.
        self.meter = {"load_kwh": 0.0, "solar_kwh": 0.0, "battery_kwh": 0.0, "grid_kwh": 0.0}

    def add_switch(self, device_id: str, watts: float, run_minutes: float | None = None) -> Switch:
        """Add an appliance beyond the article's (e.g. the ones habit learning makes suggestions for)."""
        room, name = device_id.split("/")
        existing = next((d for d in self.devices if d.id == device_id), None)
        if isinstance(existing, Switch):
            # e.g. the article's own coffee machine: keep it, but give it the cycle length.
            existing.run_minutes = run_minutes if run_minutes is not None else existing.run_minutes
            return existing
        device = Switch(self.bus, self.clock, room, name, watts=watts, run_minutes=run_minutes)
        self.devices.append(device)
        return device

    def add_appliance(self, device_id: str, control: str, watts: float, standby_w: float = 0.0,
                      run_minutes: float | None = None):
        """Add an appliance with its control type: "auto", "manual" (behind a smart plug) or "armable"."""
        if control == "auto":
            return self.add_switch(device_id, watts, run_minutes)
        existing = next((d for d in self.devices if d.id == device_id), None)
        if existing is not None:
            return existing
        room, name = device_id.split("/")
        if control == "manual":
            device = PluggedAppliance(self.bus, self.clock, room, name, watts, standby_w, run_minutes)
        elif control == "armable":
            device = ArmableAppliance(self.bus, self.clock, room, name, watts, standby_w, run_minutes or 60)
            self.scheduler.add_machine(device)
        else:
            raise ValueError(f"unknown control type {control!r}")
        self.devices.append(device)
        return device

    def battery_load_w(self) -> float:
        # The article powers the lamp and the coffee machine from the battery.
        return self.lamp.power_w() + self.coffee.power_w()

    def load_w(self) -> float:
        return sum(d.power_w() for d in self.devices)

    def _update_physics(self, dt: float) -> None:
        p = self.physics
        minutes = dt / 60.0

        for sensor in self.temps:
            sensor.reading = p.base_temp_c + p.extra_heat_c.get(sensor.id, 0.0)

        openings = sum(o.value == "open" for o in (self.garage_door, self.door, self.window))
        clear_rate = (p.smoke_clear_rate_closed + openings * p.smoke_clear_rate_per_opening
                      + self.fan.value * p.smoke_clear_rate_per_fan_level)
        smoke = self.smoke.reading * math.exp(-clear_rate * minutes)
        if self.car.value == "on":
            smoke += p.smoke_per_min_car * minutes
        self.smoke.reading = min(smoke, 100.0)

        self.solar.reading = self.solar_w(self.clock.now())

        # Energy flow: the solar panel serves the house first, any surplus charges the
        # battery, and what solar cannot cover comes from the battery (only for the
        # article's battery-backed lamp and coffee machine, while it has charge) or the grid.
        hours = dt / 3600
        load = self.load_w()
        from_solar = min(load, self.solar.reading)
        surplus = self.solar.reading - from_solar
        deficit = load - from_solar
        from_battery = min(deficit, self.battery_load_w()) if self.battery.reading > 0 else 0.0
        soc = self.battery.reading + 100 * (surplus - from_battery) * hours / p.battery_capacity_wh
        self.battery.reading = min(max(soc, 0.0), 100.0)

        self.meter["load_kwh"] += load * hours / 1000
        self.meter["solar_kwh"] += from_solar * hours / 1000
        self.meter["battery_kwh"] += from_battery * hours / 1000
        self.meter["grid_kwh"] += (deficit - from_battery) * hours / 1000

    def step(self, dt: float) -> None:
        """Advance simulated time by dt seconds: physics, then device reports, then time rules."""
        self.clock.advance(dt)
        self._update_physics(dt)
        for device in self.devices:
            device.tick(dt)
        self.bus.flush()
        self.engine.evaluate()
        self.scheduler.tick()
        self.bus.flush()

    def run(self, seconds: float, dt: float = 10.0) -> None:
        for _ in range(int(seconds // dt)):
            self.step(dt)

    def announce(self) -> None:
        # Every actuator publishes its starting state once, so subscribers know it.
        for device in self.devices:
            if not hasattr(device, "reading"):
                device.publish_state()
        self.bus.flush()
