"""Simulated devices, one class per kind used in the article's scenarios and the added appliances.

Two families:

- Sensors only report. The simulation sets what they "measure" (`reading`), and
  they publish it on their state topic every `report_every` seconds, the way the
  article's smoke sensor reports every 60 s.
- Actuators obey commands. They listen on their .../set topic, check the command
  is one they accept, switch, and publish their new state.

Every actuator has a control type that says what AUTOMATIONS may do with it.
People (the user, the dashboard, the resident) can always operate a device; a
command from a rule or the scheduler is checked against the control type, and
refused if it is not allowed. The device enforces this itself, so no rule,
however it was learned, can do something unsafe.

    auto      automations may switch it freely (lamp, fan, window, a coffee
              machine with a timer)
    manual    needs a person to use it (TV, kettle). It sits behind a smart plug:
              automations may only cut its standby power or restore it
    armable   needs a person to prepare it (dishwasher, washing machine). The
              person arms it with a deadline; automations may then start it,
              and only then

Each device also knows its electrical power in each state (`power_w`).
"""

from datetime import datetime, timedelta
from typing import Any

from home.bus import Bus
from home.events import ALERTS_TOPIC, StateEvent, command, set_topic, state_topic


class SimClock:
    """Simulated time. The simulation moves it forward; nothing waits on the wall clock."""

    def __init__(self, start: datetime):
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def is_automation(source: str) -> bool:
    # Commands from rules ("rule:<name>"), the smart-start scheduler and the habit forest's
    # guard (habits/forest.py) are automations; everything else ("user", "dashboard",
    # "resident") is a person.
    return source.startswith("rule:") or source in ("scheduler", "forest")


class Device:
    kind = "device"

    def __init__(self, bus: Bus, clock: SimClock, room: str, name: str, value: Any, source: str = "sim"):
        self.bus = bus
        self.clock = clock
        self.id = f"{room}/{name}"
        self.room = room
        self.value = value
        self.source = source

    def state_extra(self) -> dict:
        # Device-specific details sent with the state (e.g. an armed machine's deadline).
        return {}

    def publish_state(self, **extra) -> None:
        event = StateEvent(self.id, self.kind, self.value, self.clock.now().isoformat(), self.source,
                           {**self.state_extra(), **extra})
        self.bus.publish(state_topic(self.id), event.to_payload())

    def power_w(self) -> float:
        return 0.0

    def tick(self, dt: float) -> None:
        # Called by the simulation every step; most devices do nothing.
        pass


class Sensor(Device):
    def __init__(self, bus, clock, room, name, reading: float = 0.0, report_every: float = 60.0):
        super().__init__(bus, clock, room, name, value=reading)
        self.reading = reading
        self.report_every = report_every
        self._since_report = report_every  # report on the first tick

    def tick(self, dt: float) -> None:
        self._since_report += dt
        if self._since_report >= self.report_every:
            self._since_report = 0.0
            self.value = round(self.reading, 2)
            self.publish_state()


class TemperatureSensor(Sensor):
    kind = "temperature"  # degrees Celsius


class SmokeSensor(Sensor):
    kind = "smoke"  # 0-100 units, as in the article's analogRead/10


class SolarPanel(Sensor):
    kind = "solar"  # current output in watts (Packet Tracer labels this value "Wh")


class Battery(Sensor):
    kind = "battery"  # state of charge, percent


class Actuator(Device):
    allowed: tuple = ()
    power_by_state: dict = {}
    control = "auto"

    def __init__(self, bus, clock, room, name, value):
        super().__init__(bus, clock, room, name, value)
        bus.subscribe(set_topic(self.id), self._on_set)

    def permitted(self, value: Any, source: str) -> bool:
        """May this source ask for this value now? People always may; automations depend on the control type."""
        return True

    def _on_set(self, topic: str, payload: dict) -> None:
        value, source = payload.get("value"), payload.get("source", "user")
        if value not in self.allowed:
            return  # ignore values this device cannot take
        if not self.permitted(value, source):
            self.bus.publish(ALERTS_TOPIC, {
                "kind": "refused", "device": self.id, "value": value, "source": source,
                "ts": self.clock.now().isoformat(),
            })
            return
        self.apply(value, source, payload)

    def apply(self, value: Any, source: str, payload: dict) -> None:
        if value != self.value:
            self.value = value
            self.publish_state()

    def set(self, value: Any, source: str = "user", **fields) -> None:
        # Send a command the same way anyone else would (scripts, tests, the resident).
        self.bus.publish(set_topic(self.id), {**command(value, self.clock.now(), source), **fields})

    def power_w(self) -> float:
        return float(self.power_by_state.get(self.value, 0.0))


class Lamp(Actuator):
    kind = "lamp"
    allowed = ("off", "dim", "on")
    power_by_state = {"off": 0, "dim": 20, "on": 60}

    def __init__(self, bus, clock, room, name, value="off"):
        super().__init__(bus, clock, room, name, value)


class Fan(Actuator):
    kind = "fan"
    allowed = (0, 1, 2)  # off, low, high (the article writes 2 for "high")
    power_by_state = {0: 0, 1: 30, 2: 60}

    def __init__(self, bus, clock, room, name, value=0):
        super().__init__(bus, clock, room, name, value)


class Opening(Actuator):
    # Doors, windows, the garage door: open or closed, no power draw.
    kind = "opening"
    allowed = ("open", "closed")

    def __init__(self, bus, clock, room, name, value="closed"):
        super().__init__(bus, clock, room, name, value)


class _Cycle:
    """Mixin: switch off by itself `run_minutes` after being switched on (a kettle boils, a wash ends)."""

    run_minutes: float | None = None
    _on_for = 0.0

    def tick(self, dt: float) -> None:
        if self.value != "on" or self.run_minutes is None:
            self._on_for = 0.0
            return
        self._on_for += dt
        if self._on_for >= self.run_minutes * 60:
            self._on_for = 0.0
            self.finished()

    def finished(self) -> None:
        self.value = "off"
        self.publish_state()


class Switch(_Cycle, Actuator):
    # An appliance automations may switch freely (control "auto"), e.g. a coffee
    # machine with a timer. `watts` is its draw when on.
    kind = "switch"
    allowed = ("off", "on")

    def __init__(self, bus, clock, room, name, watts: float, value="off", run_minutes: float | None = None):
        self.power_by_state = {"off": 0, "on": watts}
        self.run_minutes = run_minutes
        super().__init__(bus, clock, room, name, value)


class PluggedAppliance(_Cycle, Actuator):
    """A manual appliance (TV, kettle) behind a smart plug.

    States: "off" (plugged in, drawing standby power), "on" (in use), "cut" (the
    plug has cut all power). People use it normally. Automations may only cut or
    restore power:

    - A cut asked for while the appliance is in use is held back and applied
      when it is switched off, so nobody's film is interrupted.
    - If a person switches it on while the power is cut, the plug gives way
      (they pressed its button). The state carries "override": the house was
      wrong, and the rule engine counts it against the rule that cut.
    """

    kind = "plugged"
    control = "manual"
    allowed = ("off", "on", "cut")

    def __init__(self, bus, clock, room, name, watts: float, standby_w: float = 0.0,
                 run_minutes: float | None = None):
        self.power_by_state = {"off": standby_w, "on": watts, "cut": 0}
        self.run_minutes = run_minutes
        self.cut_pending = False
        super().__init__(bus, clock, room, name, "off")

    def permitted(self, value, source) -> bool:
        if not is_automation(source):
            return True
        return value in ("cut", "off")  # cut, or restore ("off" from an automation never switches it off)

    def apply(self, value, source, payload) -> None:
        if is_automation(source):
            if value == "cut":
                if self.value == "on":
                    self.cut_pending = True  # wait until it is switched off
                elif self.value != "cut":
                    self.value = "cut"
                    self.publish_state()
            else:  # restore: only does something if the power is cut
                self.cut_pending = False
                if self.value == "cut":
                    self.value = "off"
                    self.publish_state()
            return
        # A person.
        override = value == "on" and self.value == "cut"
        if value == "off" and self.cut_pending:
            value = "cut"  # switched off by hand while a cut was waiting: the plug cuts now
        self.cut_pending = False if value in ("on", "cut") else self.cut_pending
        if value != self.value:
            self.value = value
            self.publish_state(**({"override": True} if override else {}))

    def finished(self) -> None:
        self.value = "cut" if self.cut_pending else "off"
        self.cut_pending = False
        self.publish_state()


class ArmableAppliance(_Cycle, Actuator):
    """A machine a person must prepare (dishwasher, washing machine), with delay start.

    States: "off", "armed" (loaded and waiting, with a deadline), "on" (running).
    A person arms it with a deadline ("finish by 18:00"); the smart-start scheduler
    then picks the start time, and may start it only while it is armed. A person
    can also start it directly ("on") or cancel ("off").
    """

    kind = "armable"
    control = "armable"
    allowed = ("off", "armed", "on")

    def __init__(self, bus, clock, room, name, watts: float, standby_w: float = 0.0, run_minutes: float = 60):
        self.power_by_state = {"off": standby_w, "armed": standby_w, "on": watts}
        self.watts = watts
        self.run_minutes = run_minutes
        self.deadline: datetime | None = None
        super().__init__(bus, clock, room, name, "off")

    def state_extra(self) -> dict:
        return {"deadline": self.deadline.isoformat()} if self.value == "armed" and self.deadline else {}

    def permitted(self, value, source) -> bool:
        if not is_automation(source):
            return True
        return value == "on" and self.value == "armed"

    def apply(self, value, source, payload) -> None:
        if value == "armed":
            deadline = payload.get("deadline")
            if deadline is None:
                return  # arming needs a deadline
            self.deadline = datetime.fromisoformat(deadline)
        elif value in ("on", "off"):
            self.deadline = None
        if value != self.value or value == "armed":
            self.value = value
            self.publish_state()
