"""Smart start: when an armed machine should run, given the user's deadline and the solar forecast.

The user arms a machine (dishwasher, washing machine) and says when it must be
finished. The deadline is a hard limit; inside it, the scheduler looks at every
start time in 15-minute steps and scores each by how much of the run the solar
panel would cover. It then:

1. publishes up to three options (the best, plus well-separated alternatives)
   on "home/plans/<room>/<name>", so the dashboard can show them,
2. preselects the best one, so it works without anyone choosing,
3. lets the user pick another (`choose`), as long as it still meets the deadline,
4. starts the machine at the chosen time (a "scheduler" command, which the
   machine accepts only while it is armed).

If no start time gets any sun (a deadline before morning), the earliest start
is chosen, so the machine finishes as early as possible.

The forecast here is the simulation's clear-sky solar curve. A real home would
use a weather-based solar forecast; the scheduler only needs "watts at time t".
In the REFIT demo it is load-aware: the solar expected to be left after the
house's usual load at that hour of the week (dashboard/app.py), which put
53.6% instead of 46.6% of machine energy on solar in the evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from home.bus import Bus
from home.devices import ArmableAppliance, SimClock
from home.events import ROOT, device_id_from_topic, set_topic

PLANS_TOPIC = f"{ROOT}/plans"
SLOT = timedelta(minutes=15)
MIN_GAP = timedelta(hours=1)  # alternatives at least this far from each other


@dataclass
class Option:
    start: datetime
    solar_kwh: float
    solar_share: float  # 0..1 of the run's energy covered by the panel

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "solar_kwh": round(self.solar_kwh, 3),
                "solar_share": round(self.solar_share, 3)}


@dataclass
class Plan:
    device: str
    deadline: datetime
    latest_start: datetime
    options: list[Option]
    chosen: datetime
    chosen_by: str = "scheduler"  # "scheduler" (best, preselected) or "user"
    started: bool = False

    def to_dict(self) -> dict:
        return {"device": self.device, "deadline": self.deadline.isoformat(),
                "latest_start": self.latest_start.isoformat(),
                "options": [o.to_dict() for o in self.options],
                "chosen": self.chosen.isoformat(), "chosen_by": self.chosen_by}


def solar_coverage(start: datetime, minutes: float, watts: float, forecast: Callable[[datetime], float]) -> float:
    """kWh of a run starting at `start` that the panel would cover (5-minute resolution)."""
    kwh, t, end = 0.0, start, start + timedelta(minutes=minutes)
    while t < end:
        step = min(timedelta(minutes=5), end - t)
        kwh += min(watts, forecast(t)) * step.total_seconds() / 3600 / 1000
        t += step
    return kwh


def ceil_to_slot(t: datetime) -> datetime:
    base = t.replace(second=0, microsecond=0)
    extra = base.minute % 15
    return base if extra == 0 and t == base else base + timedelta(minutes=15 - extra)


def make_plan(device: str, now: datetime, deadline: datetime, watts: float, run_minutes: float,
              forecast: Callable[[datetime], float], n_options: int = 3) -> Plan:
    run = timedelta(minutes=run_minutes)
    latest = deadline - run
    first = ceil_to_slot(now)
    if latest < first:
        latest = first  # too late to make the deadline: start as soon as possible
    total_kwh = watts * run_minutes / 60 / 1000
    scored = []
    t = first
    while t <= latest:
        kwh = solar_coverage(t, run_minutes, watts, forecast)
        scored.append(Option(t, kwh, kwh / total_kwh if total_kwh else 0.0))
        t += SLOT

    # Best first (earlier wins ties); then alternatives at least MIN_GAP apart.
    ranked = sorted(scored, key=lambda o: (-round(o.solar_kwh, 4), o.start))
    if ranked[0].solar_kwh == 0:
        ranked = sorted(scored, key=lambda o: o.start)  # no sun at all: earliest first
    options: list[Option] = []
    for o in ranked:
        if all(abs(o.start - p.start) >= MIN_GAP for p in options):
            options.append(o)
        if len(options) == n_options:
            break
    return Plan(device, deadline, latest, options, chosen=options[0].start)


class SmartStart:
    def __init__(self, bus: Bus, clock: SimClock, machines: list[ArmableAppliance],
                 forecast: Callable[[datetime], float]):
        self.bus = bus
        self.clock = clock
        self.machines = {m.id: m for m in machines}
        self.forecast = forecast
        self.plans: dict[str, Plan] = {}
        bus.subscribe(f"{ROOT}/+/+/state", self._on_state)

    def add_machine(self, machine: ArmableAppliance) -> None:
        self.machines[machine.id] = machine

    def _on_state(self, topic: str, payload: dict) -> None:
        device = device_id_from_topic(topic)
        machine = self.machines.get(device)
        if machine is None:
            return
        if payload["value"] == "armed" and "deadline" in payload.get("extra", {}):
            deadline = datetime.fromisoformat(payload["extra"]["deadline"])
            self.plans[device] = make_plan(device, self.clock.now(), deadline, machine.watts,
                                           machine.run_minutes, self.forecast)
            self._publish(device)
        elif payload["value"] in ("on", "off"):
            self.plans.pop(device, None)  # started, finished, or cancelled

    def choose(self, device: str, start: datetime) -> bool:
        """The user picks a start time. It must be in the future and still meet the deadline."""
        plan = self.plans.get(device)
        if plan is None or plan.started or not (self.clock.now() <= start <= plan.latest_start):
            return False
        plan.chosen, plan.chosen_by = start, "user"
        self._publish(device)
        return True

    def tick(self) -> None:
        now = self.clock.now()
        for device, plan in list(self.plans.items()):
            if not plan.started and now >= plan.chosen:
                plan.started = True
                self.bus.publish(set_topic(device), {"value": "on", "ts": now.isoformat(), "source": "scheduler"})

    def _publish(self, device: str) -> None:
        self.bus.publish(f"{PLANS_TOPIC}/{device}", {**self.plans[device].to_dict(), "ts": self.clock.now().isoformat()})
