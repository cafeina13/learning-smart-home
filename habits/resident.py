"""A simulated resident who lives in the home, following the synthetic household's habits.

Each simulated day the resident decides, from habits/synthetic.py's HABITS, which
appliances they will switch on and when. At those times they act as a person
would, by sending ordinary commands over the bus (source "resident"):

- most appliances: switch them on (if a smart plug has cut the power, switching
  on restores it: an override, which the rule engine notices);
- a machine with smart start enabled: load it and arm it with the default
  deadline instead of starting it, and let the house pick the start time.

The appliances switch themselves off when their cycle ends.

The point: the home's own event log then contains realistic appliance use, so
habit learning can run on the home's own data (habits/from_log.py), the way it
would in a real house. Because the habits are known, we can check that the
system rediscovers them.

Like the synthetic household, this is for demos and tests only.
"""

from datetime import date, datetime, time, timedelta

import numpy as np

from habits.suggestions import device_id
from habits.synthetic import HABITS, Habit
from home.devices import ArmableAppliance, SimClock


class Resident:
    def __init__(self, clock: SimClock, switches: dict, habits: list[Habit] = HABITS, seed: int = 0,
                 smart_start: dict[str, dict] | None = None):
        # switches: device id -> device for every appliance the habits mention.
        # smart_start: device id -> accepted setting {"default_deadline": "18:00", "next_day": bool};
        # the dict is shared with whoever accepts suggestions, so changes take effect at once.
        self.clock = clock
        self.switches = switches
        self.smart_start = smart_start if smart_start is not None else {}
        self.habits = [h for h in habits if device_id(h.appliance) in switches]
        self.rng = np.random.default_rng(seed)
        self._day: date | None = None
        self._plan: list[tuple[datetime, str]] = []  # (when, device id), sorted

    def _plan_day(self, day: date) -> None:
        plan = []
        for h in self.habits:
            if day.weekday() not in h.weekdays or self.rng.random() > h.probability:
                continue
            minutes = float(np.clip(h.hour * 60 + self.rng.normal(0, h.spread_min), 0, 24 * 60 - 1))
            plan.append((datetime.combine(day, datetime.min.time()) + timedelta(minutes=minutes), device_id(h.appliance)))
        self._plan = sorted(plan)
        self._day = day

    def act(self) -> None:
        """Call once per simulation step: switches on whatever is due."""
        now = self.clock.now()
        if self._day != now.date():
            self._plan_day(now.date())
        while self._plan and self._plan[0][0] <= now:
            _, dev = self._plan.pop(0)
            device = self.switches[dev]
            setting = self.smart_start.get(dev)
            if isinstance(device, ArmableAppliance) and setting:
                if device.value == "off":
                    device.set("armed", source="resident", deadline=self._deadline(now, setting).isoformat())
            elif device.value not in ("on", "armed"):
                device.set("on", source="resident")

    @staticmethod
    def _deadline(now: datetime, setting: dict) -> datetime:
        at = time.fromisoformat(setting["default_deadline"])
        deadline = datetime.combine(now.date() + timedelta(days=1 if setting.get("next_day") else 0), at)
        return deadline if deadline > now else deadline + timedelta(days=1)
