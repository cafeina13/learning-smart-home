"""Replay a real REFIT household inside the simulated home.

Every recorded use of the household's appliances becomes what a person did:
switch it on at the recorded start, and (for appliances a person switches off,
like a TV or a computer) switch it off at the recorded end. Commands go over the
bus with source "resident", exactly like the synthetic resident's, so the rules,
the smart plugs, the override counting and the event log all see real behaviour.

Machines with smart start enabled are armed instead of started: the person
loads them at the recorded time and the house picks the start (they then run
their usual cycle length). Without smart start they start at the recorded time.

Times are replayed on their real calendar dates, so the simulation clock runs on
those dates too (and the solar panel can use that day's real weather).
"""

from datetime import datetime, time, timedelta

import pandas as pd

from home.devices import ArmableAppliance, PluggedAppliance, SimClock


class ReplayResident:
    def __init__(self, clock: SimClock, devices: dict, uses_by_device: dict[str, pd.DataFrame],
                 smart_start: dict[str, dict] | None = None):
        # devices: device id -> device; uses_by_device: device id -> uses (start, end) to replay.
        self.clock = clock
        self.devices = devices
        self.smart_start = smart_start if smart_start is not None else {}
        events = []
        for dev, uses in uses_by_device.items():
            for start, end in zip(uses["start"], uses["end"]):
                events.append((start.to_pydatetime(), 0, dev, "start"))
                if isinstance(devices[dev], PluggedAppliance):
                    events.append((end.to_pydatetime(), 1, dev, "end"))
        self._events = sorted(events)
        self._next = 0

    def skip_to(self, t: datetime) -> None:
        """Drop events before t (when the replay starts mid-way)."""
        while self._next < len(self._events) and self._events[self._next][0] < t:
            self._next += 1

    def act(self) -> None:
        """Call once per simulation step: perform every recorded action that is due."""
        now = self.clock.now()
        while self._next < len(self._events) and self._events[self._next][0] <= now:
            _, _, dev, what = self._events[self._next]
            self._next += 1
            device = self.devices[dev]
            if what == "end":
                if device.value == "on":
                    device.set("off", source="resident")
                continue
            setting = self.smart_start.get(dev)
            if isinstance(device, ArmableAppliance):
                if device.value != "off":
                    continue  # already armed or running
                if setting:
                    device.set("armed", source="resident", deadline=deadline(now, setting).isoformat())
                else:
                    device.set("on", source="resident")
            elif device.value != "on":
                device.set("on", source="resident")

    @property
    def remaining(self) -> int:
        return len(self._events) - self._next


def deadline(now: datetime, setting: dict) -> datetime:
    """The smart-start default deadline after `now`, e.g. 18:00, tomorrow if set so."""
    at = time.fromisoformat(setting["default_deadline"])
    d = datetime.combine(now.date() + timedelta(days=1 if setting.get("next_day") else 0), at)
    return d if d > now else d + timedelta(days=1)
