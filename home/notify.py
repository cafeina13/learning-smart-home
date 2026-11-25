"""Phone notifications: the home's alerts and smart-start plans, sent to the resident through ntfy.

The article's home raises alerts that only its own screen shows. Here a
notifier listens on the bus like every other part of the home and turns the
events a person should know about into short phone messages:

    smoke_high fired       the article's scenario 3: smoke in the garage, doors
                           and windows opened, fan on               (urgent)
    temperature_deviation  a room much hotter than the others        (high)
    rule_disabled          a learned rule switched itself off after 3 strikes (overrides or
                           cuts the habit forest ended early)
    refused                an automation was not allowed to switch something
    forest_restored        the habit forest ended a standby cut early (habits/forest.py)
    smart-start plan       a machine was armed: when it will start, how solar
    scheduler start        an armed machine started on schedule

ntfy (https://ntfy.sh, or a self-hosted server) delivers an HTTP POST to the
phone app subscribed to the same topic. The address is read from the NTFY_URL
environment variable and never stored in the code: anyone who knows a topic can
read it. Messages are sent from a background thread, so a slow network never
holds up the home; if the queue backs up (e.g. while skipping simulated days),
new messages are dropped rather than delivered late.
"""

import os
import queue
import threading
import urllib.request
from datetime import datetime
from typing import Callable

from home.bus import Bus
from home.events import ALERTS_TOPIC, ROOT, device_id_from_topic
from home.scheduler import PLANS_TOPIC

Sender = Callable[[str, str, str, str], None]  # (title, message, priority, tags)


def nice_name(device: str) -> str:
    """'kitchen/washing_machine' -> 'Washing machine (kitchen)'."""
    room, _, name = device.partition("/")
    return f"{name.replace('_', ' ').capitalize()} ({room})"


def _time(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%H:%M")


class PhoneNotifier:
    def __init__(self, bus: Bus, send: Sender):
        self.send = send
        self.paused = False  # e.g. while the dashboard skips days at full speed
        self._plans: dict[str, dict] = {}  # device -> latest smart-start plan
        bus.subscribe(ALERTS_TOPIC, self._on_alert)
        bus.subscribe(f"{PLANS_TOPIC}/#", self._on_plan)
        bus.subscribe(f"{ROOT}/+/+/set", self._on_command)

    def _notify(self, title: str, message: str, priority: str = "default", tags: str = "") -> None:
        if not self.paused:
            self.send(title, message, priority, tags)

    def _on_alert(self, topic: str, a: dict) -> None:
        kind = a.get("kind")
        if kind == "temperature_deviation":
            self._notify(f"Too hot: {nice_name(a['device'])}",
                         f"{a['value']:.1f} °C, the home's average is {a['average']:.1f} °C ({_time(a['ts'])}).",
                         "high", "thermometer")
        elif kind == "rule_disabled":
            forest = a.get("forest", 0)
            why = (f"you overrode it {a['overrides']} times" if not forest else
                   f"it was wrong {a['overrides']} times (you overrode it {a['overrides'] - forest}, "
                   f"the habit model stepped in {forest})")
            self._notify("Automation switched off",
                         f"'{a['rule']}' is off: {why}. "
                         "It stays off.", "default", "no_entry")
        elif kind == "forest_restored":
            self._notify(f"Power back on: {nice_name(a['device'])}",
                         f"The habit model expects you to use it this hour ({a['probability']:.0%}), "
                         f"so its standby cut ended early ({_time(a['ts'])}).", "low", "electric_plug")
        elif kind == "refused":
            self._notify(f"Not switched: {nice_name(a['device'])}",
                         f"{a['source']} asked for '{a['value']}' at {_time(a['ts'])}; automations may not do "
                         "that on this device (a machine must be armed first, a TV only you switch on).",
                         "low", "hand")

    def _on_plan(self, topic: str, plan: dict) -> None:
        device = plan["device"]
        before = self._plans.get(device)
        self._plans[device] = plan
        if before and before["chosen"] == plan["chosen"]:
            return  # republished, nothing new for the person
        chosen = next((o for o in plan["options"] if o["start"] == plan["chosen"]), None)
        solar = f", {chosen['solar_share']:.0%} on solar" if chosen else ""
        verb = "Moved: starts" if before else "Armed: starts"
        self._notify(f"Smart start: {nice_name(device)}",
                     f"{verb} at {_time(plan['chosen'])}{solar}; done before {_time(plan['deadline'])}.",
                     "default", "sunny")

    def _on_command(self, topic: str, c: dict) -> None:
        device, source = device_id_from_topic(topic), c.get("source", "")
        if source == "scheduler" and c.get("value") == "on":
            self._plans.pop(device, None)
            self._notify(f"Started: {nice_name(device)}", f"Started on schedule at {_time(c['ts'])}.",
                         "low", "white_check_mark")
        elif source == "rule:smoke_high" and device == "living/fan":  # once per firing: the rule's last action
            self._notify("SMOKE in the garage",
                         f"Smoke detected at {_time(c['ts'])}. Doors and window opened, fan on.",
                         "urgent", "rotating_light")


class NtfySender:
    """Posts messages to an ntfy address from a background thread."""

    def __init__(self, address: str, max_queue: int = 20):
        self.address = address
        self.sent = 0
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        threading.Thread(target=self._run, daemon=True, name="ntfy").start()

    def __call__(self, title: str, message: str, priority: str = "default", tags: str = "") -> None:
        try:
            self._queue.put_nowait((title, message, priority, tags))
        except queue.Full:
            pass  # too many at once: drop rather than deliver late

    def _run(self) -> None:
        while True:
            title, message, priority, tags = self._queue.get()
            request = urllib.request.Request(
                self.address, data=message.encode("utf-8"), method="POST",
                headers={"Title": title.encode("ascii", "replace").decode(), "Priority": priority, "Tags": tags})
            try:
                urllib.request.urlopen(request, timeout=10).read()
                self.sent += 1
            except Exception as e:  # noqa: BLE001 - a lost notification must not break the home
                print(f"[ntfy] not sent: {e}", flush=True)


def attach_from_env(bus: Bus) -> PhoneNotifier | None:
    """A notifier if NTFY_URL is set, otherwise None (the home runs the same either way)."""
    address = os.environ.get("NTFY_URL")
    return PhoneNotifier(bus, NtfySender(address)) if address else None
