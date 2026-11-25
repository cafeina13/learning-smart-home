"""Automation rules in the format of the article's IoT gateway (its Fig. 11):

    IF  <match any | all>  of  [device  field  operator  value, ...]
    THEN set  [device -> value, ...]

Example, the article's battery rule:
    IF any of [solar <= 1, battery < 20]  THEN set [lamp -> dim, window -> open]

In the article every rule is typed in by hand. Here a rule is plain data, so a
person can write one, and later the habit model can propose one in the same
format. The `origin` field records which ("manual" or "learned").

Rules fire on the transition from false to true, not on every message while true.
Otherwise a rule would re-send its commands every time a sensor reports.
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Literal

from home.bus import Bus
from home.events import ALERTS_TOPIC, ROOT, command, device_id_from_topic, set_topic

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class Condition:
    device: str  # "<room>/<name>"
    op: str  # one of OPS
    value: Any

    def check(self, states: dict[str, Any]) -> bool:
        if self.device not in states:
            return False  # no reading yet: the condition cannot be true
        try:
            return OPS[self.op](states[self.device], self.value)
        except TypeError:
            return False  # e.g. comparing "on" < 20


@dataclass
class TimeWindow:
    # A daily time window, optionally limited to some weekdays (0 = Monday).
    # Learned rules use this, e.g. "weekdays, 07:00-07:30".
    # A window whose end is before its start crosses midnight ("23:30-06:30");
    # its weekdays name the day it starts on. start == end means a full 24 hours.
    start: time
    end: time
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    def check(self, now: datetime) -> bool:
        t, day = now.time(), now.weekday()
        if self.start < self.end:
            return day in self.weekdays and self.start <= t < self.end
        return (day in self.weekdays and t >= self.start) or ((day - 1) % 7 in self.weekdays and t < self.end)

    def describe(self) -> str:
        days = {(0, 1, 2, 3, 4): "weekdays", (5, 6): "weekends", (0, 1, 2, 3, 4, 5, 6): "every day"}
        names = days.get(tuple(self.weekdays)) or ", ".join("Mon Tue Wed Thu Fri Sat Sun".split()[d] for d in self.weekdays)
        if self.start == self.end:
            return f"{names} all day from {self.start:%H:%M}"
        return f"{names} {self.start:%H:%M}-{self.end:%H:%M}"


@dataclass
class Action:
    device: str
    value: Any


@dataclass
class Rule:
    name: str
    conditions: list[Condition]
    actions: list[Action]
    match: Literal["any", "all"] = "all"
    when: TimeWindow | None = None
    origin: Literal["manual", "learned"] = "manual"
    enabled: bool = True
    _was_true: bool = field(default=False, repr=False)

    def holds(self, states: dict[str, Any], now: datetime) -> bool:
        if self.when is not None and not self.when.check(now):
            return False
        if not self.conditions:
            return self.when is not None  # a pure time rule
        results = [c.check(states) for c in self.conditions]
        return any(results) if self.match == "any" else all(results)

    def to_dict(self) -> dict:
        # Plain JSON-able form, for storing suggestions and showing them in the dashboard.
        return {
            "name": self.name,
            "match": self.match,
            "conditions": [vars(c) for c in self.conditions],
            "actions": [vars(a) for a in self.actions],
            "when": None if self.when is None else {
                "start": self.when.start.strftime("%H:%M"),
                "end": self.when.end.strftime("%H:%M"),
                "weekdays": list(self.when.weekdays),
            },
            "origin": self.origin,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        when = d.get("when")
        return cls(
            name=d["name"],
            match=d.get("match", "all"),
            conditions=[Condition(**c) for c in d.get("conditions", [])],
            actions=[Action(**a) for a in d.get("actions", [])],
            when=None if when is None else TimeWindow(
                time.fromisoformat(when["start"]), time.fromisoformat(when["end"]), tuple(when["weekdays"])),
            origin=d.get("origin", "manual"),
            enabled=d.get("enabled", True),
        )


class RuleEngine:
    """Watches every state message and runs rules whose conditions just became true.

    It also learns from being wrong: when a person switches on an appliance whose
    power a rule had cut (an "override"), that rule gets a strike. So does a cut
    the habit forest ended early because it expected a use (habits/forest.py
    publishes "forest_restored"): the rule was about to be wrong, and the
    resident was spared only by the forest. After MAX_OVERRIDES strikes the rule
    is disabled, and an alert says so.
    """

    MAX_OVERRIDES = 3
    FOREST_STRIKES = True  # a forest rescue counts as a strike (False: only a person's override does)

    def __init__(self, bus: Bus, clock, rules: list[Rule] | None = None):
        self.bus = bus
        self.clock = clock
        self.rules: list[Rule] = list(rules or [])
        self.states: dict[str, Any] = {}  # latest value of every device
        self.fired: list[tuple[datetime, str]] = []  # (time, rule name), for logs and tests
        self.overrides: dict[str, int] = {}  # rule name -> strikes: a person's overrides plus forest rescues
        self.forest_strikes: dict[str, int] = {}  # rule name -> of which, cuts the forest ended early
        self._last_cut: dict[str, Rule] = {}  # device -> the rule that last cut its power
        bus.subscribe(f"{ROOT}/+/+/state", self._on_state)
        bus.subscribe(ALERTS_TOPIC, self._on_alert)

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def _on_state(self, topic: str, payload: dict) -> None:
        device = device_id_from_topic(topic)
        self.states[device] = payload["value"]
        if payload.get("extra", {}).get("override") and device in self._last_cut:
            self._count_override(self._last_cut[device])
        self.evaluate()

    def _on_alert(self, topic: str, payload: dict) -> None:
        if payload.get("kind") == "forest_restored" and self.FOREST_STRIKES and payload["device"] in self._last_cut:
            rule = self._last_cut[payload["device"]]
            self.forest_strikes[rule.name] = self.forest_strikes.get(rule.name, 0) + 1
            self._count_override(rule)

    def _count_override(self, rule: Rule) -> None:
        self.overrides[rule.name] = self.overrides.get(rule.name, 0) + 1
        if self.overrides[rule.name] >= self.MAX_OVERRIDES and rule.enabled:
            rule.enabled = False
            self.bus.publish(ALERTS_TOPIC, {
                "kind": "rule_disabled", "rule": rule.name, "overrides": self.overrides[rule.name],
                "forest": self.forest_strikes.get(rule.name, 0), "ts": self.clock.now().isoformat(),
            })

    def evaluate(self) -> None:
        # Also called by the simulation on every step, so time-window rules fire
        # even when no device happens to report at that moment.
        now = self.clock.now()
        for rule in self.rules:
            if not rule.enabled:
                continue
            is_true = rule.holds(self.states, now)
            if is_true and not rule._was_true:
                self.fired.append((now, rule.name))
                for action in rule.actions:
                    if action.value == "cut":
                        self._last_cut[action.device] = rule
                    self.bus.publish(set_topic(action.device), command(action.value, now, f"rule:{rule.name}"))
            rule._was_true = is_true
