"""The random forest inside the running home: it checks suggestions and guards standby cuts.

The research path (habits/models.py) shows that a random forest predicts "will
this appliance be switched on in this hour?" better than the hour-of-week
baseline. Here the same forest, with the same 15 past-only features
(habits/features.py), does two jobs in the system itself:

1. Check (before a suggestion reaches the dashboard). Routines and quiet
   windows are found by counting (habits/routines.py), which a person can read
   and confirm. The forest then checks each one on recent days it did not learn
   from: for a routine, on how many of those days it also predicts a use in the
   window; for a standby cut, on how many it predicts none. That share is shown
   with the suggestion, and one the forest agrees with on fewer than
   MIN_AGREEMENT of the days is held back.
2. Guard (continuously, while a standby cut is on). At the start of every hour
   in which a device's power is cut, the forest predicts whether it will be
   switched on in that hour, from what happened before it (used recently, the
   rest of the house active). If so, power is restored ahead of the person, and
   stays on until the window ends. Counting finds the usual quiet hours; the
   forest catches the unusual day.

The learning days are split like the evaluation: the forest trains on the first
60%, its decision threshold is the best-F1 one on the next 20% (as in
habits/models.py), and the check uses the last 20%, which it never saw.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from habits.dataset import hourly_labels
from habits.features import build_features
from habits.models import best_threshold, make_model
from home.bus import Bus
from home.events import ALERTS_TOPIC, ROOT, command, device_id_from_topic, set_topic

MIN_AGREEMENT = 0.5  # a suggestion the forest agrees with on fewer of the check days is held back
SOURCE = "forest"  # the source of the guard's restore commands (an automation: home/devices.py)


@dataclass
class HabitForest:
    model: object
    threshold: float
    check_days: list[date]  # the last 20% of the learning days, never trained on
    probability: pd.Series  # P(switched on in this hour) for every learning hour, from earlier hours only

    @classmethod
    def fit(cls, uses: pd.DataFrame, hours: pd.DatetimeIndex, other_starts: pd.Series | None = None,
            ) -> "HabitForest | None":
        """Train on an appliance's uses over the given (covered) learning hours. None if there is
        nothing to learn (no uses in the training or threshold days)."""
        hours = pd.DatetimeIndex(hours).sort_values()
        days = sorted(set(hours.date))
        if len(days) < 5:
            return None
        a, b = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
        labels = hourly_labels(uses, hours)
        X = build_features(labels, other_starts)
        train = labels.index < pd.Timestamp(a)
        val = (labels.index >= pd.Timestamp(a)) & (labels.index < pd.Timestamp(b))
        if labels[train].nunique() < 2 or labels[val].sum() == 0:
            return None
        model = make_model("random_forest").fit(X[train], labels[train])
        p = pd.Series(model.predict_proba(X)[:, 1], index=labels.index)
        threshold = best_threshold(labels[val], p[val].to_numpy())
        return cls(model, threshold, [d for d in days if d >= b], p)

    def predict(self, labels: pd.Series, other_starts: pd.Series | None = None) -> pd.Series:
        """P(switched on) for every hour of `labels` (0/1 per covered hour), each from earlier hours only."""
        X = build_features(labels, other_starts)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=labels.index)

    def agreement(self, weekdays: tuple[int, ...], start_slot: int, n_slots: int, expect_use: bool) -> float:
        """Share of the check days (of these weekdays) on which the forest agrees with the window:
        it predicts a switch-on in it (a routine, expect_use=True) or none (a quiet window).
        A window may run past midnight. NaN if no check day has data in the window."""
        verdicts = []
        for d in self.check_days:
            if d.weekday() not in weekdays:
                continue
            p = self.probability.reindex(window_hours(d, start_slot, n_slots)).dropna()
            if len(p):
                verdicts.append(bool((p >= self.threshold).any()) == expect_use)
        return float(np.mean(verdicts)) if verdicts else float("nan")


def window_hours(day: date, start_slot: int, n_slots: int) -> pd.DatetimeIndex:
    """The clock hours a window of half-hour slots touches, e.g. 07:00-07:30 -> [07:00]."""
    start = pd.Timestamp(day) + pd.Timedelta(minutes=30 * start_slot)
    end = start + pd.Timedelta(minutes=30 * n_slots)
    return pd.date_range(start.floor("h"), (end - pd.Timedelta(minutes=1)).floor("h"), freq="h")


def restore_time(start: datetime, end: datetime, probability: pd.Series, threshold: float) -> datetime:
    """When the guard would restore a cut that runs from `start` to `end`: the start of the first
    hour the forest flags (the window's start if that hour is flagged), or `end` if none is."""
    p = probability.reindex(pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end), freq="h",
                                          inclusive="left")).dropna()
    flagged = p.index[p.to_numpy() >= threshold]
    return max(start, flagged[0].to_pydatetime()) if len(flagged) else end


@dataclass
class Guarded:
    forest: HabitForest
    labels: pd.Series  # 0/1 per covered hour of the learning period
    others: pd.Series  # the rest of the house's switch-ons per hour, learning period (as in training)


class ForestGuard:
    """Job 2, in the running home: one more subscriber on the bus, like the rule engine.

    It watches the household's appliances, so it knows when each was switched on.
    Each hour, and whenever a guarded device's power is cut, it asks that
    device's forest about the current hour, with the same inputs as in training
    (its own past uses, the rest of the house's). A "likely" answer restores the
    power: an automation command with source "forest".
    """

    def __init__(self, bus: Bus, clock, guarded: dict[str, Guarded], watched: set[str]):
        self.bus = bus
        self.clock = clock
        self.guarded = guarded
        self.watched = set(watched) | set(guarded)  # the household's appliances (not sensors, lamps, ...)
        self.starts: dict[str, list[datetime]] = {d: [] for d in self.watched}  # switch-ons seen live
        self.states: dict[str, str] = {}
        self.restored: list[tuple[datetime, str, float]] = []  # (time, device, probability), for logs and tests
        self._hour: datetime | None = None
        self._checked: set[tuple[str, datetime]] = set()
        bus.subscribe(f"{ROOT}/+/+/state", self._on_state)

    def _on_state(self, topic: str, payload: dict) -> None:
        device = device_id_from_topic(topic)
        if device not in self.watched:
            return
        value, before = payload["value"], self.states.get(device)
        self.states[device] = value
        if value == "on" and before != "on":
            self.starts[device].append(datetime.fromisoformat(payload["ts"]))
        if value == "cut" and device in self.guarded:
            self._check(device)

    def probability_now(self, device: str) -> float:
        g = self.guarded[device]
        hour = pd.Timestamp(self.clock.now()).floor("h")
        recent = pd.date_range(g.labels.index.max() + pd.Timedelta(hours=1), hour, freq="h")
        labels = pd.concat([g.labels, pd.Series(0, index=recent, dtype="int8")])
        labels[labels.index.isin(pd.DatetimeIndex(self.starts[device]).floor("h"))] = 1
        live = [t for d, ts in self.starts.items() if d != device for t in ts]
        others = pd.concat([g.others, pd.DatetimeIndex(live).floor("h").value_counts()])
        return float(g.forest.predict(labels, others.groupby(level=0).sum()).iloc[-1])

    def _check(self, device: str) -> None:
        hour = self.clock.now().replace(minute=0, second=0, microsecond=0)
        if (device, hour) in self._checked:
            return
        self._checked.add((device, hour))
        p = self.probability_now(device)
        if p >= self.guarded[device].forest.threshold:
            now = self.clock.now()
            self.restored.append((now, device, p))
            self.bus.publish(set_topic(device), command("off", now, SOURCE))  # "off" from an automation = restore
            self.bus.publish(ALERTS_TOPIC, {"kind": "forest_restored", "device": device,
                                            "probability": round(p, 3), "ts": now.isoformat()})

    def tick(self) -> None:
        """Called every simulation step; does its work once per hour."""
        hour = self.clock.now().replace(minute=0, second=0, microsecond=0)
        if hour == self._hour:
            return
        self._hour = hour
        for device in self.guarded:
            if self.states.get(device) == "cut":
                self._check(device)
