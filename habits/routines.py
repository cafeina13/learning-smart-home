"""Find routines in an appliance's uses: when it is reliably switched on, and when it is never used.

The day is cut into 48 half-hour slots. Weekdays and weekends are treated as
separate kinds of day, because most households live them differently.

- A routine is a 30- or 60-minute window in which the appliance is switched on
  on most days of that kind ("kettle, weekdays 07:00-07:30, on 85% of days").
  Support = share of days of that kind with at least one switch-on in the window.
- A quiet window is the longest stretch (at least `min_hours`) in which the
  appliance is almost never running. Cutting standby power there costs the
  household nothing.

The same `holds_rate` that scores a window on the training days scores it on
later, unseen days: that is how suggestions are evaluated.
"""

from dataclasses import dataclass
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

WEEKDAYS = (0, 1, 2, 3, 4)
WEEKEND = (5, 6)
SLOTS = 48  # half hours per day


def slot_of(ts: pd.Timestamp) -> int:
    return ts.hour * 2 + ts.minute // 30


def slot_time(slot: int) -> time:
    slot %= SLOTS
    return time(slot // 2, 30 * (slot % 2))


def days_of_kind(days: list[date], kind: tuple[int, ...]) -> list[date]:
    return [d for d in days if d.weekday() in kind]


@dataclass
class Routine:
    appliance: str
    weekdays: tuple[int, ...]
    start_slot: int
    n_slots: int
    support: float  # share of training days of this kind with a switch-on in the window
    n_days: int  # how many training days of this kind there were
    energy_wh_per_use: float
    duration_s_per_use: float

    @property
    def start(self) -> time:
        return slot_time(self.start_slot)

    @property
    def end(self) -> time:
        return slot_time(self.start_slot + self.n_slots)


@dataclass
class QuietWindow:
    appliance: str
    weekdays: tuple[int, ...]
    start_slot: int
    n_slots: int  # may run past midnight
    active_rate: float  # share of those slots in which the appliance ran

    @property
    def start(self) -> time:
        return slot_time(self.start_slot)

    @property
    def end(self) -> time:
        return slot_time(self.start_slot + self.n_slots)

    @property
    def hours(self) -> float:
        return self.n_slots / 2


def _start_matrix(uses: pd.DataFrame, days: list[date]) -> np.ndarray:
    """days x 48 matrix: 1 where at least one use started in that slot on that day."""
    index = {d: i for i, d in enumerate(days)}
    m = np.zeros((len(days), SLOTS), dtype=bool)
    for start in uses["start"]:
        i = index.get(start.date())
        if i is not None:
            m[i, slot_of(start)] = True
    return m


def _active_matrix(uses: pd.DataFrame, days: list[date]) -> np.ndarray:
    """days x 48 matrix: 1 where the appliance was running at any point in that slot."""
    index = {d: i for i, d in enumerate(days)}
    m = np.zeros((len(days), SLOTS), dtype=bool)
    for start, end in zip(uses["start"], uses["end"]):
        for t in pd.date_range(start.floor("30min"), end, freq="30min"):
            i = index.get(t.date())
            if i is not None:
                m[i, slot_of(t)] = True
    return m


SINGLE_DAYS = tuple((d,) for d in range(7))


def find_routines(uses: pd.DataFrame, days: list[date], appliance: str, min_support: float = 0.5,
                  min_days: int = 8, kinds: tuple[tuple[int, ...], ...] = (WEEKDAYS, WEEKEND)) -> list[Routine]:
    # kinds: which groups of days to look at. Daily habits use weekdays/weekend;
    # loads that run a few times a week (washing) use SINGLE_DAYS, so "Saturday"
    # is not diluted by the six other days.
    routines = []
    for kind in kinds:
        kind_days = days_of_kind(days, kind)
        if len(kind_days) < min_days:
            continue
        starts = _start_matrix(uses, kind_days)
        # Every 30- and 60-minute window, scored by support; a small penalty makes a
        # narrow window win when it explains the routine almost as well as a wide one.
        candidates = []
        for n in (1, 2):
            for s in range(SLOTS - n + 1):
                support = starts[:, s:s + n].any(axis=1).mean()
                candidates.append((support - 0.05 * (n - 1), support, s, n))
        taken = np.zeros(SLOTS, dtype=bool)
        for _, support, s, n in sorted(candidates, reverse=True):
            if support < min_support:
                break
            if taken[s:s + n].any():
                continue
            taken[s:s + n] = True
            in_window = uses[[d.weekday() in kind and s <= slot_of(d) < s + n for d in uses["start"]]]
            mean = lambda column: (float(in_window[column].mean())  # noqa: E731
                                   if len(in_window) and column in in_window else 0.0)  # CASAS: no energy
            routines.append(Routine(appliance, kind, s, n, float(support), len(kind_days),
                                    mean("energy_wh"), mean("duration_s")))
    return routines


def find_quiet_window(uses: pd.DataFrame, days: list[date], appliance: str, kind: tuple[int, ...],
                      max_active_rate: float = 0.02, min_hours: float = 4) -> QuietWindow | None:
    """The longest stretch, starting on a day of this kind, where the appliance almost never runs.

    A stretch may run past midnight into the next day, so each day of this kind is
    looked at together with the day that follows it (96 slots). The part after
    midnight is judged on those real next days (a Sunday night is followed by
    Mondays), not on the same kind of day. A stretch is at most 24 hours.
    """
    index = {d: i for i, d in enumerate(days)}
    pairs = [(index[d], index[d + timedelta(days=1)]) for d in days_of_kind(days, kind)
             if d + timedelta(days=1) in index]
    if not pairs:
        return None
    active = _active_matrix(uses, days)
    two_days = np.array([np.concatenate([active[a], active[b]]) for a, b in pairs])
    rate = two_days.mean(axis=0)
    quiet = rate <= max_active_rate

    best_start, best_len = 0, 0
    for s in range(SLOTS):  # the stretch must start on the first day
        n = 0
        while n < SLOTS and quiet[s + n]:
            n += 1
        if n > best_len:
            best_start, best_len = s, n
    if best_len / 2 < min_hours:
        return None
    return QuietWindow(appliance, kind, best_start, best_len,
                       float(rate[best_start:best_start + best_len].mean()))


def holds_rate(weekdays: tuple[int, ...], start_slot: int, n_slots: int, uses: pd.DataFrame, days: list[date]) -> float:
    """Share of the given days (of this kind) with a switch-on in the window."""
    kind_days = days_of_kind(days, weekdays)
    if not kind_days:
        return float("nan")
    return float(_start_matrix(uses, kind_days)[:, start_slot:start_slot + n_slots].any(axis=1).mean())
