"""Turn learned habits into the right kind of help for each appliance, with its energy effect.

What the house may do depends on the appliance's control type (home/devices.py):

automation   for "auto" appliances that can safely run unattended (a coffee
             machine with a timer): "Coffee machine on, weekdays 06:30".
             Energy is about the same; the gain is comfort.
standby_cut  for "manual" appliances behind a smart plug (TV, computer): cut
             their standby power in the longest window in which they are
             practically never used, and restore it at the end. Never cuts
             while in use; switching the appliance on restores power (and
             counts as an override against the rule).
smart_start  for "armable" machines (dishwasher, washing machine): enable delay
             start with a learned default deadline. The person still loads and
             arms the machine; the house picks the start time with the most
             solar inside the deadline. Energy effect = kWh per week moved from
             the grid to the panel.

Manual appliances are never switched ON by the house, and armable machines
never start unless a person armed them. A kettle (manual, no standby draw) gets
no suggestion at all: there is nothing useful to automate.

Each suggestion is identified by kind, device and days (not its exact time
window), so newer evidence updates a pending suggestion instead of adding a
near-duplicate. Accepted and rejected ones are remembered.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from habits.forest import MIN_AGREEMENT
from habits.refit import category
from habits.routines import SINGLE_DAYS, WEEKDAYS, WEEKEND, find_quiet_window, find_routines
from home.rules import Action, Rule, TimeWindow
from home.scheduler import make_plan, solar_coverage
from home.simulation import clear_sky_solar_w

ROOMS = {
    "kitchen": ("kettle", "toaster", "microwave", "coffee", "bread", "mixer", "magimix", "k mix", "dishwasher"),
    "utility": ("washing", "washer", "dryer"),
    "living": ("television", "tv", "hi-fi", "games", "overhead fan"),
    "study": ("computer",),
}
# Habit appliances that can run unattended on a timer.
TIMER_CAPABLE = ("coffee", "electric heater", "overhead fan")
DEFAULT_DEADLINE = time(18, 0)  # smart start's suggested "finish by"


def device_id(appliance: str) -> str:
    """'Television Site' -> 'living/television_site' (room guessed from the name)."""
    lower = appliance.lower()
    room = next((r for r, words in ROOMS.items() if any(w in lower for w in words)), "home")
    name = "".join(ch if ch.isalnum() else "_" for ch in lower).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return f"{room}/{name}"


def control_of(appliance: str) -> str | None:
    """'auto', 'manual', 'armable', or None for appliances that run themselves (fridges)."""
    cat = category(appliance)
    if cat == "automatic":
        return None
    if cat == "shiftable":
        return "armable"
    return "auto" if any(w in appliance.lower() for w in TIMER_CAPABLE) else "manual"


def _days_name(weekdays: tuple[int, ...]) -> str:
    return TimeWindow(time(0), time(1), weekdays).describe().rsplit(" ", 1)[0]


@dataclass
class Suggestion:
    kind: str  # "automation", "standby_cut", "smart_start"
    device: str
    appliance: str
    control: str
    days: tuple[int, ...]
    title: str  # one line a person can accept or reject
    rules: list[Rule]  # rules to enable on acceptance (none for smart start)
    support: float  # how reliably the pattern held in the training days
    energy_kwh_week: float  # saved, or moved to solar; 0 for comfort automations
    energy_note: str
    setting: dict | None = None  # smart start: {"default_deadline": "18:00", "next_day": true}
    learned_from: str = ""  # which data: "REFIT house 3", "this home's event log", ...
    status: str = "pending"  # "pending", "accepted", "rejected"
    model_agreement: float | None = None  # share of unseen recent days the forest agrees on (habits/forest.py)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.device}:{','.join(map(str, self.days))}"


def _automations(appliance, dev, uses, days, windows) -> list[Suggestion]:
    out = []
    for r in find_routines(uses, days, appliance, kinds=(WEEKDAYS, WEEKEND)):
        windows.append((r.start_slot, r.n_slots))
        window = TimeWindow(r.start, r.end, r.weekdays)
        out.append(Suggestion(
            "automation", dev, appliance, "auto", r.weekdays,
            f"{appliance} on at {r.start:%H:%M}, {_days_name(r.weekdays)} (timer)",
            [Rule(f"auto {dev} {window.describe()}", [], [Action(dev, "on")], when=window, origin="learned")],
            r.support, 0.0,
            f"You switched it on in this window on {r.support:.0%} of these days; energy use is unchanged",
        ))
    return out


def _standby_cuts(appliance, dev, uses, days, standby_w, windows) -> list[Suggestion]:
    out = []
    for kind in (WEEKDAYS, WEEKEND):
        q = find_quiet_window(uses, days, appliance, kind)
        if q is None:
            continue
        windows.append((q.start_slot, q.n_slots))
        cut = TimeWindow(q.start, q.end, kind)
        crosses_midnight = q.start_slot + q.n_slots >= 48
        restore_days = tuple(sorted((d + 1) % 7 for d in kind)) if crosses_midnight else kind
        restore_end = (datetime.combine(date.today(), q.end) + timedelta(minutes=30)).time()
        restore = TimeWindow(q.end, restore_end, restore_days)
        saved = standby_w * q.hours * len(kind) / 1000
        out.append(Suggestion(
            "standby_cut", dev, appliance, "manual", kind,
            f"{appliance}: cut standby power {cut.describe()}",
            [Rule(f"cut {dev} {cut.describe()}", [], [Action(dev, "cut")], when=cut, origin="learned"),
             Rule(f"restore {dev} {restore.describe()}", [], [Action(dev, "off")], when=restore, origin="learned")],
            1 - q.active_rate, saved,
            f"Unused in this window on {1 - q.active_rate:.0%} of these days; saves about {saved:.2f} kWh/week "
            f"at {standby_w:.1f} W standby. Never cuts while in use; switching it on restores power",
        ))
    return out


MIN_RUNS_PER_WEEK = 0.5
MIN_KWH_PER_WEEK = 0.1


def _deadline_after(start: datetime) -> datetime:
    """The default deadline for a machine loaded at `start`: 18:00, tomorrow if loaded after 14:00."""
    day = start.date() + timedelta(days=1 if start.time() >= time(14, 0) else 0)
    return datetime.combine(day, DEFAULT_DEADLINE)


def _smart_start(appliance, dev, uses, days, forecast) -> list[Suggestion]:
    """Smart start needs no routine: the person arms the machine whenever they load it.

    So the suggestion replays every run in the learning data: for each, the solar
    energy it got when it really ran vs. at the best start the house would pick
    before the default deadline, using the expected solar output. It is offered
    when the machine runs often enough and the gain is worth it.
    """
    if len(uses) == 0 or not days:
        return []
    weeks = max(1.0, len(days) / 7)
    runs_per_week = len(uses) / weeks
    gains = []
    for r in uses.itertuples():
        start = r.start.to_pydatetime(warn=False)  # sub-second precision is irrelevant here
        minutes = r.duration_s / 60
        watts = r.energy_wh / (r.duration_s / 3600) if r.duration_s else 0.0
        now_kwh = solar_coverage(start, minutes, watts, forecast)
        deadline = _deadline_after(start)
        best_kwh = (make_plan(dev, start, deadline, watts, minutes, forecast).options[0].solar_kwh
                    if deadline - timedelta(minutes=minutes) > start else now_kwh)
        gains.append(max(0.0, best_kwh - now_kwh))
    moved = sum(gains) / weeks
    if runs_per_week < MIN_RUNS_PER_WEEK or moved < MIN_KWH_PER_WEEK:
        return []

    starts = uses["start"]
    next_day = (starts.dt.hour >= 14).mean() >= 0.5  # most runs are loaded in the afternoon or evening
    typical = f"{int(starts.dt.hour.median()):02d}:00"
    helped = sum(g > 0 for g in gains) / len(gains)
    when = "tomorrow " if next_day else ""
    return [Suggestion(
        "smart_start", dev, appliance, "armable", (0, 1, 2, 3, 4, 5, 6),
        f"{appliance}: smart start, finish by {when}{DEFAULT_DEADLINE:%H:%M}",
        [], helped, moved,
        f"About {runs_per_week:.1f} runs/week, typically loaded around {typical} "
        f"({uses['energy_wh'].mean() / 1000:.2f} kWh each). You still load and arm it; the house picks the "
        f"start with the most solar to spare. {helped:.0%} of past runs would have used more solar: about {moved:.2f} kWh/week "
        f"from the grid to the panel",
        setting={"default_deadline": DEFAULT_DEADLINE.strftime("%H:%M"), "next_day": bool(next_day)},
    )]


def _check_with_forest(out: list[Suggestion], windows: list[tuple[int, int]], forest) -> list[Suggestion]:
    """Job 1 of habits/forest.py: the forest checks each window on days it never saw; weak ones are held back."""
    kept = []
    for s, (start_slot, n_slots) in zip(out, windows):
        a = forest.agreement(s.days, start_slot, n_slots, expect_use=s.kind == "automation")
        if a == a:  # not NaN: the check days had data in this window
            s.model_agreement = a
            s.energy_note += f". The habit model agrees on {a:.0%} of recent days it did not learn from"
            if hasattr(forest, "window_reasons"):
                try:
                    why = forest.window_reasons(s.days, start_slot, n_slots, expect_use=s.kind == "automation")
                except Exception:  # an explanation must never hold back a suggestion
                    why = []
                if why:
                    s.energy_note += f", mostly from {' and '.join(why)}"
            if a < MIN_AGREEMENT:
                continue
        kept.append(s)
    return kept


def suggest_for_appliance(appliance: str, uses: pd.DataFrame, days: list[date], standby_w: float = 0.0,
                          learned_from: str = "",
                          forecast: Callable[[datetime], float] = lambda t: clear_sky_solar_w(t, 400.0),
                          forest=None) -> list[Suggestion]:
    """Suggestions for one appliance. With a HabitForest (habits/forest.py), automations and
    standby cuts are first checked by it; smart start is not (the person arms the machine)."""
    control = control_of(appliance)
    dev = device_id(appliance)
    windows: list[tuple[int, int]] = []
    if control == "auto":
        out = _automations(appliance, dev, uses, days, windows)
    elif control == "manual":
        out = _standby_cuts(appliance, dev, uses, days, standby_w, windows) if standby_w > 0 else []
    elif control == "armable":
        out = _smart_start(appliance, dev, uses, days, forecast)
    else:
        out = []  # fridges and freezers run themselves
    if forest is not None and windows:
        out = _check_with_forest(out, windows, forest)
    for s in out:
        s.learned_from = learned_from
    return out


SCHEMA = """
CREATE TABLE IF NOT EXISTS suggestions (
    key        TEXT PRIMARY KEY,   -- kind:device:days
    kind       TEXT NOT NULL,
    device     TEXT NOT NULL,
    appliance  TEXT NOT NULL,
    control    TEXT NOT NULL,
    title      TEXT NOT NULL,
    rules      TEXT NOT NULL,      -- JSON list of Rule.to_dict()
    setting    TEXT,               -- JSON, smart start only
    support    REAL NOT NULL,
    energy_kwh_week REAL NOT NULL,
    energy_note TEXT NOT NULL,
    learned_from TEXT NOT NULL,
    status     TEXT NOT NULL,      -- pending, accepted, rejected
    created    TEXT NOT NULL,
    decided    TEXT
);
"""


@dataclass
class Accepted:
    device: str
    kind: str
    rules: list[Rule] = field(default_factory=list)
    setting: dict | None = None


class SuggestionStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")  # see home/eventlog.py
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def offer(self, s: Suggestion) -> bool:
        """Store a suggestion. Returns True only if it is new.

        If the same suggestion (kind, device, days) is still pending, the newer one
        replaces it: its window, numbers and evidence (e.g. first seen in offline
        data, now confirmed by this home's own log). Decided ones are left alone.
        """
        rules = json.dumps([r.to_dict() for r in s.rules])
        setting = json.dumps(s.setting) if s.setting else None
        self.db.execute(
            "UPDATE suggestions SET title = ?, rules = ?, setting = ?, support = ?, energy_kwh_week = ?, "
            "energy_note = ?, learned_from = ? WHERE key = ? AND status = 'pending'",
            (s.title, rules, setting, s.support, s.energy_kwh_week, s.energy_note, s.learned_from, s.key),
        )
        cur = self.db.execute(
            "INSERT OR IGNORE INTO suggestions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL)",
            (s.key, s.kind, s.device, s.appliance, s.control, s.title, rules, setting, s.support,
             s.energy_kwh_week, s.energy_note, s.learned_from, datetime.now().isoformat(timespec="seconds")),
        )
        self.db.commit()
        return cur.rowcount == 1

    def list(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM suggestions", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        rows = [dict(r) for r in self.db.execute(sql + " ORDER BY energy_kwh_week DESC, support DESC", args)]
        for r in rows:
            r["rules"] = json.loads(r["rules"])
            r["setting"] = json.loads(r["setting"]) if r["setting"] else None
        return rows

    def decide(self, key: str, accept: bool) -> Accepted | None:
        """Mark a pending suggestion. When accepted, returns what to enable (rules and/or a setting)."""
        row = self.db.execute("SELECT * FROM suggestions WHERE key = ?", (key,)).fetchone()
        if row is None or row["status"] != "pending":
            return None
        status = "accepted" if accept else "rejected"
        self.db.execute("UPDATE suggestions SET status = ?, decided = ? WHERE key = ?",
                        (status, datetime.now().isoformat(timespec="seconds"), key))
        self.db.commit()
        if not accept:
            return None
        return Accepted(row["device"], row["kind"], [Rule.from_dict(r) for r in json.loads(row["rules"])],
                        json.loads(row["setting"]) if row["setting"] else None)
