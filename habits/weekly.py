"""Weekly habit finding with a model: every week, from the last three weeks, which habits to suggest
next week and which existing ones to keep or drop.

Counting (habits/routines.py) finds habits once, from all the learning days, and the three-strike
rule drops a habit only after it has annoyed the resident three times. Here a model re-decides
every week. It predicts, for one appliance, one half-hour slot, one kind of day (weekdays or
weekends) and the coming week, the share of those days on which the appliance will be switched on
in that slot. Two thresholds turn that into decisions: a slot predicted at SUGGEST or more becomes
a habit; an existing habit is kept while its prediction stays at KEEP or more (the gap between the
two stops a habit from flickering on and off). One example per appliance, so a slot can hold any
number of habits (multi-label): kettle and toaster at 07:00 are two predictions.

Inputs, all from weeks before the predicted one:
  A  the slot        time of day (sine, cosine), weekday or weekend, month, appliance type
  B  own past        use in this slot in each of the last 3 weeks (a trend), the neighbouring
                     slots (+-30 and +-60 min), overall use per day in each of those weeks
  C  other appliances, four ways (an ablation tests which ones help):
     C1 same slot    each other appliance type's use in this slot
     C2 order        each other type's use in the slot before and the slot after
     C3 together     how often this appliance was used around this slot on days the other type
                     was used, and on days it was not (tells "always together" from "never")
     C4 room groups  kitchen, living, machines (activities: kitchen, living, personal): use in
                     this slot of the group's other appliances
  D  data and home   days with data in the last 3 weeks, the home's overall activity

"Use" is the share of that kind of day, in that week, with a switch-on starting in the slot.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from habits.dataset import covered_days
from habits.evaluate import MIN_USES, appliance_type

SLOTS = 48
WEEKS_BACK = 3
SUGGEST, KEEP = 0.5, 0.3
MIN_DAYS = {0: 2, 1: 1}  # days of the kind with data needed in a week: weekdays, weekends

GROUP_WORDS = {  # room groups (C4) by words in the type's name
    "kitchen": ("kettle", "toaster", "microwave", "bread", "mix", "oven", "hob", "coffee", "cook", "meal",
                "eat", "breakfast", "lunch", "dinner", "dish", "kitchen", "drink"),
    "living": ("television", "computer", "games", "hi-fi", "fan", "heater", "tv", "watch", "relax", "read",
               "work", "phone", "entertain", "music"),
    "machines": ("washing", "washer", "dryer", "dishwasher", "laundry"),
    "personal": ("sleep", "bed", "toilet", "bath", "shower", "groom", "hygiene", "dress", "medic", "wake",
                 "leave", "enter", "home"),
}


def group_of(type_name: str) -> str:
    lower = type_name.lower()
    if "dishwasher" in lower:
        return "machines"
    for group, words in GROUP_WORDS.items():
        if any(w in lower for w in words):
            return group
    return "other"


def column_name(type_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", type_name.lower()).strip("_")


@dataclass
class Home:
    house: int
    days: list[date]
    weeks: pd.DatetimeIndex  # every calendar week (Monday) from the first to the last day with data
    day_week: np.ndarray  # week index of each day
    day_kind: np.ndarray  # 0 weekday, 1 weekend
    starts: dict[str, np.ndarray]  # appliance column -> days x 48 bool (a switch-on started in the slot)
    types: dict[str, str]  # appliance column -> type
    tasks: list[str]  # the columns with enough uses to predict


def home_data(house: int, uses: pd.DataFrame, covered: pd.DataFrame, appliances: pd.DataFrame) -> Home | None:
    days = covered_days(covered.loc[covered["house"] == house, "hour"])
    if len(days) < 7 * (WEEKS_BACK + 2):
        return None
    index = {d: i for i, d in enumerate(days)}
    mondays = pd.DatetimeIndex([pd.Timestamp(d) - pd.Timedelta(days=d.weekday()) for d in days])
    weeks = pd.date_range(mondays.min(), mondays.max(), freq="7D")
    day_week = ((mondays - weeks[0]).days // 7).to_numpy()
    day_kind = np.array([1 if d.weekday() >= 5 else 0 for d in days])
    rows = appliances[(appliances["house"] == house) & appliances["category"].isin(["habit", "shiftable"])]
    starts, types, tasks = {}, {}, []
    for a in rows.itertuples():
        m = np.zeros((len(days), SLOTS), dtype=bool)
        for s in uses.loc[(uses["house"] == house) & (uses["column"] == a.column), "start"]:
            i = index.get(s.date())
            if i is not None:
                m[i, s.hour * 2 + s.minute // 30] = True
        starts[a.column], types[a.column] = m, appliance_type(a.appliance)
        if a.uses >= MIN_USES:
            tasks.append(a.column)
    return Home(house, days, weeks, day_week, day_kind, starts, types, tasks)


def weekly_sums(home: Home, m: np.ndarray) -> np.ndarray:
    """weeks x 2 kinds x 48 slots: how many days of that week and kind had a start in the slot."""
    out = np.zeros((len(home.weeks), 2, SLOTS))
    np.add.at(out, (home.day_week, home.day_kind), m.astype(float))
    return out


def past(x: np.ndarray, k: int) -> np.ndarray:
    """x shifted k weeks later along axis 0 (row w holds week w-k); zeros where there is none."""
    out = np.zeros_like(x)
    out[k:] = x[:-k] if k else x
    return out


def past3(x: np.ndarray) -> np.ndarray:
    return sum(past(x, k) for k in range(1, WEEKS_BACK + 1))


def roll(x: np.ndarray, shift: int) -> np.ndarray:
    """Slot s of the result holds slot s - shift (shift=1: the slot before), around midnight."""
    return np.roll(x, shift, axis=-1)


def safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def features(home: Home, all_types: list[str]) -> pd.DataFrame:
    """One row per task appliance x week x kind x slot, for weeks with enough data now and before."""
    n_days = np.zeros((len(home.weeks), 2))
    np.add.at(n_days, (home.day_week, home.day_kind), 1.0)
    n3 = past3(n_days)
    sums = {c: weekly_sums(home, m) for c, m in home.starts.items()}
    by_type: dict[str, list[str]] = {}
    for c, t in home.types.items():
        by_type.setdefault(t, []).append(c)
    frames = []
    for col in home.tasks:
        own_type = home.types[col]
        rate = safe_div(sums[col], n_days[:, :, None])  # this week's share of days (the label)
        own = [safe_div(past(sums[col], k), past(n_days, k)[:, :, None]) for k in (1, 2, 3)]
        own3 = safe_div(past3(sums[col]), n3[:, :, None])
        f = {"own_w1": own[0], "own_w2": own[1], "own_w3": own[2], "own_3w": own3}
        for shift, name in ((1, "own_prev_slot"), (-1, "own_next_slot"), (2, "own_prev_hour"), (-2, "own_next_hour")):
            f[name] = roll(own3, shift)
        for k in (1, 2, 3):
            per_day = safe_div(past(sums[col], k).sum(axis=2), past(n_days, k))
            f[f"own_per_day_w{k}"] = np.repeat(per_day[:, :, None], SLOTS, axis=2)
        f["days_3w"] = np.repeat(n3[:, :, None], SLOTS, axis=2)
        house_total = sum(past3(s).sum(axis=2) for s in sums.values())
        f["house_per_day_3w"] = np.repeat(safe_div(house_total, n3)[:, :, None], SLOTS, axis=2)
        # C: the other appliances, by type (several of one type are combined: any of them used)
        a_day = home.starts[col]
        groups: dict[str, list[np.ndarray]] = {}
        for t in all_types:
            others = [c for c in by_type.get(t, []) if c != col]
            name = column_name(t)
            if not others:
                for prefix in ("c1_", "c2b_", "c2a_", "c3w_", "c3n_"):
                    f[prefix + name] = np.full_like(own3, np.nan)
                continue
            m_t = np.logical_or.reduce([home.starts[c] for c in others])
            t3 = safe_div(past3(weekly_sums(home, m_t)), n3[:, :, None])
            f["c1_" + name], f["c2b_" + name], f["c2a_" + name] = t3, roll(t3, 1), roll(t3, -1)
            around = m_t | np.roll(m_t, 1, axis=1)  # the other type used in this slot or the one before
            both = past3(weekly_sums(home, a_day & around))
            with_t = past3(weekly_sums(home, around))
            own_sum = past3(sums[col])
            f["c3w_" + name] = safe_div(both, with_t)
            f["c3n_" + name] = safe_div(own_sum - both, n3[:, :, None] - with_t)
            groups.setdefault(group_of(t), []).append(t3)
        for g in ("kitchen", "living", "machines", "personal"):
            f["c4_" + g] = np.nanmax(np.stack(groups[g]), axis=0) if g in groups else np.full_like(own3, np.nan)
        # one row per week x kind x slot, kept where this week and the 3 before have enough data
        w_idx, k_idx, s_idx = np.meshgrid(np.arange(len(home.weeks)), [0, 1], np.arange(SLOTS), indexing="ij")
        keep = ((n_days >= np.array([MIN_DAYS[0], MIN_DAYS[1]])) & (n3 >= 2 * np.array([MIN_DAYS[0], MIN_DAYS[1]])))
        keep = np.repeat(keep[:, :, None], SLOTS, axis=2) & (w_idx >= WEEKS_BACK)
        frame = pd.DataFrame({name: v[keep] for name, v in f.items()})
        frame.insert(0, "label", rate[keep])
        frame.insert(0, "slot", s_idx[keep])
        frame.insert(0, "kind", k_idx[keep])
        frame.insert(0, "week", home.weeks[w_idx[keep]])
        frame.insert(0, "column", col)
        frame.insert(0, "house", home.house)
        frame["type"] = own_type
        frame["slot_sin"] = np.sin(2 * np.pi * frame["slot"] / SLOTS)
        frame["slot_cos"] = np.cos(2 * np.pi * frame["slot"] / SLOTS)
        frame["month"] = frame["week"].dt.month
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


INPUT_GROUPS = {  # for the ablation and for SHAP
    "A": lambda c: c in ("slot_sin", "slot_cos", "kind", "month", "type_code"),
    "B": lambda c: c.startswith("own_"),
    "C1": lambda c: c.startswith("c1_"),
    "C2": lambda c: c.startswith(("c2b_", "c2a_")),
    "C3": lambda c: c.startswith(("c3w_", "c3n_")),
    "C4": lambda c: c.startswith("c4_"),
    "D": lambda c: c in ("days_3w", "house_per_day_3w"),
}


def columns_for(frame: pd.DataFrame, groups: tuple[str, ...]) -> list[str]:
    return [c for c in frame.columns if any(INPUT_GROUPS[g](c) for g in groups)]


def group_of_column(column: str) -> str:
    return next(g for g, test in INPUT_GROUPS.items() if test(column))


TEST_SHARE = 0.2  # each home's test period: its last 20% of days


def cut_of(home: Home) -> pd.Timestamp:
    """The first day of the home's test period."""
    return pd.Timestamp(home.days[int(len(home.days) * (1 - TEST_SHARE))])


@dataclass
class Fold:
    houses: list[int]  # the homes tested in this fold
    cut: pd.Timestamp  # the earliest test-period start among them: nothing at or after it is learned from
    other_past: np.ndarray  # rows of the OTHER homes before the cut
    own_past: np.ndarray  # rows of the tested homes before the cut
    tested: np.ndarray  # every row of the tested homes (scored on their own test weeks only)


def time_folds(frame: pd.DataFrame, homes: dict, k: int = 5) -> list[Fold]:
    """Folds of homes with similar test-period starts. Every model learns only from weeks before the
    earliest test start in its fold, from all homes: a home's data from the weeks being predicted, or
    any later week of any home, is never learned from (REFIT's homes were recorded at the same time,
    so another home's same weeks would tell the model about the very season being predicted)."""
    order = sorted(homes, key=lambda h: cut_of(homes[h]))
    houses = frame["house"].to_numpy()
    weeks = frame["week"].to_numpy()
    folds = []
    for chunk in np.array_split(np.array(order), min(k, len(order))):
        chunk = [int(h) for h in chunk]
        cut = min(cut_of(homes[h]) for h in chunk)
        in_chunk = np.isin(houses, chunk)
        past = weeks < np.datetime64(cut)
        folds.append(Fold(chunk, cut, np.flatnonzero(~in_chunk & past), np.flatnonzero(in_chunk & past),
                          np.flatnonzero(in_chunk)))
    return folds


def decide(predicted: np.ndarray, previous: np.ndarray, suggest: float = SUGGEST,
           keep: float | None = None) -> np.ndarray:
    """Habit slots for the coming week: new at `suggest` or more, kept while at `keep` or more
    (by default 0.6 x suggest, the ratio of 0.3 to 0.5)."""
    keep = suggest * KEEP / SUGGEST if keep is None else keep
    return (predicted >= suggest) | (previous & (predicted >= keep))


def best_threshold(predicted: np.ndarray, label: np.ndarray) -> float:
    """The cut-off with the best F1 for "a habit next week" (used on at least half of the days),
    chosen on homes the model did not learn from."""
    target = label >= SUGGEST
    best, best_f1 = SUGGEST, -1.0
    for t in np.linspace(0.05, 0.6, 23):
        hit = predicted >= t
        tp = float((hit & target).sum())
        f1 = 2 * tp / (hit.sum() + target.sum()) if hit.sum() + target.sum() else 0.0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best
