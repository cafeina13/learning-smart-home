"""A made-up household with known routines. FOR DEMOS AND TESTS ONLY.

It lets the suggestion pipeline and the dashboard be built and checked before
the real dataset is available, and it gives tests a ground truth: we know the
kettle is used around 07:10 on weekdays because we wrote it that way.

No accuracy reported in the project comes from this data; reported results
use REFIT only.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass
class Habit:
    appliance: str
    weekdays: tuple[int, ...]
    hour: float  # usual switch-on time, in hours (7.15 = 07:09)
    spread_min: float  # standard deviation of the switch-on time, minutes
    probability: float  # chance it happens on a matching day
    minutes: float  # how long it runs
    watts: float


HABITS = [
    Habit("Coffee Machine", (0, 1, 2, 3, 4), 6.9, 6, 0.9, 5, 1000),
    Habit("Kettle", (0, 1, 2, 3, 4), 7.15, 8, 0.9, 3, 2000),
    Habit("Kettle", (5, 6), 9.25, 25, 0.8, 3, 2000),
    Habit("Kettle", (0, 1, 2, 3, 4, 5, 6), 16.5, 60, 0.5, 3, 2000),
    Habit("Television Site", (0, 1, 2, 3, 4), 19.6, 15, 0.85, 200, 90),
    Habit("Television Site", (5, 6), 14.0, 60, 0.6, 240, 90),
    Habit("Washing Machine", (5,), 18.9, 20, 0.85, 110, 450),
    Habit("Washing Machine", (2,), 19.0, 20, 0.6, 110, 450),
    Habit("Dishwasher", (0, 1, 2, 3, 4, 5, 6), 21.25, 15, 0.7, 120, 600),
]

STANDBY_W = {"Coffee Machine": 1.5, "Kettle": 0.0, "Television Site": 9.0,
             "Washing Machine": 1.0, "Dishwasher": 0.8}


def household(weeks: int = 12, start: date = date(2014, 1, 6), seed: int = 0):
    """Returns ({appliance: uses DataFrame}, list of covered days, standby watts per appliance)."""
    rng = np.random.default_rng(seed)
    days = [start + timedelta(days=i) for i in range(weeks * 7)]
    rows: dict[str, list] = {}
    for day in days:
        for h in HABITS:
            if day.weekday() not in h.weekdays or rng.random() > h.probability:
                continue
            minutes = h.hour * 60 + rng.normal(0, h.spread_min)
            # Local wall-clock time without a time zone, like the rest of the project.
            begin = pd.Timestamp(day) + pd.Timedelta(minutes=float(np.clip(minutes, 0, 24 * 60 - h.minutes - 1)))
            end = begin + pd.Timedelta(minutes=h.minutes)
            rows.setdefault(h.appliance, []).append((begin, end, h.minutes * 60, h.watts * h.minutes / 60))
    uses = {a: pd.DataFrame(r, columns=["start", "end", "duration_s", "energy_wh"]).sort_values("start", ignore_index=True)
            for a, r in rows.items()}
    return uses, days, dict(STANDBY_W)
