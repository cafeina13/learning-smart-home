"""Shared evaluation on REFIT: build the per-appliance datasets and score models the same way.

Every model is judged on the same question, on the same split:

    "Will this appliance be switched on in this hour?"

for each house x appliance a person switches on (habit and shiftable
categories; fridges and the like are excluded). Hours are split by time:
the first 60% train, the next 20% tune (e.g. the decision threshold), the
last 20% test. Nothing from the test weeks is seen before scoring.

Appliances with fewer than MIN_USES uses are skipped: there is no habit to
learn from a handful of uses, and F1 on a few positives is noise.
"""

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from habits.dataset import hourly_labels, time_split
from habits.features import other_appliance_starts

MIN_USES = 50


@dataclass
class Task:
    house: int
    column: str  # "Appliance1".."Appliance9": unique within a house (names are not: house 13 has two microwaves)
    appliance: str
    category: str
    labels: pd.Series  # 0/1 per covered hour, time-ordered
    uses: pd.DataFrame
    train: pd.Series
    val: pd.Series
    test: pd.Series
    other_starts: pd.Series  # per-hour switch-ons of the house's other (non-automatic) appliances


def tasks(uses: pd.DataFrame, covered: pd.DataFrame, appliances: pd.DataFrame,
          categories: tuple[str, ...] = ("habit", "shiftable")) -> Iterator[Task]:
    """One task per house x appliance worth evaluating."""
    uses = uses.merge(appliances[["house", "column", "category"]], on=["house", "column"], how="left")
    for row in appliances.itertuples():
        if row.category not in categories or row.uses < MIN_USES:
            continue
        hours = pd.DatetimeIndex(covered.loc[covered["house"] == row.house, "hour"]).sort_values()
        house_uses = uses[uses["house"] == row.house]
        app_uses = house_uses[house_uses["column"] == row.column]
        labels = hourly_labels(app_uses, hours)
        train, val, test = time_split(labels)
        yield Task(row.house, row.column, row.appliance, row.category, labels, app_uses, train, val, test,
                   other_appliance_starts(house_uses, row.column))


def random_split(task: Task, seed: int = 0, train_frac: float = 0.6, val_frac: float = 0.2) -> Task:
    """The same task with its hours dealt at random into train / validation / test (stratified,
    same shares), the way Gad et al. (2025) split: test hours then sit between training hours.
    labels is reordered train, validation, test, as the sequence models expect."""
    hours = np.arange(len(task.labels))
    y = task.labels.to_numpy()
    strat = y if np.bincount(y).min() >= 3 else None
    train, rest = train_test_split(hours, train_size=train_frac, stratify=strat, random_state=seed)
    val, test = train_test_split(rest, train_size=val_frac / (1 - train_frac),
                                 stratify=None if strat is None else y[rest], random_state=seed)
    parts = [task.labels.iloc[np.sort(i)] for i in (train, val, test)]
    return replace(task, labels=pd.concat(parts), train=parts[0], val=parts[1], test=parts[2])


def appliance_type(name: str) -> str:
    """Group appliance names across houses: 'Washing Machine(1)' -> 'washing machine'."""
    lower = name.lower()
    # machines first: UK-DALE house 4 meters "washing machine + microwave + breadmaker" on one channel
    for key in ("dishwasher", "tumble dryer", "washer dryer", "washing machine", "kettle", "toaster",
                "microwave", "television", "tv", "computer", "hi-fi", "games"):
        if key in lower:
            return "television" if key == "tv" else key
    return lower.split("(")[0].strip()
