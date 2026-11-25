"""The dashboard's real-data demo: one REFIT household, learned from and then replayed.

- Suggestions are learned from the household's first 80% of days, with the same
  code the evaluation uses (habits/suggestions.py), its measured standby draws,
  and an expected-solar profile from PVGIS for those months.
- The remaining 20% of days (never learned from) are replayed in the simulated
  home: every real kettle boil, TV session and machine run at its real time
  (habits/replay.py), with the solar panel producing that day's real output.

So when a suggestion is accepted in the dashboard, it acts on real behaviour the
system has not seen. Each appliance a suggestion can be checked for also gets
its habit forest (habits/forest.py), trained on the learning days only: it
checks the suggestions before they reach the dashboard, and the dashboard runs
its guard while standby cuts are on. (One shortcut, for the demo only: the standby draw is the
appliance's average over its whole recording; the evaluation scripts use the
learning days only.)
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Callable

import pandas as pd

from habits.dataset import covered_days, hourly_labels
from habits.features import other_appliance_starts
from habits.forest import Guarded, HabitForest
from habits.prepare import load_aggregate, load_prepared
from habits.solar import hourly_pv_w
from habits.suggestions import Suggestion, control_of, device_id, suggest_for_appliance

DEMO_HOUSE = 6  # dishwasher and washing machine, a TV and two computers with real standby, a busy kettle
PV_KWP = 3.0


@dataclass
class DemoAppliance:
    name: str
    device: str
    control: str  # "auto", "manual", "armable"
    watts: float  # typical power while in use
    standby_w: float
    run_minutes: float | None  # cycle length for machines; None when a person switches it off


@dataclass
class RefitDemo:
    house: int
    appliances: list[DemoAppliance]
    replay_uses: dict[str, pd.DataFrame]  # device -> uses in the replayed (unseen) days
    learn_uses: dict[str, pd.DataFrame]  # device -> uses in the learning days
    learn_days: list
    start: datetime  # first replayed morning
    end: datetime
    suggestions: list[Suggestion]
    solar_w: Callable[[datetime], float]  # real panel output at a given time
    learned_from: str = ""
    notes: list[str] = field(default_factory=list)
    guarded: dict[str, Guarded] = field(default_factory=dict)  # device -> its forest, for the guard
    expected_load_w: Callable[[datetime], float] = lambda t: 0.0  # the house's usual load at this hour of the week


def build_refit_demo(house: int = DEMO_HOUSE, kwp: float = PV_KWP) -> RefitDemo:
    uses, covered, appliances = load_prepared()
    days = covered_days(covered.loc[covered["house"] == house, "hour"])
    split = days[int(len(days) * 0.8)]
    learn_days = [d for d in days if d < split]
    split_ts = pd.Timestamp(split)

    pv = hourly_pv_w(kwp)
    pv_by_hour = {ts.to_pydatetime(): float(v) for ts, v in pv.items()}

    def solar_w(t: datetime) -> float:
        return pv_by_hour.get(t.replace(minute=0, second=0, microsecond=0), 0.0)

    # What the scheduler and the suggestions may assume about the sun: the average
    # output at each hour of the day over the learning months (not the future days).
    learn_pv = pv[(pv.index >= pd.Timestamp(learn_days[0])) & (pv.index < split_ts)]
    expected = learn_pv.groupby(learn_pv.index.hour).mean().to_dict()

    def expected_solar_w(t: datetime) -> float:
        return float(expected.get(t.hour, 0.0))

    # The house's usual load at each hour of the week over the learning days: smart start ranks start
    # times by the solar expected to be left after it (scripts/eval_suggestions.py: load-aware).
    agg = load_aggregate()
    load = agg.loc[(agg["house"] == house) & (agg["hour"] < split_ts)].set_index("hour")["aggregate_w"]
    typical = load.groupby([load.index.dayofweek, load.index.hour]).median().to_dict()

    def expected_load_w(t: datetime) -> float:
        return float(typical.get((t.weekday(), t.hour), 0.0))

    learned_from = f"REFIT house {house}, {len(learn_days)} real days ({learn_days[0]:%b %Y} - {learn_days[-1]:%b %Y})"
    demo_appliances, replay, learn, suggestions, guarded = [], {}, {}, [], {}
    house_uses = uses[uses["house"] == house].merge(appliances[["house", "column", "category"]],
                                                    on=["house", "column"], how="left")
    house_uses = house_uses[house_uses["start"] < split_ts]
    hours = pd.DatetimeIndex(covered.loc[covered["house"] == house, "hour"]).sort_values()
    learn_hours = hours[hours < split_ts]
    for row in appliances[appliances["house"] == house].itertuples():
        control = control_of(row.appliance)
        if control is None:
            continue  # fridges and freezers run themselves
        dev = device_id(row.appliance)
        u = uses[(uses["house"] == house) & (uses["column"] == row.column)]
        power = (u["energy_wh"] / (u["duration_s"] / 3600)).median() if len(u) else 0.0
        run = float(u["duration_s"].median() / 60) if control == "armable" and len(u) else None
        demo_appliances.append(DemoAppliance(row.appliance, dev, control, round(float(power), 1),
                                             float(row.standby_w), run))
        learn[dev] = u[u["start"] < split_ts]
        replay[dev] = u[u["start"] >= split_ts]
        forest = None
        if control in ("auto", "manual"):  # the kinds of suggestion the forest can check
            others = other_appliance_starts(house_uses, row.column)
            forest = HabitForest.fit(learn[dev], learn_hours, others)
            if forest is not None and control == "manual":
                guarded[dev] = Guarded(forest, hourly_labels(learn[dev], learn_hours), others)
        suggestions += suggest_for_appliance(row.appliance, learn[dev], learn_days, float(row.standby_w),
                                             learned_from=learned_from, forecast=expected_solar_w, forest=forest)

    return RefitDemo(house, demo_appliances, replay, learn, learn_days,
                     start=datetime.combine(split, time(6, 30)), end=datetime.combine(days[-1], time(23, 59)),
                     suggestions=suggestions, solar_w=solar_w, learned_from=learned_from, guarded=guarded,
                     expected_load_w=expected_load_w)
