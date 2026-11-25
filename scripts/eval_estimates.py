"""Are the suggestions' energy estimates right? Estimated before vs. measured after, on REFIT.

    .venv\\Scripts\\python.exe scripts\\eval_estimates.py

The proposal's objective: "estimated vs. measured energy of each suggestion in
the unseen weeks: within 15%". Every estimate is made from a home's first 80% of
days only, and compared with what really happened in its last 20%.

Standby cuts   estimated: the appliance's average idle draw on the learning days
               x the window's hours x the test nights.
               measured: the energy it really drew while idle inside the window
               on the test nights, from the raw 8-second readings.
Smart start    (a) the machine's energy per week: learning weeks vs. test weeks.
               (b) kWh per week moved to solar: computed on the learning runs vs.
                   on the test runs (from data/results/smart_start.csv, so run
                   scripts\\eval_suggestions.py smart first).

Writes data/results/estimates.csv.
"""

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.paths import RESULTS as PATHS_RESULTS  # noqa: E402

import eval_suggestions as ev  # noqa: E402
from habits.prepare import load_prepared, local_naive, read_house  # noqa: E402
from habits.routines import WEEKDAYS, WEEKEND  # noqa: E402
from habits.suggestions import control_of  # noqa: E402

RESULTS = PATHS_RESULTS  # data/results, or data/results/ukdale
WITHIN = 0.15


def idle_energy_kwh(t: np.ndarray, p: np.ndarray, threshold: float, start: datetime, end: datetime) -> float:
    """Energy drawn while idle (power under the threshold) between start and end, from raw samples."""
    lo, hi = np.searchsorted(t, np.datetime64(start)), np.searchsorted(t, np.datetime64(end))
    if hi <= lo:
        return 0.0
    seg_t, seg_p = t[lo:hi], p[lo:hi]
    dt = np.diff(seg_t, append=np.datetime64(end)).astype("timedelta64[s]").astype(float)
    dt = np.minimum(dt, 30.0)  # a gap in the data is not 'idle'
    idle = seg_p < threshold
    return float(np.sum(seg_p[idle] * dt[idle]) / 3.6e6)


def standby_estimates(uses, covered, appliances) -> list[dict]:
    rows = []
    raw_cache: dict[int, pd.DataFrame] = {}
    for a, u, train, val, test in ev.standby_candidates(uses, covered, appliances):
        if a.house not in raw_cache:
            df = read_house(a.house)
            df.index = local_naive(df.index)
            raw_cache.clear()  # keep one house in memory at a time
            raw_cache[a.house] = df.sort_index()
        df = raw_cache[a.house]
        t, p = df.index.to_numpy(), df[a.column].to_numpy()
        learn_days, test_set = train + val, set(test)
        learn_mask = t < np.datetime64(datetime.combine(test[0], time()))
        learn_p = p[learn_mask]
        learn_idle = learn_p[learn_p < a.on_threshold_w]
        standby_learn = float(learn_idle.mean()) if len(learn_idle) else 0.0
        learn_uses = u[u["start"] < pd.Timestamp(test[0])]
        for kind, kind_name in ((WEEKDAYS, "weekdays"), (WEEKEND, "weekends")):
            window = ev.learn_window(learn_uses, learn_days, a.appliance, kind, 0)
            if window is None:
                continue
            start_slot, n_slots = window
            nights = [d for d in test if d.weekday() in kind and d + timedelta(days=1) in test_set]
            if not nights:
                continue
            measured = sum(idle_energy_kwh(t, p, a.on_threshold_w,
                                           datetime.combine(d, time()) + timedelta(minutes=30 * start_slot),
                                           datetime.combine(d, time()) + timedelta(minutes=30 * (start_slot + n_slots)))
                           for d in nights)
            estimated = standby_learn * (n_slots / 2) * len(nights) / 1000
            rows.append({"suggestion": "standby_cut", "house": a.house, "appliance": a.appliance, "days": kind_name,
                         "estimated_kwh": estimated, "measured_kwh": measured})
    return rows


def smart_start_estimates(uses, covered, appliances) -> list[dict]:
    runs = pd.read_csv(RESULTS / "smart_start.csv", parse_dates=["start", "chosen"])
    rows = []
    for a in appliances.itertuples():
        if control_of(a.appliance) != "armable" or a.uses < ev.MIN_USES:
            continue
        days = ev.covered_days(covered.loc[covered["house"] == a.house, "hour"])
        split = pd.Timestamp(days[int(len(days) * 0.8)])
        learn_weeks = (split - pd.Timestamp(days[0])).days / 7
        test_weeks = (pd.Timestamp(days[-1]) - split).days / 7 + 1 / 7
        r = runs[(runs["house"] == a.house) & (runs["appliance"] == a.appliance)]
        learn, test = r[r["start"] < split], r[r["start"] >= split]
        if len(learn) < 10 or len(test) < 5:
            continue
        rows.append({"suggestion": "smart_start: energy/week", "house": a.house, "appliance": a.appliance,
                     "days": "all", "estimated_kwh": learn["kwh"].sum() / learn_weeks,
                     "measured_kwh": test["kwh"].sum() / test_weeks})
        moved = lambda g: (g["solar_kwh_smart"] - g["solar_kwh_actual"]).sum()  # noqa: E731
        rows.append({"suggestion": "smart_start: kWh/week to solar", "house": a.house, "appliance": a.appliance,
                     "days": "all", "estimated_kwh": moved(learn) / learn_weeks,
                     "measured_kwh": moved(test) / test_weeks})
    return rows


def main() -> None:
    uses, covered, appliances = load_prepared()
    df = pd.DataFrame(standby_estimates(uses, covered, appliances) + smart_start_estimates(uses, covered, appliances))
    df["error"] = (df["estimated_kwh"] - df["measured_kwh"]) / df["measured_kwh"].replace(0, np.nan)
    df.to_csv(RESULTS / "estimates.csv", index=False)
    print(f"Estimated (first 80% of days) vs. measured (last 20%); target: within {WITHIN:.0%}\n")
    summary = df.groupby("suggestion").agg(
        cases=("error", "size"),
        median_error=("error", "median"),
        median_abs_error=("error", lambda e: e.abs().median()),
        within_15=("error", lambda e: (e.abs() <= WITHIN).mean()),
        total_estimated=("estimated_kwh", "sum"), total_measured=("measured_kwh", "sum"))
    summary["total_error"] = summary["total_estimated"] / summary["total_measured"] - 1
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
