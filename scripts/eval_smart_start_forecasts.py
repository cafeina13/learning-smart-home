"""How much of smart start's gain survives without a perfect solar forecast, and how much a
better load forecast could add.

    .venv\\Scripts\\python.exe scripts\\eval_smart_start_forecasts.py

scripts/eval_suggestions.py ranks the start times with the real solar output of the day (a
perfect forecast) and the rest of the house's typical load (median of the same hour of the
week over the 4 weeks before). That makes its 53.6% (REFIT) an upper bound. Here the same runs,
deadlines and scoring are replayed with other forecasts. Every variant is scored the same way:
with the solar REALLY left over after the rest of the house's REAL load.

Solar forecasts, all known when the machine is armed (only days that ended before the arming
day are used):
  perfect       the real output of that day (as in the report)
  persistence   yesterday's output at the same hour (yesterday = the day before arming)
  climatology   the median output at that hour over the 14 days before arming
Load forecasts:
  typical       the 4-week median of the same hour of the week (as in the report)
  perfect       the real load (no real forecast can beat this: the ceiling for any load model)

Writes data/results/smart_start_forecasts.csv (one row per run).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from eval_suggestions import (MIN_USES, PV_KWP, RESULTS, appliance_type, control_of, deadline_for,  # noqa: E402
                              machine_hourly_w, solar_kwh, solar_kwh_many)
from habits.prepare import load_aggregate, load_prepared  # noqa: E402
from habits.solar import hourly_pv_w  # noqa: E402

CLIMATOLOGY_DAYS = 14


def daily_profiles(pv: pd.Series) -> pd.DataFrame:
    """date x hour-of-day table of the PV output."""
    frame = pv.to_frame("w").assign(day=pv.index.normalize(), hour=pv.index.hour)
    # The clock change in spring leaves one night hour (01:00) without a value: no sun then.
    return frame.pivot_table(index="day", columns="hour", values="w").reindex(columns=range(24)).fillna(0.0)


def profile_forecast(profile: pd.Series, hours: pd.DatetimeIndex) -> pd.Series:
    """Turn a 24-value profile (by hour of day) into an hourly series over `hours`."""
    return pd.Series(profile.reindex(hours.hour).to_numpy(), index=hours).fillna(0.0)


def best_start(candidates, minutes, watts, solar: pd.Series, expected_left: pd.Series, start):
    """The candidate start with the most expected leftover solar, as in eval_suggestions: if the
    load is expected to use up all the sun, rank by the solar alone; with no sun at all, start now."""
    for series in (expected_left, solar):
        score = solar_kwh_many(candidates, minutes, watts, series)
        if max(score) > 0:
            return candidates[int(np.argmax(score))]
    return start


def replay(uses, appliances, aggregate, pv) -> pd.DataFrame:
    days = daily_profiles(pv)
    persistence = days.shift(1)  # row d: the profile of day d-1
    climatology = days.rolling(CLIMATOLOGY_DAYS, min_periods=CLIMATOLOGY_DAYS).median().shift(1)
    rows = []
    for a in appliances.itertuples():
        if control_of(a.appliance) != "armable" or a.uses < MIN_USES:
            continue
        runs = uses[(uses["house"] == a.house) & (uses["column"] == a.column)]
        runs = runs[(runs["start"] >= pv.index.min()) & (runs["end"] <= pv.index.max())]
        house_w = aggregate.loc[aggregate["house"] == a.house].set_index("hour")["aggregate_w"]
        other = house_w.sub(machine_hourly_w(runs), fill_value=0.0).clip(lower=0.0)
        other = other.reindex(pv.index).fillna(house_w.median())
        available = (pv - other).clip(lower=0.0)  # what was really left: used to score every variant
        typical = pd.concat([other.shift(168 * k) for k in range(1, 5)], axis=1).median(axis=1)
        typical = typical.fillna(house_w.median())
        for r in runs.itertuples():
            minutes = r.duration_s / 60
            watts = r.energy_wh / (r.duration_s / 3600)
            latest = deadline_for(r.start) - pd.Timedelta(minutes=minutes)
            candidates = pd.date_range(r.start.ceil("15min"), latest, freq="15min") if latest > r.start else []
            row = {"house": a.house, "appliance": a.appliance, "type": appliance_type(a.appliance),
                   "start": r.start, "kwh": r.energy_wh / 1000,
                   "solar_kwh_actual": solar_kwh(r.start, minutes, watts, available)}
            if not len(candidates):
                for name in ("perfect_typical", "perfect_perfect", "persistence_typical", "climatology_typical"):
                    row[name] = row["solar_kwh_actual"]
                rows.append(row)
                continue
            # Up to the deadline: a run that starts late keeps running until then.
            hours = pd.date_range(r.start.floor("h"), deadline_for(r.start).ceil("h"), freq="h")
            arm_day = r.start.normalize()
            solar = {"perfect": pv.reindex(hours).fillna(0.0)}
            for name, table in (("persistence", persistence), ("climatology", climatology)):
                profile = table.loc[arm_day] if arm_day in table.index else None
                solar[name] = (profile_forecast(profile, hours) if profile is not None and profile.notna().all()
                               else None)  # not enough history yet: the run is left out of this variant
            load = {"typical": typical.reindex(hours).fillna(house_w.median()),
                    "perfect": other.reindex(hours).fillna(house_w.median())}
            for s_name, l_name in (("perfect", "typical"), ("perfect", "perfect"),
                                   ("persistence", "typical"), ("climatology", "typical")):
                if solar[s_name] is None:
                    row[f"{s_name}_{l_name}"] = np.nan
                    continue
                expected_left = (solar[s_name] - load[l_name]).clip(lower=0.0)
                chosen = best_start(candidates, minutes, watts, solar[s_name], expected_left, r.start)
                row[f"{s_name}_{l_name}"] = solar_kwh(chosen, minutes, watts, available)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    uses, _, appliances = load_prepared()
    runs = replay(uses, appliances, load_aggregate(), hourly_pv_w(PV_KWP))
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS / "smart_start_forecasts.csv", index=False)
    variants = {"perfect_typical": "perfect solar, typical load (the report)",
                "perfect_perfect": "perfect solar, perfect load (ceiling)",
                "persistence_typical": "yesterday's solar, typical load",
                "climatology_typical": f"{CLIMATOLOGY_DAYS}-day median solar, typical load"}
    # Compare on the same runs: those that every variant could schedule.
    common = runs.dropna(subset=list(variants))
    print(f"Smart start, {len(common)} of {len(runs)} runs in {common['house'].nunique()} homes "
          f"(runs without {CLIMATOLOGY_DAYS} days of solar history left out), {PV_KWP:g} kWp\n")
    total = common["kwh"].sum()
    print(f"  {'as the runs happened':<42} solar share {common['solar_kwh_actual'].sum() / total:.1%}")
    for col, label in variants.items():
        print(f"  {label:<42} solar share {common[col].sum() / total:.1%}")
    homes = common.groupby("house")[["solar_kwh_actual", *variants]].sum()
    print()
    for col in ("persistence_typical", "climatology_typical"):
        better = homes[col] > homes["solar_kwh_actual"]
        p = wilcoxon(homes[col], homes["solar_kwh_actual"], alternative="greater").pvalue
        kept = (homes[col] - homes["solar_kwh_actual"]).sum() / (homes["perfect_typical"] - homes["solar_kwh_actual"]).sum()
        print(f"  {variants[col]}: better than as-happened in {better.sum()} of {len(homes)} homes "
              f"(one-sided Wilcoxon p = {p:.1e}); keeps {kept:.0%} of the perfect-forecast gain")
    gap = (homes["perfect_perfect"] - homes["perfect_typical"]).sum() / total
    print(f"  perfect load instead of typical load: +{gap:.1%} points of solar share at most")


if __name__ == "__main__":
    main()
