"""Cost-aware smart start: what a home pays for its machine runs under three UK tariffs.

    .venv\\Scripts\\python.exe scripts\\eval_smart_start_tariffs.py     (CAPSTONE_DATASET=ukdale for UK-DALE)

The same runs, deadlines and leftover solar as scripts/eval_suggestions.py and
scripts/eval_smart_start_forecasts.py, but the scheduler now picks the start time that costs
least instead of the one with the most solar. The cost of a run is

    grid energy x import price at that time  +  solar energy used x export price

(solar the machine uses is solar the home does not export: habits/tariffs.py). Prices are
Octopus Energy's 2019 tariffs (flat, Economy 7, Agile), replayed on the recording's calendar.
Agile publishes the next day's prices at 16:00, so a machine armed in the afternoon for the
next day is assumed to be re-planned once they are out; the prices are treated as known.

Three home set-ups:
  nopanel       no solar panel: only the tariff matters
  panel         a 3 kWp panel, perfect solar forecast (the report's upper bound)
  panel_naive   the same panel, solar forecast = the median output at that hour over 14 days
In both panel set-ups the rest of the house's load is forecast by the 4-week typical load.

Two settings for the resident: night starts allowed, or quiet hours (no start from 22:00 to
07:00). Every choice is scored with the real solar left over and the real prices.

Writes data/results/smart_start_tariffs.csv (one row per run).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from eval_smart_start_forecasts import CLIMATOLOGY_DAYS, daily_profiles, profile_forecast  # noqa: E402
from eval_suggestions import (MIN_USES, PV_KWP, RESULTS, appliance_type, control_of, deadline_for,  # noqa: E402
                              machine_hourly_w)
from habits.prepare import load_aggregate, load_prepared  # noqa: E402
from habits.solar import hourly_pv_w  # noqa: E402
from habits.tariffs import EXPORT, REGION, TARIFFS, Prices  # noqa: E402

QUIET = (22, 7)  # no start from 22:00 to 07:00 when the resident chooses quiet hours
SETUPS = ("nopanel", "panel", "panel_naive")
STEP_MIN = 5.0


def steps(minutes: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (minutes from the start) and lengths of the 5-minute steps of a run."""
    n_full, rest = divmod(minutes, STEP_MIN)
    lengths = np.r_[np.full(int(n_full), STEP_MIN), [rest] if rest > 1e-9 else []]
    return np.r_[0.0, np.cumsum(lengths)[:-1]], lengths


def step_times(starts: pd.DatetimeIndex, offsets: np.ndarray) -> pd.DatetimeIndex:
    """Every step of every candidate start, row by row (candidates x steps, flattened)."""
    times = starts.values[:, None] + (offsets * 60e9).astype("timedelta64[ns]")[None, :]
    return pd.DatetimeIndex(times.ravel())


def lookup(series: pd.Series | None, times: pd.DatetimeIndex, shape) -> np.ndarray:
    """Hourly values of `series` at each step (0 where there is none, or no series at all)."""
    if series is None:
        return np.zeros(shape)
    pos = series.index.get_indexer(times.floor("h"))
    return np.where(pos >= 0, series.to_numpy()[np.maximum(pos, 0)], 0.0).reshape(shape)


def costs(watts: float, lengths: np.ndarray, leftover: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Pence per candidate: grid energy at the import price plus used solar at the export price."""
    solar_w = np.minimum(watts, np.maximum(leftover, 0.0))
    kwh = lengths[None, :] / 60 / 1000
    return ((watts - solar_w) * kwh * price + solar_w * kwh * EXPORT).sum(axis=1)


def allowed(starts: pd.DatetimeIndex) -> np.ndarray:
    hours = np.asarray(starts.hour)
    return ~((hours >= QUIET[0]) | (hours < QUIET[1]))


def replay(uses, appliances, aggregate, pv) -> pd.DataFrame:
    prices = {t: Prices(t) for t in TARIFFS}
    days = daily_profiles(pv)
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
        available = (pv - other).clip(lower=0.0)  # the solar really left over: scores every choice
        typical = pd.concat([other.shift(168 * k) for k in range(1, 5)], axis=1).median(axis=1)
        typical = typical.fillna(house_w.median())
        for r in runs.itertuples():
            minutes = r.duration_s / 60
            watts = r.energy_wh / (r.duration_s / 3600)
            offsets, lengths = steps(minutes)
            latest = deadline_for(r.start) - pd.Timedelta(minutes=minutes)
            candidates = pd.date_range(r.start.ceil("15min"), latest, freq="15min") if latest > r.start else None
            # Row 0 is the run as it happened; rows 1.. are the candidate starts.
            starts = pd.DatetimeIndex([r.start]).append(candidates) if candidates is not None else pd.DatetimeIndex([r.start])
            times = step_times(starts, offsets)
            shape = (len(starts), len(lengths))
            price = {t: p.at(times).reshape(shape) for t, p in prices.items()}
            real_left = lookup(available, times, shape)
            hours = pd.date_range(r.start.floor("h"), deadline_for(r.start).ceil("h"), freq="h")
            load = typical.reindex(hours).fillna(house_w.median())
            arm_day = r.start.normalize()
            naive = (climatology.loc[arm_day] if arm_day in climatology.index else None)
            forecast_left = {
                "nopanel": None,
                "panel": (pv.reindex(hours).fillna(0.0) - load).clip(lower=0.0),
                "panel_naive": ((profile_forecast(naive, hours) - load).clip(lower=0.0)
                                if naive is not None and naive.notna().all() else None),
            }
            ok_night = np.r_[True, np.ones(len(starts) - 1, bool)]
            ok_quiet = np.r_[True, allowed(starts[1:])] if len(starts) > 1 else ok_night
            row = {"house": a.house, "appliance": a.appliance, "type": appliance_type(a.appliance),
                   "start": r.start, "kwh": r.energy_wh / 1000}
            for setup in SETUPS:
                if setup == "panel_naive" and forecast_left[setup] is None:
                    for t in TARIFFS:
                        for rule in ("night", "quiet"):
                            row[f"{setup}_{t}_{rule}"] = np.nan
                        row[f"{setup}_{t}_actual"] = np.nan
                    continue
                left_fc = lookup(forecast_left[setup], times, shape)
                left_real = np.zeros(shape) if setup == "nopanel" else real_left
                for t in TARIFFS:
                    real = costs(watts, lengths, left_real, price[t])
                    row[f"{setup}_{t}_actual"] = real[0]
                    if len(starts) == 1:  # no time left to choose: the run starts now
                        row[f"{setup}_{t}_night"] = row[f"{setup}_{t}_quiet"] = real[0]
                        continue
                    planned = costs(watts, lengths, left_fc, price[t])
                    for rule, ok in (("night", ok_night), ("quiet", ok_quiet)):
                        pick = np.flatnonzero(ok[1:]) + 1  # a candidate, never the as-happened row
                        best = pick[np.argmin(planned[pick])] if len(pick) else 0
                        row[f"{setup}_{t}_{rule}"] = real[best]
            rows.append(row)
    return pd.DataFrame(rows)


def per_home_per_year(runs: pd.DataFrame, col: str) -> pd.Series:
    """Pounds per year per home for a cost column (pence per run), over each home's own span."""
    span = runs.groupby("house")["start"].agg(lambda s: max((s.max() - s.min()).days / 365, 0.1))
    return runs.groupby("house")[col].sum() / 100 / span


def main() -> None:
    uses, _, appliances = load_prepared()
    runs = replay(uses, appliances, load_aggregate(), hourly_pv_w(PV_KWP))
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS / "smart_start_tariffs.csv", index=False)
    print(f"Cost-aware smart start: {len(runs)} runs in {runs['house'].nunique()} homes, Octopus 2019 prices "
          f"(region {REGION}), export {EXPORT} p/kWh\n")
    for setup in SETUPS:
        sub = runs.dropna(subset=[f"{setup}_{t}_actual" for t in TARIFFS])
        print(f"  {setup} ({len(sub)} runs): machine cost per home, median GBP/year "
              f"[as happened -> smart start, night allowed / quiet hours]")
        for t in TARIFFS:
            actual = per_home_per_year(sub, f"{setup}_{t}_actual")
            parts = []
            for rule in ("night", "quiet"):
                smart = per_home_per_year(sub, f"{setup}_{t}_{rule}")
                saved = actual - smart
                p = wilcoxon(actual, smart, alternative="greater").pvalue if (saved != 0).any() else 1.0
                share = (sub[f"{setup}_{t}_actual"].sum() - sub[f"{setup}_{t}_{rule}"].sum()) / sub[f"{setup}_{t}_actual"].sum()
                parts.append(f"{smart.median():6.2f} (saves {saved.median():5.2f}, {share:5.1%}; "
                             f"cheaper in {(saved > 0).sum()}/{len(saved)}, p = {p:.1e})")
            print(f"    {t:<9} {actual.median():6.2f} -> " + " / ".join(parts))
        print()


if __name__ == "__main__":
    main()
