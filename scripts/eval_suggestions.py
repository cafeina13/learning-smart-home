"""Evaluate the energy suggestions on real REFIT homes.

    .venv\\Scripts\\python.exe scripts\\eval_suggestions.py [cuts] [smart]   (both if none given)

A. Standby cuts (manual appliances with a measurable standby draw).
   Days are split 60 / 20 / 20. A safety margin on the window edges is chosen on
   the validation 20% (the smallest whose median interruption rate is under 5%).
   Quiet windows are then LEARNED from each home's first 80% of days (weekdays and
   weekends separately) and CHECKED once on its last 20%, with and without the
   system's own safeguard (three overrides switch a rule off), and with and
   without the habit forest (habits/forest.py), trained on the same learning
   days only: its check (a window it agrees with on fewer than half of its
   unseen check days is not suggested) and its hourly guard (power restored
   at the start of any cut hour it flags):
     - interruption rate: share of cut windows in which a use started while the
       power was cut (the person would have had to press the plug: an override).
       A use already running when the window opens is not an interruption: the
       cut waits until it ends.
     - energy saved: standby watts x cut hours, counting only up to the first
       interruption in each window (the person restores the power then).

B. Smart start (dishwashers, washing machines, dryers).
   Nothing is learned per home: the rule is "finish by 18:00, tomorrow if armed
   after 14:00", applied to EVERY real run in the data (all seasons). For each
   run: solar energy it got at its real start vs. at the start the scheduler
   would pick. Solar is real (PVGIS, Loughborough, the same day and hour) for a
   PV_KWP system, minus the rest of the house's real consumption that hour (the
   house uses the panel first). The scheduler picks by that day's solar forecast,
   assumed perfect: an optimistic assumption.

Writes data/results/standby_cuts.csv and data/results/smart_start.csv.
"""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from habits.paths import RESULTS as PATHS_RESULTS  # noqa: E402

from scipy.stats import wilcoxon  # noqa: E402

from habits.dataset import covered_days, hourly_labels  # noqa: E402
from habits.evaluate import MIN_USES, appliance_type  # noqa: E402
from habits.features import other_appliance_starts  # noqa: E402
from habits.forest import MIN_AGREEMENT, HabitForest, restore_time  # noqa: E402
from habits.prepare import load_aggregate, load_prepared  # noqa: E402
from habits.routines import WEEKDAYS, WEEKEND, find_quiet_window, slot_time  # noqa: E402
from habits.solar import hourly_pv_w  # noqa: E402
from habits.suggestions import control_of  # noqa: E402

RESULTS = PATHS_RESULTS  # data/results, or data/results/ukdale
PV_KWP = 3.0  # a typical UK home system
MIN_STANDBY_W = 0.5  # below this a cut saves too little to suggest
DEADLINE = time(18, 0)


# --- A. standby cuts --------------------------------------------------------------------------

def check_window(uses: pd.DataFrame, start: datetime, end: datetime, standby_w: float) -> tuple[bool, float]:
    """(interrupted?, kWh saved) for one cut window on one real night."""
    started_inside = uses[(uses["start"] >= start) & (uses["start"] < end)]
    running_at_start = uses[(uses["start"] < start) & (uses["end"] > start)]
    cut_from = max([start] + list(running_at_start["end"]))  # the cut waits for a use in progress
    cut_until = min([end] + list(started_inside["start"]))  # the first use inside ends the cut
    hours = max(0.0, (cut_until - cut_from).total_seconds() / 3600)
    return len(started_inside) > 0, standby_w * hours / 1000


MARGINS_MIN = (0, 30, 60, 90, 120, 180)  # trimmed off EACH edge of a learned window
TARGET_INTERRUPTIONS = 0.05
MIN_WINDOW_SLOTS = 8  # a trimmed window shorter than 4 hours is dropped


def learn_window(uses, days, appliance, kind, margin_min: int):
    """(start slot, length in slots) of the learned quiet window, trimmed by `margin_min` on each side."""
    q = find_quiet_window(uses, days, appliance, kind)
    if q is None:
        return None
    m = margin_min // 30
    if q.n_slots - 2 * m < MIN_WINDOW_SLOTS:
        return None
    return q.start_slot + m, q.n_slots - 2 * m


def score_window(u, window, days, kind, standby_w, max_overrides: int | None = None,
                 guard: tuple[pd.Series, float] | None = None, forest_strikes: bool = False,
                 ) -> tuple[int, int, float]:
    """(windows the rule was active, windows interrupted, kWh saved) over the given days, in date order.

    With max_overrides, the system's own safeguard is applied: after that many
    interruptions the rule switches itself off (as RuleEngine does after three
    overrides), and saves nothing on later days. With guard = (the forest's
    hourly probabilities, its threshold), the cut ends when the forest's guard
    restores the power (habits/forest.py: restore_time). With forest_strikes,
    a night the guard restored the power before anyone was interrupted also
    counts towards max_overrides (as RuleEngine does with FOREST_STRIKES).
    """
    start_slot, n_slots = window
    day_set = set(days)
    windows = interrupted = strikes = 0
    kwh = 0.0
    for d in sorted(days):
        if d.weekday() not in kind or d + timedelta(days=1) not in day_set:
            continue
        if max_overrides is not None and strikes >= max_overrides:
            break  # the rule disabled itself
        start = datetime.combine(d, time()) + timedelta(minutes=30 * start_slot)
        end = full_end = start + timedelta(minutes=30 * n_slots)
        if guard is not None:
            end = restore_time(start, full_end, *guard)
        hit, saved = check_window(u, start, end, standby_w)
        windows += 1
        interrupted += hit
        strikes += hit or (forest_strikes and end < full_end)
        kwh += saved
    return windows, interrupted, kwh


def standby_candidates(uses, covered, appliances):
    """(appliance row, its uses, days split 60/20/20) for every appliance a cut is worth suggesting for."""
    for a, u, train, val, test, _ in standby_candidates_with_house(uses, covered, appliances):
        yield a, u, train, val, test


def standby_candidates_with_house(uses, covered, appliances):
    """As standby_candidates, plus all the house's uses (the forest needs the other appliances)."""
    uses = uses.merge(appliances[["house", "column", "category"]], on=["house", "column"], how="left")
    for a in appliances.itertuples():
        if control_of(a.appliance) != "manual" or a.standby_w < MIN_STANDBY_W or a.uses < MIN_USES:
            continue
        days = covered_days(covered.loc[covered["house"] == a.house, "hour"])
        n = len(days)
        train, val, test = days[:int(n * 0.6)], days[int(n * 0.6):int(n * 0.8)], days[int(n * 0.8):]
        u = uses[(uses["house"] == a.house) & (uses["column"] == a.column)]
        yield a, u, train, val, test, uses[uses["house"] == a.house]


def habit_forest(a, u, house_uses, covered, learn_days, test_days):
    """The forest trained on the learning days only, and its probability for every hour (each from
    the hours before it, as the guard sees them live). (None, None) if there is nothing to learn."""
    hours = pd.DatetimeIndex(covered.loc[covered["house"] == a.house, "hour"]).sort_values()
    others = other_appliance_starts(house_uses, a.column)
    learn, end = set(learn_days), pd.Timestamp(test_days[0])
    forest = HabitForest.fit(u[u["start"] < end], hours[[d in learn for d in hours.date]],
                             others[others.index < end])
    if forest is None:
        return None, None
    return forest, forest.predict(hourly_labels(u, hours), others)


def choose_margin(uses, covered, appliances) -> tuple[int, pd.DataFrame]:
    """The smallest edge margin whose median interruption rate on the VALIDATION days is under target."""
    rows = []
    for a, u, train, val, _ in standby_candidates(uses, covered, appliances):
        train_uses = u[u["start"] < pd.Timestamp(val[0])]
        for kind in (WEEKDAYS, WEEKEND):
            for margin in MARGINS_MIN:
                window = learn_window(train_uses, train, a.appliance, kind, margin)
                if window is None:
                    continue
                n, hit, _ = score_window(u, window, val, kind, a.standby_w)
                if n:
                    rows.append({"margin_min": margin, "house": a.house, "column": a.column,
                                 "kind": len(kind), "rate": hit / n})
    table = pd.DataFrame(rows).groupby("margin_min")["rate"].agg(windows="size", median_rate="median").reset_index()
    ok = table[table["median_rate"] < TARGET_INTERRUPTIONS]
    margin = int(ok["margin_min"].min()) if len(ok) else MARGINS_MIN[-1]
    return margin, table


def standby_cuts(uses, covered, appliances, margin_min: int = 0, with_forest: bool = False) -> pd.DataFrame:
    """Learn on the first 80% of days (train + validation) with the given margin; score on the last 20%."""
    rows = []
    for a, u, train, val, test, house_uses in standby_candidates_with_house(uses, covered, appliances):
        learn_days = train + val
        learn_uses = u[u["start"] < pd.Timestamp(test[0])]
        forest = probability = None
        if with_forest:
            forest, probability = habit_forest(a, u, house_uses, covered, learn_days, test)
        for kind, kind_name in ((WEEKDAYS, "weekdays"), (WEEKEND, "weekends")):
            window = learn_window(learn_uses, learn_days, a.appliance, kind, margin_min)
            if window is None:
                continue
            n, hit, kwh = score_window(u, window, test, kind, a.standby_w)
            # The same window with the system's safeguard: three overrides switch the rule off.
            all_nights = n
            _, hit3, kwh3 = score_window(u, window, test, kind, a.standby_w, max_overrides=3)
            if n:
                start_slot, n_slots = window
                rows.append({"house": a.house, "column": a.column, "appliance": a.appliance,
                             "type": appliance_type(a.appliance), "days": kind_name, "margin_min": margin_min,
                             "window": f"{slot_time(start_slot):%H:%M}-{slot_time(start_slot + n_slots):%H:%M}",
                             "start_slot": start_slot, "n_slots": n_slots, "hours": n_slots / 2,
                             "standby_w": a.standby_w, "test_windows": n,
                             "interruption_rate": round(hit / n, 3),
                             "kwh_saved_per_year": round(kwh / n * len(kind) * 52, 2),
                             # with the 3-override safeguard: interruptions per test night, and savings
                             # over all test nights (nights after the rule switched off save nothing)
                             "safeguard_interruption_rate": round(hit3 / all_nights, 3),
                             "safeguard_kwh_per_year": round(kwh3 / all_nights * len(kind) * 52, 2),
                             "safeguard_disabled": hit3 >= 3,
                             **forest_columns(u, window, test, kind, a.standby_w, forest, probability, all_nights)})
    return pd.DataFrame(rows)


# --- B. smart start ---------------------------------------------------------------------------

def machine_hourly_w(runs: pd.DataFrame) -> pd.Series:
    """The machine's own average power per hour, from its runs (to subtract it from the aggregate)."""
    parts = {}
    for r in runs.itertuples():
        watts = r.energy_wh / (r.duration_s / 3600) if r.duration_s else 0.0
        t = r.start
        while t < r.end:
            hour = t.floor("h")
            step = min(hour + pd.Timedelta(hours=1), r.end) - t
            parts[hour] = parts.get(hour, 0.0) + watts * step.total_seconds() / 3600
            t += step
    return pd.Series(parts, dtype=float)


def solar_kwh(start: pd.Timestamp, minutes: float, watts: float, available: pd.Series) -> float:
    """kWh of a run starting at `start` covered by the solar left over for it (5-minute steps)."""
    kwh, t, end = 0.0, start, start + pd.Timedelta(minutes=minutes)
    while t < end:
        step = min(pd.Timedelta(minutes=5), end - t)
        kwh += min(watts, available.get(t.floor("h"), 0.0)) * step.total_seconds() / 3600 / 1000
        t += step
    return kwh


def solar_kwh_many(starts, minutes: float, watts: float, available: pd.Series) -> np.ndarray:
    """solar_kwh for many start times of one run at once (the same 5-minute steps, in numpy)."""
    starts = pd.DatetimeIndex(starts)
    if len(starts) == 0:
        return np.empty(0)
    n_full, rest = divmod(minutes, 5.0)
    lengths = np.r_[np.full(int(n_full), 5.0), [rest] if rest > 1e-9 else []]  # minutes per step
    offsets = np.r_[0.0, np.cumsum(lengths)[:-1]]
    times = starts.values[:, None] + (offsets * 60e9).astype("timedelta64[ns]")[None, :]
    hours = pd.DatetimeIndex(times.ravel()).floor("h")
    pos = available.index.get_indexer(hours)
    values = np.where(pos >= 0, available.to_numpy()[np.maximum(pos, 0)], 0.0).reshape(times.shape)
    return (np.minimum(watts, values) * lengths[None, :] / 60 / 1000).sum(axis=1)


def deadline_for(start: pd.Timestamp) -> pd.Timestamp:
    day = start.normalize() + pd.Timedelta(days=1 if start.time() >= time(14, 0) else 0)
    return day + pd.Timedelta(hours=DEADLINE.hour)


def smart_start(uses, appliances, aggregate, pv) -> pd.DataFrame:
    rows = []
    for a in appliances.itertuples():
        if control_of(a.appliance) != "armable" or a.uses < MIN_USES:
            continue
        runs = uses[(uses["house"] == a.house) & (uses["column"] == a.column)]
        runs = runs[(runs["start"] >= pv.index.min()) & (runs["end"] <= pv.index.max())]
        house_w = aggregate.loc[aggregate["house"] == a.house].set_index("hour")["aggregate_w"]
        other = house_w.sub(machine_hourly_w(runs), fill_value=0.0).clip(lower=0.0)
        other = other.reindex(pv.index).fillna(house_w.median())
        available = (pv - other).clip(lower=0.0)  # solar left over after the rest of the house
        # Load-aware ranking: the rest of the house's typical load at this hour of the week over the
        # 4 weeks before (known when the machine is armed), subtracted from the solar forecast.
        expected_other = pd.concat([other.shift(168 * k) for k in range(1, 5)], axis=1).median(axis=1)
        expected_left = (pv - expected_other.fillna(house_w.median())).clip(lower=0.0)
        for r in runs.itertuples():
            minutes = r.duration_s / 60
            watts = r.energy_wh / (r.duration_s / 3600)
            actual = solar_kwh(r.start, minutes, watts, available)
            latest = deadline_for(r.start) - pd.Timedelta(minutes=minutes)
            candidates = pd.date_range(r.start.ceil("15min"), latest, freq="15min") if latest > r.start else []
            if len(candidates):
                # The scheduler ranks by the solar forecast for the panel (it cannot know the rest of
                # the house's load in advance); the chosen start is then scored with what was really left.
                forecast = solar_kwh_many(candidates, minutes, watts, pv)
                chosen = candidates[int(np.argmax(forecast))] if max(forecast) > 0 else r.start
                # The load-aware variant ranks by the solar expected to be left after the house's typical load.
                left = solar_kwh_many(candidates, minutes, watts, expected_left)
                chosen_la = candidates[int(np.argmax(left))] if max(left) > 0 else chosen
            else:
                chosen = chosen_la = r.start
            shifted = solar_kwh(chosen, minutes, watts, available)
            shifted_la = solar_kwh(chosen_la, minutes, watts, available)
            rows.append({"house": a.house, "appliance": a.appliance, "type": appliance_type(a.appliance),
                         "start": r.start, "chosen": chosen, "kwh": r.energy_wh / 1000,
                         "solar_kwh_actual": actual, "solar_kwh_smart": shifted,
                         "chosen_load_aware": chosen_la, "solar_kwh_load_aware": shifted_la})
    return pd.DataFrame(rows)


def forest_columns(u, window, test, kind, standby_w, forest, probability, nights) -> dict:
    """The forest's check (its agreement on its unseen check days) and its guard on the test nights,
    alone and together with the 3-override safeguard."""
    if forest is None:
        return {"forest_trained": False}
    start_slot, n_slots = window
    agreement = forest.agreement(kind, start_slot, n_slots, expect_use=False)
    guard = (probability, forest.threshold)
    _, hit, kwh = score_window(u, window, test, kind, standby_w, guard=guard)
    _, hit3, kwh3 = score_window(u, window, test, kind, standby_w, max_overrides=3, guard=guard)
    _, hits, kwhs = score_window(u, window, test, kind, standby_w, max_overrides=3, guard=guard, forest_strikes=True)
    per_year = len(kind) * 52 / nights
    return {"forest_trained": True, "forest_threshold": round(forest.threshold, 3),
            "forest_agreement": round(agreement, 3), "forest_held": bool(agreement < MIN_AGREEMENT),
            "guard_interruption_rate": round(hit / nights, 3), "guard_kwh_per_year": round(kwh * per_year, 2),
            "guard_safeguard_interruption_rate": round(hit3 / nights, 3),
            "guard_safeguard_kwh_per_year": round(kwh3 * per_year, 2),
            "guard_strikes_interruption_rate": round(hits / nights, 3),
            "guard_strikes_kwh_per_year": round(kwhs * per_year, 2)}


def report_forest(cuts: pd.DataFrame) -> None:
    """Interruptions and savings without and with the forest, on the windows it could be trained for."""
    c = cuts[cuts["forest_trained"].astype(bool)].copy()
    print(f"\n  The habit forest (habits/forest.py), trained on the learning days only: "
          f"{len(c)} of {len(cuts)} windows (the rest: nothing to learn)")
    if not len(c):
        return
    c["forest_held"] = c["forest_held"].astype(bool)
    kept = c[~c["forest_held"]]
    variants = [
        ("counting only", c, "interruption_rate", "kwh_saved_per_year"),
        ("+ 3-override rule", c, "safeguard_interruption_rate", "safeguard_kwh_per_year"),
        ("+ forest guard", c, "guard_interruption_rate", "guard_kwh_per_year"),
        ("+ guard + 3-override", c, "guard_safeguard_interruption_rate", "guard_safeguard_kwh_per_year"),
        ("+ guard rescues = strikes", c, "guard_strikes_interruption_rate", "guard_strikes_kwh_per_year"),
        ("forest check: kept only", kept, "interruption_rate", "kwh_saved_per_year"),
        ("check+guard+3-override", kept, "guard_safeguard_interruption_rate", "guard_safeguard_kwh_per_year"),
    ]
    for label, d, rate, kwh in variants:
        if len(d):
            report_cuts(d.assign(interruption_rate=d[rate], kwh_saved_per_year=d[kwh]), label)
    print(f"  forest check held back {c['forest_held'].sum()} of {len(c)} windows; mean interruption rate "
          f"(counting only) of the held-back ones {c.loc[c['forest_held'], 'interruption_rate'].mean():.1%}, "
          f"of the kept ones {kept['interruption_rate'].mean():.1%}")
    for a, b in (("interruption_rate", "guard_interruption_rate"),
                 ("safeguard_interruption_rate", "guard_safeguard_interruption_rate"),
                 ("guard_safeguard_interruption_rate", "guard_strikes_interruption_rate"),
                 ("safeguard_interruption_rate", "guard_strikes_interruption_rate"),
                 ("kwh_saved_per_year", "guard_kwh_per_year"),
                 ("safeguard_kwh_per_year", "guard_strikes_kwh_per_year")):
        d = c[b] - c[a]
        p = wilcoxon(c[b], c[a], alternative="less").pvalue if d.any() else 1.0
        print(f"  {b} vs {a}: lower on {(d < 0).sum()}, higher on {(d > 0).sum()} of {len(d)} windows "
              f"(one-sided Wilcoxon p = {p:.1e})")


def report_cuts(cuts: pd.DataFrame, label: str) -> None:
    per_home = cuts.groupby("house")["kwh_saved_per_year"].sum()
    print(f"  {label:<22} {len(cuts):>3} windows, {cuts['house'].nunique():>2} homes | median interruption "
          f"{cuts['interruption_rate'].median():5.1%} | {(cuts['interruption_rate'] < TARGET_INTERRUPTIONS).mean():4.0%} "
          f"of windows under 5% | saved per home: median {per_home.median():5.1f} kWh/year")


def main(parts: set[str]) -> None:
    uses, covered, appliances = load_prepared()
    pd.set_option("display.width", 200)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if "cuts" in parts:
        margin, table = choose_margin(uses, covered, appliances)
        print("A. Standby cuts. Edge margin chosen on the VALIDATION days (learned on the first 60%):\n")
        print(table.round(3).to_string(index=False))
        print(f"\n  -> chosen margin: {margin} min on each edge (smallest with median interruption < 5%)\n")
        print("  Learned on the first 80% of days, scored once on the last 20% (unseen):")
        plain = standby_cuts(uses, covered, appliances, margin_min=0, with_forest=True)
        tuned = standby_cuts(uses, covered, appliances, margin_min=margin)
        report_cuts(plain, "no margin")
        if margin:
            report_cuts(tuned, f"{margin}-min margin")
        guarded = plain.assign(interruption_rate=plain["safeguard_interruption_rate"],
                               kwh_saved_per_year=plain["safeguard_kwh_per_year"])
        report_cuts(guarded, "with 3-override rule")
        print(f"  ({plain['safeguard_disabled'].sum()} of {len(plain)} rules switched themselves off "
              f"during the test days)")
        report_forest(plain)
        # With a 0-min margin, `tuned` is the same windows as `plain` (without the forest): write them once.
        (pd.concat([plain, tuned]) if margin else plain).to_csv(RESULTS / "standby_cuts.csv", index=False)
        print("\n  by type, with the chosen margin:")
        print(tuned.groupby("type").agg(windows=("house", "size"), standby_w=("standby_w", "median"),
                                        interruption_rate=("interruption_rate", "median"),
                                        kwh_per_year=("kwh_saved_per_year", "median")).round(3).to_string())

    if "smart" not in parts:
        return
    runs = smart_start(uses, appliances, load_aggregate(), hourly_pv_w(PV_KWP))
    runs.to_csv(RESULTS / "smart_start.csv", index=False)
    print(f"\nB. Smart start: {len(runs)} real machine runs in {runs['house'].nunique()} homes, all seasons, "
          f"{PV_KWP:g} kWp PV (PVGIS, real weather)\n")
    by = runs.groupby("type").agg(runs=("kwh", "size"), kwh=("kwh", "sum"),
                                  actual=("solar_kwh_actual", "sum"), smart=("solar_kwh_smart", "sum"))
    by["solar_share_now"] = (by["actual"] / by["kwh"]).round(3)
    by["solar_share_smart"] = (by["smart"] / by["kwh"]).round(3)
    print(by[["runs", "solar_share_now", "solar_share_smart"]].to_string())
    total, now, smart = runs["kwh"].sum(), runs["solar_kwh_actual"].sum(), runs["solar_kwh_smart"].sum()
    print(f"\n  all runs: solar share {now / total:.1%} as they happened -> {smart / total:.1%} with smart start")
    # Per home, over that home's own recording period (homes were recorded for different spans).
    per_home = runs.groupby("house").apply(
        lambda g: (g["solar_kwh_smart"].sum() - g["solar_kwh_actual"].sum())
        / max((g["start"].max() - g["start"].min()).days / 365, 0.1), include_groups=False)
    print(f"  moved from grid to solar per home: median {per_home.median():.0f} kWh/year "
          f"(range {per_home.min():.0f}-{per_home.max():.0f})")

    # Load-aware smart start: ranked by the solar expected to be left after the house's typical load.
    la = runs["solar_kwh_load_aware"].sum()
    print(f"\n  load-aware smart start: solar share {la / total:.1%} (solar-only ranking {smart / total:.1%})")
    homes = runs.groupby("house")[["solar_kwh_smart", "solar_kwh_load_aware"]].sum()
    d = homes["solar_kwh_load_aware"] - homes["solar_kwh_smart"]
    p = wilcoxon(homes["solar_kwh_load_aware"], homes["solar_kwh_smart"], alternative="greater").pvalue if d.any() else 1.0
    moved = runs["chosen_load_aware"] != runs["chosen"]
    print(f"  more solar in {(d > 0).sum()}, less in {(d < 0).sum()} of {len(d)} homes (one-sided Wilcoxon p = {p:.1e}); "
          f"a different start for {moved.mean():.0%} of runs; "
          f"per run where it differs: {(runs.loc[moved, 'solar_kwh_load_aware'] - runs.loc[moved, 'solar_kwh_smart']).mean() * 1000:+.0f} Wh")
    by_season = runs.assign(season=pd.to_datetime(runs["start"]).dt.month.map(
        lambda m: "winter" if m in (11, 12, 1, 2) else "summer" if m in (5, 6, 7, 8) else "spring/autumn"))
    for season, g in by_season.groupby("season"):
        print(f"    {season:<14} solar share: as happened {g['solar_kwh_actual'].sum() / g['kwh'].sum():.1%}, "
              f"solar-only {g['solar_kwh_smart'].sum() / g['kwh'].sum():.1%}, "
              f"load-aware {g['solar_kwh_load_aware'].sum() / g['kwh'].sum():.1%}")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"cuts", "smart"})  # e.g. "eval_suggestions.py cuts" runs only part A
