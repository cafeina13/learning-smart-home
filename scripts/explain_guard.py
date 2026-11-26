"""Why the forest guard restored power, on real test nights: the reasons behind rescues and false alarms.

    python scripts/explain_guard.py          (CAPSTONE_DATASET=ukdale for UK-DALE; runs on Kaggle)

The same standby cuts and forest guard as scripts/eval_suggestions.py (windows learned on each
home's first 80% of days, no margin; the forest trained on those days only): on every test night,
the guard restores the power at the start of the first hour its forest flags. For each restore,
the forest's reasons for that hour (habits/explain.py, SHAP: the input groups pushing the
prediction most towards a switch-on) are recorded, together with what really happened:

    rescue        a use started after the restore, before the window ended (the person would
                  otherwise have found the plug cut)
    false alarm   no use started in the rest of the window

Writes data/results/guard_reasons.csv (one row per restore) and prints how often each input group
is the main reason, and among the top three, for rescues and for false alarms.
"""

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from eval_suggestions import habit_forest, learn_window, standby_candidates_with_house  # noqa: E402
from habits.dataset import hourly_labels  # noqa: E402
from habits.evaluate import appliance_type  # noqa: E402
from habits.features import build_features, other_appliance_starts  # noqa: E402
from habits.forest import restore_time  # noqa: E402
from habits.paths import DATASET, RESULTS  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402
from habits.routines import WEEKDAYS, WEEKEND  # noqa: E402


def restores(uses, covered, appliances, max_appliances: int | None = None) -> pd.DataFrame:
    rows = []
    for i, (a, u, train, val, test, house_uses) in enumerate(standby_candidates_with_house(uses, covered, appliances)):
        if max_appliances is not None and i >= max_appliances:
            break
        learn_days = train + val
        forest, probability = habit_forest(a, u, house_uses, covered, learn_days, test)
        if forest is None:
            continue
        hours = pd.DatetimeIndex(covered.loc[covered["house"] == a.house, "hour"]).sort_values()
        X = build_features(hourly_labels(u, hours), other_appliance_starts(house_uses, a.column))
        explainer = forest.explainer()
        learn_uses = u[u["start"] < pd.Timestamp(test[0])]
        test_set = set(test)
        for kind, kind_name in ((WEEKDAYS, "weekdays"), (WEEKEND, "weekends")):
            window = learn_window(learn_uses, learn_days, a.appliance, kind, 0)
            if window is None:
                continue
            start_slot, n_slots = window
            for d in sorted(test):
                if d.weekday() not in kind or d + timedelta(days=1) not in test_set:
                    continue
                start = datetime.combine(d, time()) + timedelta(minutes=30 * start_slot)
                end = start + timedelta(minutes=30 * n_slots)
                restored = restore_time(start, end, probability, forest.threshold)
                hour = pd.Timestamp(restored).floor("h")
                if restored >= end or hour not in X.index:
                    continue  # no restore that night
                later = u[(u["start"] >= restored) & (u["start"] < end)]
                reasons = explainer.reasons(X.loc[[hour]], top=3)
                row = {"house": a.house, "appliance": a.appliance, "type": appliance_type(a.appliance),
                       "days": kind_name, "night": d, "restored": restored,
                       "probability": round(float(probability.get(hour, float("nan"))), 3),
                       "outcome": "rescue" if len(later) else "false alarm",
                       "minutes_to_use": (later["start"].min() - restored).total_seconds() / 60 if len(later) else None}
                for k, r in enumerate(reasons, 1):
                    row[f"reason{k}"], row[f"phrase{k}"], row[f"contribution{k}"] = r.group, r.phrase, round(r.contribution, 4)
                rows.append(row)
    return pd.DataFrame(rows)


def main(max_appliances: int | None = None) -> None:
    uses, covered, appliances = load_prepared()
    table = restores(uses, covered, appliances, max_appliances)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "guard_reasons.csv", index=False)
    if table.empty:
        print("no restores")
        return
    print(f"Forest guard restores on the test nights, {DATASET}: {len(table)} restores, "
          f"{table['house'].nunique()} homes, {table.groupby(['house', 'appliance']).ngroups} appliances")
    print(table["outcome"].value_counts().to_string(), "\n")
    for outcome, g in table.groupby("outcome"):
        top1 = g["reason1"].value_counts(normalize=True)
        in_top3 = pd.concat([g[c] for c in ("reason1", "reason2", "reason3") if c in g]).value_counts() / len(g)
        summary = pd.DataFrame({"main reason": top1, "among top three": in_top3}).fillna(0).sort_values(
            "main reason", ascending=False)
        print(f"{outcome} ({len(g)}): share of restores")
        print((summary * 100).round(1).to_string(), "\n")
        example = g.iloc[0]
        print(f"  e.g. {example['appliance']}, house {example['house']}, {example['restored']:%a %d %b %H:%M}: "
              f"{'; '.join(str(example[c]) for c in ('phrase1', 'phrase2', 'phrase3') if pd.notna(example.get(c)))}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[args.index("--max-appliances") + 1]) if "--max-appliances" in args else None)
