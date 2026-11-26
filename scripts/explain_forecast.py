"""Explain the load forecaster with SHAP: what drives a home's next-hour load forecast.

    python scripts/explain_forecast.py [--max-homes N]       (CAPSTONE_DATASET=ukdale; runs on Kaggle)

The model is the best forecaster of scripts/eval_forecast_habits.py: gradient boosting trained on
the absolute error, with every input (past load, calendar, weather, the habit models' predictions
and which appliances ran in the past hour), trained on each home's first 80% of hours. TreeSHAP
splits each forecast (watts) into one contribution per input, on up to 1,000 of the home's test
hours. Inputs are grouped (for example every lag and mean of the past load is "past load", every
appliance's habit probability and expected energy is "habit predictions"); per home, a group's
mean |SHAP| becomes a share of the home's total, and the summary is the mean share over homes.

Writes data/results/forecast_shap.csv (one row per home and input) and data/results/figures/
xai-forecast-importance-<dataset>.png. Compare Moon et al. (2024), who explain regional
building-load forecasters the same way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from energy.data import FEATURES, house_hours, weather, windows  # noqa: E402
from eval_forecast_habits import habit_features, load_features, make  # noqa: E402
from explain_habits import FIGURES, importance_figure  # noqa: E402
from habits.paths import DATASET, RESULTS  # noqa: E402
from habits.prepare import load_aggregate, load_prepared  # noqa: E402

MAX_HOURS = 1000


def group_of(column: str) -> str:
    if column.startswith(("lag_", "mean_", "std_", "max_")):
        return "Past load (lags, means, spread)"
    if column in ("temperature_1h", "irradiance_1h", "wind_1h"):
        return "Weather, previous hour"
    if column.startswith(("p_", "wh_")) or column == "expected_wh":
        return "Habit predictions for the hour"
    if column.startswith("on_"):
        return "Appliances running, previous hour"
    return {"hour": "Hour of day", "weekday": "Day of week", "month": "Month"}[column]


def explain_house(house: int, aggregate, wx, prepared, seed: int = 0) -> pd.DataFrame:
    X, y, when = windows(house_hours(house, aggregate, wx), FEATURES)
    b = int(len(X) * 0.8)  # as eval_forecast_habits.py: train on the first 80%, test on the rest
    habits, devices = habit_features(house, *prepared, when, cutoff=when[b])
    inputs = load_features(X, when).join(habits).join(devices)
    F = inputs.to_numpy(dtype="float32")
    model = make("gradient_boosting").fit(F[:b], y[:b])
    test = np.arange(b, len(X))
    if len(test) > MAX_HOURS:
        test = np.sort(np.random.default_rng(seed).choice(test, MAX_HOURS, replace=False))
    values = shap.TreeExplainer(model).shap_values(F[test])
    per_column = pd.DataFrame(values, columns=inputs.columns)
    per_group = per_column.T.groupby(group_of).sum().T  # signed sums within a group, per hour
    magnitude = per_group.abs().mean()
    return pd.DataFrame({"house": house, "input": magnitude.index, "mean_abs_shap_w": magnitude.to_numpy(),
                         "shap_share": (magnitude / magnitude.sum()).to_numpy()})


def main(max_homes: int | None = None) -> None:
    aggregate, wx, prepared = load_aggregate(), weather(), load_prepared()
    homes = sorted(aggregate["house"].unique())[:max_homes]
    table = pd.concat([explain_house(h, aggregate, wx, prepared) for h in homes], ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "forecast_shap.csv", index=False)
    summary = table.groupby("input")[["shap_share", "mean_abs_shap_w"]].mean().sort_values("shap_share", ascending=False)
    print(f"SHAP for the gradient-boosting load forecaster, {DATASET}: {len(homes)} homes, "
          f"up to {MAX_HOURS} test hours each\n")
    print(summary.assign(shap_share=lambda d: (d.shap_share * 100).round(1),
                         mean_abs_shap_w=lambda d: d.mean_abs_shap_w.round(1)).to_string())
    FIGURES.mkdir(parents=True, exist_ok=True)
    importance_figure(summary, len(homes), FIGURES / f"xai-forecast-importance-{DATASET}.png",
                      unit="homes")
    print(f"\nFigure: {FIGURES}/xai-forecast-importance-{DATASET}.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[args.index("--max-homes") + 1]) if "--max-homes" in args else None)
