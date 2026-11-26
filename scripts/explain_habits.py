"""Explain the habit random forests with SHAP: which inputs drive the predictions, and how.

    python scripts/explain_habits.py [--max-tasks N]     (CAPSTONE_DATASET=ukdale or casas; runs on Kaggle)

The forests are the ones of scripts/eval_refit.py: same tasks, same 15 inputs, same settings
(habits/models.py), trained on each task's first 60% of hours. TreeSHAP (Lundberg et al., the
`shap` library) splits each prediction, the forest's probability that the appliance is switched
on in that hour, into one contribution per input. It is computed on up to 1,000 test hours per
task, hours the forest never saw.

Two inputs that only mean something together are summed before taking magnitudes: hour of day
(its sine and cosine) and day of week (likewise). Per task, each input's mean |SHAP| is turned
into a share of the task's total, so every task counts equally; the summary is the mean share
over tasks. The forests' own impurity importances (report section 6.3) are kept next to it for
comparison: they are biased towards inputs with many distinct values, SHAP is not.

Writes data/results/habit_shap_<model>.csv (one row per task and input) and two figures in
data/results/figures/: xai-habit-importance-<dataset>-<model>.png and xai-habit-example-<dataset>-<model>.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from habits.evaluate import appliance_type, tasks  # noqa: E402
from habits.features import build_features  # noqa: E402
from habits.models import make_model  # noqa: E402
from habits.paths import DATASET, RESULTS, ROOT  # noqa: E402
from habits.prepare import load_prepared  # noqa: E402

MAX_HOURS = 1000
FIGURES = RESULTS / "figures"
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"

GROUPS = {  # input group -> the model's columns
    "Hour of day": ["hour_sin", "hour_cos"],
    "Day of week": ["dow_sin", "dow_cos"],
    "Weekend": ["weekend"],
    "Used in the previous hour": ["lag_1h"],
    "Used at this hour yesterday": ["lag_24h"],
    "Used at this hour a week ago": ["lag_168h"],
    "Hours with a use, last 24 h": ["uses_24h"],
    "Hours with a use, last 7 days": ["uses_7d"],
    "Hours since the last use": ["hours_since_use"],
    "Share of the last 4 weeks used at this hour": ["recent_how_rate"],
    "Rest of the house active, last hour": ["others_1h"],
    "Rest of the house active, last 3 hours": ["others_3h"],
    "Previous hour had data": ["prev_hour_covered"],
}


def grouped(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum each group's columns (signed), per hour: one contribution per input group."""
    frame = pd.DataFrame(values, columns=columns)
    return pd.DataFrame({g: frame[cols].sum(axis=1) for g, cols in GROUPS.items()})


def explain_task(t, model_name: str = "random_forest", seed: int = 0):
    X = build_features(t.labels, t.other_starts)
    if t.train.nunique() < 2:
        return None
    model = make_model(model_name).fit(X.loc[t.train.index], t.train)
    test = X.loc[t.test.index]
    if len(test) > MAX_HOURS:
        test = test.sample(MAX_HOURS, random_state=seed)
    values = shap.TreeExplainer(model).shap_values(test)
    values = values[:, :, 1] if values.ndim == 3 else values  # contributions to P(switched on)
    contributions = grouped(values, list(X.columns))
    impurity = pd.Series(model.feature_importances_, index=X.columns)
    impurity = pd.Series({g: impurity[cols].sum() for g, cols in GROUPS.items()})
    return model, test, values, contributions, impurity


def importance_figure(summary: pd.DataFrame, n_tasks: int, path: Path, unit: str = "tasks") -> None:
    order = summary.sort_values("shap_share")
    fig, ax = plt.subplots(figsize=(7.2, 0.4 + 0.32 * len(order)), dpi=200)
    ax.barh(order.index, order["shap_share"] * 100, color=BLUE, height=0.62)
    for y, v in enumerate(order["shap_share"] * 100):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel(f"Share of the prediction explained, mean over {n_tasks} {unit} (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="both", labelsize=8.5, colors=INK, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.xaxis.grid(True, color="#e8e7e2", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def example_figure(test: pd.DataFrame, values: np.ndarray, title: str, path: Path) -> None:
    contributions = grouped(values, list(test.columns))
    # Feature values for colouring: the raw column for single inputs, the hour / weekday for pairs.
    shown = pd.DataFrame({g: test[cols[0]].to_numpy() for g, cols in GROUPS.items()})
    shown["Hour of day"] = test.index.hour
    shown["Day of week"] = test.index.dayofweek
    plt.figure(figsize=(7.2, 4.8), dpi=200)
    shap.summary_plot(contributions.to_numpy(), shown, feature_names=list(GROUPS), max_display=10,
                      show=False, plot_size=None)
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("SHAP value: pushes the prediction towards (+) or away from (-) a switch-on", fontsize=8.5)
    for text in plt.gcf().axes[-1].get_yticklabels() + [plt.gcf().axes[-1].yaxis.label]:
        text.set_fontsize(8)  # the colour bar
    plt.gcf().set_size_inches(8, 4.8)
    plt.gcf().tight_layout()
    plt.savefig(path, facecolor="white")
    plt.close()


def main(max_tasks: int | None = None, model_name: str = "random_forest") -> None:
    uses, covered, appliances = load_prepared()
    rows, best = [], None
    for i, t in enumerate(tasks(uses, covered, appliances)):
        if max_tasks is not None and i >= max_tasks:  # a quick check before the real run on Kaggle
            break
        result = explain_task(t, model_name)
        if result is None:
            continue
        model, test, values, contributions, impurity = result
        magnitude = contributions.abs().mean()
        share = magnitude / magnitude.sum() if magnitude.sum() > 0 else magnitude
        for g in GROUPS:
            rows.append({"house": t.house, "appliance": t.appliance, "type": appliance_type(t.appliance),
                         "input": g, "mean_abs_shap": magnitude[g], "shap_share": share[g],
                         "impurity": impurity[g]})
        # The example: the kettle with the most uses (a clear daily habit), else the busiest task.
        score = (appliance_type(t.appliance) == "kettle", int(t.uses.shape[0]))
        if best is None or score > best[0]:
            best = (score, t, test, values)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / f"habit_shap_{model_name}.csv", index=False)

    n_tasks = table.groupby(["house", "appliance"]).ngroups
    summary = table.groupby("input")[["shap_share", "impurity"]].mean().sort_values("shap_share", ascending=False)
    summary["shap_rank"] = range(1, len(summary) + 1)
    summary["impurity_rank"] = summary["impurity"].rank(ascending=False).astype(int)
    print(f"SHAP for the habit models ({model_name}), {DATASET}: {n_tasks} tasks, up to {MAX_HOURS} test hours each\n")
    print((summary.assign(shap_share=lambda d: (d.shap_share * 100).round(1),
                          impurity=lambda d: (d.impurity * 100).round(1))).to_string())
    by_type = table.pivot_table(index="type", columns="input", values="shap_share", aggfunc="mean")
    top = by_type.apply(lambda r: ", ".join(f"{k} {v:.0%}" for k, v in r.nlargest(3).items()), axis=1)
    print("\nTop three inputs per appliance type:")
    print(top.to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    importance_figure(summary, n_tasks, FIGURES / f"xai-habit-importance-{DATASET}-{model_name}.png")
    _, t, test, values = best
    example_figure(test, values, f"{t.appliance}, house {t.house}: {len(test)} test hours",
                   FIGURES / f"xai-habit-example-{DATASET}-{model_name}.png")
    print(f"\nFigures: {FIGURES}/xai-habit-importance-{DATASET}-{model_name}.png, xai-habit-example-{DATASET}-{model_name}.png "
          f"(example: {t.appliance}, house {t.house})")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[args.index("--max-tasks") + 1]) if "--max-tasks" in args else None,
         args[args.index("--model") + 1] if "--model" in args else "random_forest")
