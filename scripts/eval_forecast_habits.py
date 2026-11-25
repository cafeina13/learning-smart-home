"""Can tree ensembles and the habit models forecast a home's next-hour load better than "same as last hour"?

    .venv\\Scripts\\python.exe scripts\\eval_forecast_habits.py            # REFIT, 20 homes
    $env:CAPSTONE_DATASET = "ukdale"; .venv\\Scripts\\python.exe scripts\\eval_forecast_habits.py
    python scripts/eval_forecast_habits.py --deep        # her networks, same inputs (Kaggle GPU)

scripts/eval_forecast.py replicated Yaprakdal (2023, 2024): twelve forecasters,
none significantly better than persistence at single-home level. Two things were
not tried there, and this script tries both on exactly the same test hours (the
time protocol: each home's last 20% of usable hours, trained on the 80% before):

model type   tree ensembles, the kind that won everywhere else in this project:
             a random forest and gradient boosting (scikit-learn's histogram
             version, the same algorithm family as LightGBM), trained on the
             absolute error (the score) and, to see what that choice is worth,
             on the squared error as her models were
information  what the home itself knows and a regional forecaster does not:

    load       past load (1, 2, 3, 24, 48 hours ago; means and spread over the
               last 3, 6, 24 and 144 hours), hour, weekday, month, weather the
               hour before: the same information her models had
    +habits    for every appliance with a habit model: the habit forest's
               probability that it is switched on in the target hour (its 15
               past-only features, habits/features.py), and that times the
               appliance's median energy per use; plus their sum, the expected
               appliance energy for the hour
    +devices   also each appliance's share of the previous hour it was running
               (the smart plugs report it)

Every input for hour h uses only what was known before h. The habit forests and
the load models are trained on the 80% before the test hours, as the paper's
models were. Compared per home with persistence and with her best forecasters on
the same hours (data/results/forecast_deep.csv, time protocol, selected
features); one-sided Wilcoxon over homes.

--deep gives her networks the same chances (fairness: her method is deep
learning): the GRU and CNN-GRU (2024) and the BiLSTM RNN2 (2023), each trained
with her loss (squared error) and with the absolute error, on her selected
features (data/results/forecast_features.csv, as in her replication) and on
those plus the home's information as extra hourly channels of the 144-hour
window (at each past hour t: the habit forests' forecast for hour t+1 and its
expected energy, and the share of hour t each appliance ran). Same homes, same
test hours. Writes data/results/forecast_habits_deep.csv.

Writes data/results/forecast_habits.csv (one row per home x input set x model).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402

from energy.data import FEATURES, house_hours, weather, windows  # noqa: E402
from energy.forecasters import fit_predict, mae, mape  # noqa: E402
from habits.dataset import hourly_labels  # noqa: E402
from habits.evaluate import MIN_USES  # noqa: E402
from habits.features import build_features, other_appliance_starts  # noqa: E402
from habits.models import make_model  # noqa: E402
from habits.paths import RESULTS  # noqa: E402
from habits.prepare import load_aggregate, load_prepared  # noqa: E402

OUT = RESULTS / "forecast_habits.csv"
OUT_DEEP = RESULTS / "forecast_habits_deep.csv"
DEEP_MODELS = ["GRU", "CNN-GRU", "RNN2"]
LOSSES = ["squared", "absolute"]
INPUTS = ["load", "load+habits", "load+habits+devices"]
HER_BEST = ["ANN1&2", "GRU", "Hourly ensemble"]  # her strongest on the time protocol (scripts/eval_forecast.py)


def load_features(X: np.ndarray, when: pd.DatetimeIndex) -> pd.DataFrame:
    """Compact past-load, calendar and weather inputs from each 144-hour window (FEATURES order)."""
    load = X[:, :, FEATURES.index("load")]
    f = {f"lag_{k}h": load[:, -k] for k in (1, 2, 3, 24, 48)}
    for k in (3, 6, 24, 144):
        f[f"mean_{k}h"] = load[:, -k:].mean(axis=1)
    f["std_24h"] = load[:, -24:].std(axis=1)
    f["max_24h"] = load[:, -24:].max(axis=1)
    f["hour"], f["weekday"], f["month"] = when.hour, when.dayofweek, when.month
    for w in ("temperature", "irradiance", "wind"):
        f[f"{w}_1h"] = X[:, -1, FEATURES.index(w)]
    return pd.DataFrame(f, index=when)


def running_share(u: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.Series:
    """Share of each hour in which the appliance was running."""
    share = pd.Series(0.0, index=hours)
    for start, end in zip(u["start"], u["end"]):
        t = start
        while t < end:
            h = t.floor("h")
            step = min(h + pd.Timedelta(hours=1), end) - t
            if h in share.index:
                share[h] += step.total_seconds() / 3600
            t += step
    return share.clip(upper=1.0)


def habit_features(house: int, uses, covered, appliances, when: pd.DatetimeIndex, cutoff: pd.Timestamp):
    """(+habits columns, +devices columns) for the target hours, the habit forests trained before `cutoff`."""
    house_uses = uses[uses["house"] == house].merge(appliances[["house", "column", "category"]],
                                                    on=["house", "column"], how="left")
    hours = pd.DatetimeIndex(covered.loc[covered["house"] == house, "hour"]).sort_values()
    habits, devices = {}, {}
    rows = appliances[(appliances["house"] == house) & appliances["category"].isin(["habit", "shiftable"])
                      & (appliances["uses"] >= MIN_USES)]
    for a in rows.itertuples():
        u = house_uses[house_uses["column"] == a.column]
        labels = hourly_labels(u, hours)
        X = build_features(labels, other_appliance_starts(house_uses, a.column))
        train = labels.index < cutoff
        if labels[train].nunique() < 2:
            continue
        forest = make_model("random_forest").fit(X[train], labels[train])
        p = pd.Series(forest.predict_proba(X)[:, 1], index=labels.index).reindex(when).fillna(0.0)
        wh = float(u.loc[u["start"] < cutoff, "energy_wh"].median())
        habits[f"p_{a.column}"] = p.to_numpy()
        habits[f"wh_{a.column}"] = (p * wh).to_numpy()
        grid = pd.date_range(when.min() - pd.Timedelta(hours=1), when.max(), freq="h")
        devices[f"on_{a.column}"] = running_share(u, grid).shift(1).reindex(when).fillna(0.0).to_numpy()
    h = pd.DataFrame(habits, index=when)
    h["expected_wh"] = h.filter(like="wh_").sum(axis=1) if len(habits) else 0.0
    return h, pd.DataFrame(devices, index=when)


MODELS = ["random_forest", "gradient_boosting", "gradient_boosting_squared"]


def make(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=0)
    loss = "squared_error" if name == "gradient_boosting_squared" else "absolute_error"
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=20, early_stopping=False, random_state=0)


def evaluate_house(house: int, aggregate, wx, prepared) -> list[dict]:
    X, y, when = windows(house_hours(house, aggregate, wx), FEATURES)
    n = len(X)
    b = int(n * 0.8)  # as eval_forecast.py's time protocol: train on the first 80%, test on the rest
    trva, te = np.arange(b), np.arange(b, n)
    base = load_features(X, when)
    habits, devices = habit_features(house, *prepared, when, cutoff=when[b])
    inputs = {"load": base, "load+habits": base.join(habits),
              "load+habits+devices": base.join(habits).join(devices)}
    rows = [{"house": house, "inputs": "-", "model": "Persistence (last hour)",
             "mae_w": mae(y[te], X[te, -1, 0]), "mape": mape(y[te], X[te, -1, 0]), "appliances": habits.shape[1] // 2}]
    for inp in INPUTS:
        F = inputs[inp].to_numpy(dtype="float32")
        for name in MODELS:
            p = make(name).fit(F[trva], y[trva]).predict(F[te])
            rows.append({"house": house, "inputs": inp, "model": name, "mae_w": mae(y[te], p),
                         "mape": mape(y[te], p), "appliances": habits.shape[1] // 2})
    return rows


def selected_features(house: int) -> list[str]:
    """Her feature selection for this home, time protocol (scripts/eval_forecast.py)."""
    chosen = pd.read_csv(RESULTS / "forecast_features.csv")
    row = chosen[(chosen["house"] == house) & (chosen["protocol"] == "time")]
    return row["selected"].iloc[0].split(",")


def evaluate_house_deep(house: int, aggregate, wx, prepared) -> list[dict]:
    df = house_hours(house, aggregate, wx)
    _, _, target_hours = windows(df, FEATURES)  # the replication's target hours
    n = len(target_hours)
    b = int(n * 0.8)
    # Channel values at hour t: what is known at the end of t about hour t+1.
    habits, devices = habit_features(house, *prepared, df.index + pd.Timedelta(hours=1), cutoff=target_hours[b])
    channels = habits.join(devices).set_axis(df.index)
    channels.columns = [f"x_{c}" for c in channels.columns]
    names = FEATURES + list(channels.columns)
    X, y, when = windows(df.join(channels), names)
    assert when.equals(target_hours), "the habit channels changed the usable hours"
    hours, trva, te = when.hour.to_numpy(), np.arange(b), np.arange(b, n)
    sel = selected_features(house)
    rows = [{"house": house, "inputs": "-", "model": "Persistence (last hour)", "loss": "-",
             "mae_w": mae(y[te], X[te, -1, 0]), "mape": mape(y[te], X[te, -1, 0])}]
    for inp, feats in (("selected", sel), ("selected+habits+devices", sel + list(channels.columns))):
        Xs = X[:, :, [names.index(f) for f in feats]]
        for name in DEEP_MODELS:
            for loss in LOSSES:
                p = fit_predict(name, Xs[trva], y[trva], hours[trva], Xs[te], hours[te], feats, loss=loss)
                rows.append({"house": house, "inputs": inp, "model": name, "loss": loss,
                             "mae_w": mae(y[te], p), "mape": mape(y[te], p)})
    return rows


def summary_deep() -> None:
    df = pd.read_csv(OUT_DEEP)
    df["label"] = np.where(df["inputs"] == "-", df["model"], df["model"] + " " + df["loss"] + " | " + df["inputs"])
    wide = df.pivot_table(index="house", columns="label", values="mae_w")
    pers = wide["Persistence (last hour)"]
    if OUT.exists():  # the tree ensembles on the same hours
        gb = pd.read_csv(OUT)
        gb = gb[(gb["model"] == "gradient_boosting") & (gb["inputs"] == "load+habits+devices")]
        wide["gradient_boosting | load+habits+devices"] = gb.set_index("house")["mae_w"]
    print(f"\n{len(wide)} homes; her networks with her loss and inputs, then with the same chances as the trees:\n")
    print(f"  {'model loss | inputs':<52} {'MAE':>7}  {'better in':>9}  vs persistence")
    for col in wide.median().sort_values().index:
        d = wide[col] - pers
        p = wilcoxon(wide[col], pers, alternative="less").pvalue if col != pers.name and d.any() else float("nan")
        print(f"  {col:<52} {wide[col].median():7.1f}  {(d < 0).sum():>4} / {len(d):<3}  "
              + ("" if np.isnan(p) else f"p = {p:.1e}"))
    if "gradient_boosting | load+habits+devices" in wide:
        g = wide["gradient_boosting | load+habits+devices"]
        best = wide.drop(columns=[pers.name, g.name]).median().idxmin()
        d = wide[best] - g
        print(f"\n  best network ({best}) vs gradient boosting: better in {(d < 0).sum()} of {len(d)} homes "
              f"(two-sided Wilcoxon p = {wilcoxon(wide[best], g).pvalue:.1e})")
    for m in DEEP_MODELS:
        for loss in LOSSES:
            a, b = f"{m} {loss} | selected", f"{m} {loss} | selected+habits+devices"
            if a in wide and b in wide:
                d = wide[b] - wide[a]
                print(f"  {m} {loss}: + habits + devices lower in {(d < 0).sum()} of {len(d)} homes, "
                      f"median {d.median():+.1f} W (p = {wilcoxon(wide[b], wide[a], alternative='less').pvalue:.1e})")


def summary() -> None:
    df = pd.read_csv(OUT)
    df["label"] = np.where(df["inputs"] == "-", df["model"], df["model"] + " | " + df["inputs"])
    deep = RESULTS / "forecast_deep.csv"
    if deep.exists():
        her = pd.read_csv(deep)
        her = her[(her["protocol"] == "time") & her["model"].isin(HER_BEST)]
        her = her.assign(label="her " + her["model"])[["house", "label", "mae_w"]]
        df = pd.concat([df[["house", "label", "mae_w"]], her[her["house"].isin(df["house"])]], ignore_index=True)
    wide = df.pivot_table(index="house", columns="label", values="mae_w")
    pers = wide["Persistence (last hour)"]
    print(f"\n{len(wide)} homes, next-hour load on the last 20% of hours; median MAE (W), and against persistence:\n")
    print(f"  {'model | inputs':<45} {'MAE':>7}  {'better in':>9}  one-sided Wilcoxon")
    for col in wide.median().sort_values().index:
        d = wide[col] - pers
        p = wilcoxon(wide[col], pers, alternative="less").pvalue if col != pers.name and d.any() else float("nan")
        print(f"  {col:<45} {wide[col].median():7.1f}  {(d < 0).sum():>4} / {len(d):<3}  "
              + ("" if np.isnan(p) else f"p = {p:.1e}"))
    best = wide.drop(columns=[pers.name]).median().idxmin()
    for her_col in [c for c in wide.columns if c.startswith("her ")]:
        d = wide[best] - wide[her_col]
        p = wilcoxon(wide[best], wide[her_col], alternative="less").pvalue
        print(f"\n  {best} vs {her_col}: better in {(d < 0).sum()} of {len(d)} homes (p = {p:.1e})", end="")
    print()


def main(deep: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    aggregate, wx, prepared = load_aggregate(), weather(), load_prepared()
    out = OUT_DEEP if deep else OUT
    done = set(pd.read_csv(out)["house"]) if out.exists() else set()
    for house in sorted(aggregate["house"].unique()):
        if house in done:
            continue
        t = time.perf_counter()
        rows = (evaluate_house_deep if deep else evaluate_house)(house, aggregate, wx, prepared)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"house {house:>2}: {time.perf_counter() - t:.0f} s", flush=True)
    summary_deep() if deep else summary()


if __name__ == "__main__":
    main(deep="--deep" in sys.argv)
