"""The comparison article's method: Gad, Gad, Abdelkader & Naik (2025), Sensors 25(19):6082.

Per home, a gradient-boosted classifier predicts the activity of each sensor
interval (habits/casas.py) from the 30 most recent intervals and a 21-day
history of what the person usually does at this time of day.

Replicated as the paper describes it:

step features   per interval: room; time of day (sine, cosine); day of week;
                hourly activity count (labelled intervals in the same clock hour
                of the same date); room entropy (once per home); 3-hour rolling
                mean of the encoded activity; 6-hour rolling mean of the encoded room
configs         1: the first four, 7-day history; 2: + hourly count and room
                entropy, 21-day history; 3: all eight, 21-day history (Table 6)
history         for each of the past D days, the most frequent activity within
                +-1 h of the same time of day; weighted by w_d = 1.8 if d is a
                whole number of weeks, else max(0.2, 0.8 - 0.1 d); summarised as
                the activity with the largest weighted vote and its weight
sample          one per labelled interval with 30 intervals before it; Other_Activity,
                Entertain_Guests and Enter_Home removed
cleaning        if the largest class is over 10x the smallest, every class is cut
                to at most 5x the smallest (random intervals kept)
split           80/10/10, stratified by activity
training        2% of the training labels replaced by a random other activity;
                Table 3's settings; early stopping after 100 rounds without
                improvement on the validation log-loss
metrics         accuracy and weighted F1 on the test intervals

Checked against the paper's text (Sections 3.5-3.7, 2026-10-14). Followed as
written: the flattened 30 x 8 window (here with the history summary of the
current interval appended); the history weights, whose four cases reduce to
the two below because a whole number of weeks back always falls on the same
weekday. Read differently, and stated in the report:

- the paper stratifies when every class has >= 20 samples and otherwise splits
  at random; here activities with fewer than MIN_CLASS samples are dropped, so
  the split is always stratified and "5x the smallest class" is never 5x one;
- the paper's 3-hour activity mean covers t - 3 h <= tau <= t, which taken
  literally includes the activity being predicted; here it ends just before
  the current interval (stricter, and may cost some accuracy);
- the +-1 h window is read from 5-minute bins; day of week is 0-6.

Three variants separate where the score comes from:

paper     the paper's setup above
time      the same samples, split by time (first 80% train, next 10% validation,
          last 10% test), as a home would use it
strict    also split by time, and only what a home knows at the moment: the
          hourly count looks back one hour instead of covering the whole clock
          hour (which includes later intervals), the 3-hour activity mean is
          dropped (it needs the true activities of the last hours), and the
          class cut is applied to the training part only, so the test intervals
          keep their real mix
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from habits.casas import NOISY, room_entropy

# LightGBM names numpy columns itself, and scikit-learn then warns on every prediction; harmless
warnings.filterwarnings("ignore", message="X does not have valid feature names")

WINDOW = 30
MIN_CLASS = 20
LABEL_NOISE = 0.02
BIN = pd.Timedelta(minutes=5)
VARIANTS = ["paper", "time", "strict"]
MODELS = ["lightgbm", "xgboost", "catboost"]
STEP = ["room", "tod_sin", "tod_cos", "dow", "hourly_count", "room_entropy", "act_3h", "room_6h"]
CONFIGS = {1: (STEP[:4], 7), 2: (STEP[:6], 21), 3: (STEP, 21)}


def day_weights(days: int) -> np.ndarray:
    d = np.arange(1, days + 1)
    return np.where(d % 7 == 0, 1.8, np.maximum(0.2, 0.8 - 0.1 * d))


def step_features(iv: pd.DataFrame, codes: np.ndarray, strict: bool) -> pd.DataFrame:
    """The eight per-interval features. codes: activity code per interval, NaN if unlabelled."""
    t = pd.DatetimeIndex(iv["start"])
    rooms = {r: i for i, r in enumerate(sorted(iv["room"].unique()))}
    room = iv["room"].map(rooms).to_numpy(float)
    seconds = (t - t.normalize()).total_seconds().to_numpy()
    f = pd.DataFrame({"room": room,
                      "tod_sin": np.sin(2 * np.pi * seconds / 86400), "tod_cos": np.cos(2 * np.pi * seconds / 86400),
                      "dow": t.dayofweek.to_numpy(float)})
    labelled = pd.Series(~np.isnan(codes), index=t).astype(float)
    if strict:  # labelled intervals in the hour before, not the whole clock hour
        f["hourly_count"] = labelled.rolling("1h", closed="left").sum().fillna(0).to_numpy()
    else:
        f["hourly_count"] = labelled.groupby(t.floor("h")).transform("sum").to_numpy()
    f["room_entropy"] = room_entropy(iv["room"])
    f["act_3h"] = pd.Series(codes, index=t).rolling("3h", closed="left", min_periods=1).mean().to_numpy()
    f["room_6h"] = pd.Series(room, index=t).rolling("6h", closed="both", min_periods=1).mean().to_numpy()
    return f


def history(start: pd.Series, codes: np.ndarray, n_classes: int, days: int) -> np.ndarray:
    """(n, days + 2): the most frequent activity +-1 h around the same time d days before
    (-1 if none), then the activity with the largest weighted vote and its weight."""
    base = start.min().normalize()
    b = ((start - base) // BIN).to_numpy()
    n_bins = int(b.max()) + 1
    counts = np.zeros((n_bins + 1, n_classes), np.int32)
    ok = ~np.isnan(codes)
    np.add.at(counts, (b[ok] + 1, codes[ok].astype(int)), 1)
    cum = counts.cumsum(axis=0)  # cum[k] = labelled intervals in bins < k
    per_day, half = pd.Timedelta(days=1) // BIN, pd.Timedelta(hours=1) // BIN
    w = day_weights(days)
    out = np.full((len(b), days), -1.0)
    vote = np.zeros((len(b), n_classes))
    for d in range(1, days + 1):
        lo = np.clip(b - d * per_day - half, 0, n_bins)
        hi = np.clip(b - d * per_day + half + 1, 0, n_bins)
        window = cum[hi] - cum[lo]
        has = window.sum(axis=1) > 0
        top = window.argmax(axis=1)
        out[has, d - 1] = top[has]
        vote[np.flatnonzero(has), top[has]] += w[d - 1]
    best = np.where(vote.max(axis=1) > 0, vote.argmax(axis=1), -1)
    return np.column_stack([out, best, vote.max(axis=1)])


def samples(iv: pd.DataFrame, config: int, strict: bool = False):
    """X, activity names y, times, the history vote's guess and the previous labelled interval's
    activity (for the baselines), for every labelled interval with WINDOW intervals before it."""
    iv = iv.reset_index(drop=True)
    activity = iv["activity"].where(~iv["activity"].isin(NOISY))
    classes = sorted(activity.dropna().unique())
    codes = activity.map({c: i for i, c in enumerate(classes)}).to_numpy(float)
    names, days = CONFIGS[config]
    if strict:
        names = [n for n in names if n != "act_3h"]
    steps = step_features(iv, codes, strict)[names].to_numpy(np.float32)
    hist = history(iv["start"], codes, len(classes), days)
    rows = np.flatnonzero(~np.isnan(codes))
    rows = rows[rows >= WINDOW - 1]
    X = np.hstack([steps[rows[:, None] + np.arange(-WINDOW + 1, 1)].reshape(len(rows), -1), hist[rows]])
    y = activity.to_numpy(object)[rows]
    previous = activity.ffill().shift(1).to_numpy(object)[rows]
    guess = np.array([classes[int(k)] if k >= 0 else None for k in hist[rows, -2]], dtype=object)
    keep = pd.Series(y).map(pd.Series(y).value_counts()).to_numpy() >= MIN_CLASS
    return X[keep], y[keep], iv["start"].to_numpy()[rows][keep], guess[keep], previous[keep]


def undersample(y: np.ndarray, rng) -> np.ndarray:
    """Indices kept: if the largest class is over 10x the smallest, at most 5x the smallest per class."""
    counts = pd.Series(y).value_counts()
    if counts.max() <= 10 * counts.min():
        return np.arange(len(y))
    cap = 5 * counts.min()
    keep = [rng.choice(np.flatnonzero(y == c), size=min(n, cap), replace=False) for c, n in counts.items()]
    return np.sort(np.concatenate(keep))


def split(y: np.ndarray, variant: str, seed: int):
    """Train / validation / test indices: 80/10/10, stratified at random ('paper') or by time."""
    idx = np.arange(len(y))
    if variant == "paper":
        strat = y if pd.Series(y).value_counts().min() >= MIN_CLASS else None
        train, rest = train_test_split(idx, test_size=0.2, stratify=strat, random_state=seed)
        strat = y[rest] if strat is not None and pd.Series(y[rest]).value_counts().min() >= 2 else None
        val, test = train_test_split(rest, test_size=0.5, stratify=strat, random_state=seed)
        return train, val, test
    a, b = int(len(y) * 0.8), int(len(y) * 0.9)  # samples are in time order
    return idx[:a], idx[a:b], idx[b:]


def add_label_noise(y: np.ndarray, rng, share: float = LABEL_NOISE) -> np.ndarray:
    y = y.copy()
    classes = np.unique(y)
    if len(classes) < 2:
        return y
    for i in rng.choice(len(y), size=int(share * len(y)), replace=False):
        y[i] = rng.choice(classes[classes != y[i]])
    return y


def make_model(name: str, binary: bool = False, seed: int = 0):
    """Table 3 of the paper."""
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(learning_rate=0.01, num_leaves=15, n_estimators=1000, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.1, min_child_samples=50,
                              random_state=seed, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=2.0, reg_lambda=2.0, min_child_weight=5, early_stopping_rounds=100,
                             tree_method="hist", random_state=seed)
    if name == "catboost":
        from catboost import CatBoostClassifier
        # Table 3's min_data_in_leaf=10 is left out: CatBoost accepts it only for non-symmetric
        # trees, and the paper gives depth 6 with CatBoost's default (symmetric) trees
        return CatBoostClassifier(learning_rate=0.05, depth=6, iterations=1000, subsample=0.8, rsm=0.8,
                                  bootstrap_type="Bernoulli", l2_leaf_reg=2.0,
                                  loss_function="Logloss" if binary else "MultiClass",
                                  random_seed=seed, verbose=False, thread_count=-1)
    raise ValueError(name)


def fit(name: str, X_train, y_train, X_val, y_val, binary: bool = False, seed: int = 0):
    """Train with early stopping on the validation part (100 rounds); returns the model and the
    number of boosting rounds it kept."""
    model = make_model(name, binary, seed)
    if name == "lightgbm":
        from lightgbm import early_stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100, verbose=False)])
        return model, int(model.best_iteration_ or model.n_estimators)
    if name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model, int(model.best_iteration + 1)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    return model, int(model.get_best_iteration() + 1)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def run_home(iv: pd.DataFrame, config: int, variant: str, models, seed: int = 0) -> list[dict]:
    """Every model and baseline on one home, one config, one variant."""
    rng = np.random.default_rng(seed)
    X, y, _, guess, previous = samples(iv, config, strict=(variant == "strict"))
    n_clean = len(y)
    if variant != "strict":  # the paper cuts the classes before splitting
        keep = undersample(y, rng)
        X, y, guess, previous = X[keep], y[keep], guess[keep], previous[keep]
    train, val, test = split(y, variant, seed)
    if variant == "strict":  # only the training part is cut; validation and test keep the real mix
        train = train[undersample(y[train], rng)]
    classes = np.unique(y[train])
    code = {c: i for i, c in enumerate(classes)}
    y_train = np.array([code[c] for c in add_label_noise(y[train], rng)])
    seen_val = np.isin(y[val], classes)  # early stopping can only score activities seen in training
    y_val = np.array([code[c] for c in y[val][seen_val]])
    majority = pd.Series(y[train]).mode()[0]
    common = {"classes": len(classes), "samples_clean": n_clean, "samples_used": len(y),
              "train": len(train), "val": len(val), "test": len(test),
              "unseen_test_share": round(float(1 - np.isin(y[test], classes).mean()), 4)}
    rows = []
    for name, pred in (("majority", np.full(len(test), majority, dtype=object)),
                       ("previous_activity", np.where(pd.isna(previous[test]), majority, previous[test])),
                       ("history_vote", np.where(pd.isna(guess[test]), majority, guess[test]))):
        rows.append({"model": name, **common, **scores(y[test], pred)})
    for name in models:
        model, rounds = fit(name, X[train], y_train, X[val][seen_val], y_val, seed=seed)
        pred = classes[np.asarray(model.predict(X[test])).astype(int).ravel()]
        rows.append({"model": name, **common, "rounds": rounds, **scores(y[test], pred)})
    return rows


# ---- The same method on our hourly question (REFIT, UK-DALE, CASAS) ------------------------------

def hourly_inputs(labels: pd.Series, other_starts: pd.Series | None) -> pd.DataFrame:
    """Their design on "will this be switched on in hour h?": the 30 hours before h, each with
    its calendar, the house's activity (other switch-ons, their "hourly activity count"), the
    appliance's own use, its 3-hour mean and the house's 6-hour mean; then hour h's calendar
    and the 21-day history at the same hour +-1 h with its weighted vote. All from before h."""
    grid = pd.date_range(labels.index.min(), labels.index.max(), freq="h")
    own = labels.reindex(grid, fill_value=0).astype(float)
    others = (other_starts.reindex(grid, fill_value=0).astype(float) if other_starts is not None
              else pd.Series(0.0, index=grid))
    hour, dow = grid.hour.to_numpy(), grid.dayofweek.to_numpy()
    step = np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), dow, others, own,
                            own.rolling(3, min_periods=1).mean(), others.rolling(6, min_periods=1).mean()])
    step = np.vstack([np.zeros((WINDOW, step.shape[1])), step])  # padding: rows 0..WINDOW-1
    pos = grid.get_indexer(labels.index) + WINDOW  # hour h's row; its window is rows [h-WINDOW, h)
    window = step[pos[:, None] + np.arange(-WINDOW, 0)].reshape(len(pos), -1)
    padded_own = np.concatenate([np.zeros(24 * 21 + 1), own.to_numpy()])
    p = grid.get_indexer(labels.index) + 24 * 21 + 1
    hist = np.column_stack([padded_own[np.clip(p + k - 24 * d, 0, None)] for d in range(1, 22)
                            for k in (-1, 0, 1)]).reshape(len(p), 21, 3).max(axis=2)
    vote = hist @ day_weights(21) / day_weights(21).sum()
    cal = np.column_stack([np.sin(2 * np.pi * labels.index.hour / 24), np.cos(2 * np.pi * labels.index.hour / 24),
                           labels.index.dayofweek])
    return pd.DataFrame(np.hstack([window, cal, hist, vote[:, None]]), index=labels.index)


def fit_and_predict_hourly(task, name: str = "lightgbm", seed: int = 0) -> tuple[np.ndarray, dict]:
    """Like habits.models.fit_and_predict: their model and inputs, our split and scoring
    (threshold chosen on the validation hours)."""
    from habits.models import best_threshold
    X = hourly_inputs(task.labels, task.other_starts)
    X_train, X_val, X_test = (X.loc[s.index].to_numpy() for s in (task.train, task.val, task.test))
    if task.train.nunique() < 2 or task.val.nunique() < 2:
        return np.zeros(len(X_test), dtype=int), {}
    rng = np.random.default_rng(seed)
    model, rounds = fit(name, X_train, add_label_noise(task.train.to_numpy(), rng), X_val, task.val.to_numpy(),
                        binary=True, seed=seed)
    threshold = best_threshold(task.val, model.predict_proba(X_val)[:, 1])
    return (model.predict_proba(X_test)[:, 1] >= threshold).astype(int), {"threshold": threshold, "rounds": rounds}
