"""Weekly habit finding: inputs only from earlier weeks, the two-threshold decision, the threshold choice."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from habits.weekly import Home, SLOTS, best_threshold, decide, features, past3


def toy_home(seed: int = 0, weeks: int = 8) -> Home:
    rng = np.random.default_rng(seed)
    days = [date(2014, 1, 6) + timedelta(days=i) for i in range(7 * weeks)]  # starts on a Monday
    mondays = pd.DatetimeIndex([pd.Timestamp(d) - pd.Timedelta(days=d.weekday()) for d in days])
    week_list = pd.date_range(mondays.min(), mondays.max(), freq="7D")
    kettle = rng.random((len(days), SLOTS)) < 0.05
    kettle[:, 14] = rng.random(len(days)) < 0.8  # a 07:00 habit
    toaster = rng.random((len(days), SLOTS)) < 0.03
    return Home(1, days, week_list, ((mondays - week_list[0]).days // 7).to_numpy(),
                np.array([1 if d.weekday() >= 5 else 0 for d in days]),
                {"A1": kettle, "A2": toaster}, {"A1": "kettle", "A2": "toaster"}, ["A1"])


def test_inputs_never_see_the_week_being_predicted():
    home = toy_home()
    before = features(home, ["kettle", "toaster"])
    week = home.weeks[5]
    in_week = home.day_week == 5
    home.starts["A1"][in_week] = ~home.starts["A1"][in_week]  # change the predicted week completely
    home.starts["A2"][in_week] = ~home.starts["A2"][in_week]
    after = features(home, ["kettle", "toaster"])
    inputs = [c for c in before.columns if c not in ("label", "week", "house", "column", "type")]
    a = before[before["week"] == week][inputs].reset_index(drop=True)
    b = after[after["week"] == week][inputs].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # same inputs...
    assert not np.allclose(before[before["week"] == week]["label"], after[after["week"] == week]["label"])  # ...new label


def test_the_habit_slot_is_learned_from_the_last_three_weeks():
    f = features(toy_home(), ["kettle", "toaster"])
    seven = f[(f["slot"] == 14) & (f["kind"] == 0)]
    assert seven["own_3w"].mean() > 0.6 and f[f["slot"] == 30]["own_3w"].mean() < 0.15
    assert "c1_toaster" in f and "c3w_toaster" in f and "c1_kettle" in f  # the other appliance, and no self


def test_past3_sums_the_three_weeks_before():
    x = np.arange(6, dtype=float).reshape(6, 1)
    assert past3(x)[:, 0].tolist() == [0, 0, 1, 3, 6, 9]


def test_new_at_the_threshold_kept_above_the_lower_one():
    previous = np.array([True, True, False, False])
    predicted = np.array([0.35, 0.2, 0.35, 0.55])
    assert decide(predicted, previous).tolist() == [True, False, False, True]  # keep >= 0.3, new >= 0.5
    assert decide(predicted, previous, suggest=0.3).tolist() == [True, True, True, True]  # keep >= 0.18


def test_best_threshold_matches_the_predictions_scale():
    label = np.array([0.0] * 90 + [0.8] * 10)
    predicted = np.r_[np.full(90, 0.05), np.full(10, 0.25)]  # the model's scale: low, but ordered
    assert 0.05 < best_threshold(predicted, label) <= 0.25


def test_linear_and_mlp_inputs_fill_gaps_flag_them_and_one_hot_the_type():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_weekly_models import Prepared

    frame = pd.DataFrame({"own_3w": [0.5, np.nan, 0.1, 0.3], "c1_toaster": [np.nan, np.nan, 0.2, 0.4],
                          "type_code": [0, 1, 0, 1], "label": [0.4, 0.0, 0.1, 0.3]})
    cols = ["own_3w", "c1_toaster", "type_code"]
    prep = Prepared("linear", cols, frame, np.array([0, 1, 2]))  # scaled on the first three rows only
    X = prep.X(frame)
    assert X.shape == (4, 2 + 2 + 2)  # two inputs, two "was empty" flags, two type columns
    assert not np.isnan(X).any()
    trees = Prepared("lightgbm", cols, frame, np.array([0, 1, 2])).X(frame)
    assert list(trees.columns) == cols and trees["c1_toaster"].isna().sum() == 2  # trees keep the gaps
