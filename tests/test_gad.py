"""CASAS loading and Gad et al.'s method: labels, intervals, features from the past only, splits."""

import io

import numpy as np
import pandas as pd
import pytest

from habits import casas, gad
from habits.dataset import hourly_labels, time_split
from habits.evaluate import Task, random_split

ROUTINE = [("Bedroom", "Sleep", 0, 7), ("Kitchen", "Cook_Breakfast", 7, 8), ("LivingRoom", "Watch_TV", 19, 21),
           ("Office", None, 10, 12)]


def synthetic_csv(days: int = 30) -> str:
    """A CASAS-style file: a fixed daily routine, motion every 10 minutes lasting 3 s (plus a 1 s blip)."""
    lines = []
    for day in pd.date_range("2013-03-01", periods=days, freq="D"):
        for room, activity, h0, h1 in ROUTINE:
            times = pd.date_range(day + pd.Timedelta(hours=h0), day + pd.Timedelta(hours=h1), freq="10min")
            for k, t in enumerate(times):
                note = ""
                if activity and k == 0:
                    note = f',{activity}="begin"'
                elif activity and k == len(times) - 1:
                    note = f',{activity}="end"'
                lines.append(f"{t:%Y-%m-%d},{t:%H:%M:%S}.000000,{room},ON{note}")
                lines.append(f"{t + pd.Timedelta(seconds=3):%Y-%m-%d},{t + pd.Timedelta(seconds=3):%H:%M:%S}.5,{room},OFF")
                lines.append(f"{t + pd.Timedelta(seconds=5):%Y-%m-%d},{t + pd.Timedelta(seconds=5):%H:%M:%S},{room},ON")
                lines.append(f"{t + pd.Timedelta(seconds=6):%Y-%m-%d},{t + pd.Timedelta(seconds=6):%H:%M:%S},{room},OFF")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def home():
    events = casas.read_events(io.StringIO(synthetic_csv()))
    labels = casas.label_events(events)
    return events, labels, casas.intervals(events, labels)


def test_begin_end_label_the_events_between():
    events = casas.read_events(io.StringIO(
        "2013-03-01,08:00:00.1,Kitchen,ON,Cook=\"begin\"\n2013-03-01,08:00:05,Kitchen,OFF\n"
        "2013-03-01,08:01:00,Hall,ON,Phone\n2013-03-01,08:02:00,Kitchen,ON,Cook=\"end\"\n"
        "2013-03-01,08:03:00,Kitchen,OFF\n"))
    assert list(casas.label_events(events)) == ["Cook", "Cook", "Phone", "Cook", None]


def test_stray_commas_and_double_annotations_are_read():  # hh111 has lines ending in ","
    events = casas.read_events(io.StringIO(
        '2011-07-22,06:20:00.1,Bedroom,ON\n2011-07-22,06:22:03.9,Bedroom,OFF,Sleep="end",\n'
        '2011-07-22,06:23:00,Bathroom,ON,Toilet="begin",Groom\n\n2011-07-22,06:24:00,Bathroom,OFF\n'))
    assert list(events["note"]) == ["", 'Sleep="end"', 'Toilet="begin",Groom', ""]
    assert list(casas.label_events(events)) == [None, "Sleep", "Groom", "Toilet"]


def test_intervals_drop_blips_and_take_the_label_at_their_start(home):
    _, _, iv = home
    assert (iv["end"] - iv["start"]).dt.total_seconds().min() >= casas.MIN_INTERVAL_S  # the 1 s blips are gone
    sleep = iv[(iv["start"].dt.hour < 7) & (iv["room"] == "Bedroom")]
    assert (sleep["activity"] == "Sleep").all()
    assert iv.loc[iv["room"] == "Office", "activity"].isna().all()


def test_tables_make_activities_and_sensors_for_our_pipeline(home):
    events, labels, _ = home
    uses, covered, rows = casas.tables("hh101", events, labels)
    kinds = {r["column"]: r["category"] for r in rows}
    assert kinds == {"Sleep": "habit", "Cook_Breakfast": "habit", "Watch_TV": "habit", "sensors": "sensor"}
    starts = uses[uses["column"] == "Watch_TV"]["start"]
    assert len(starts) == 30 and (starts.dt.hour == 19).all()  # one start a day, at 19:00
    assert len(covered) == 30 * 24 and covered["house"].eq(101).all()


def test_history_only_looks_at_earlier_days(home):
    _, _, iv = home
    codes = iv["activity"].map({"Cook_Breakfast": 0, "Sleep": 1, "Watch_TV": 2}).to_numpy(float)
    hist = gad.history(iv["start"], codes, 3, days=7)
    first_day = (iv["start"] < pd.Timestamp("2013-03-02")).to_numpy()
    assert (hist[first_day, :7] == -1).all()  # nothing before the first day
    tv = ((iv["start"] >= pd.Timestamp("2013-03-10")) & (iv["start"].dt.hour == 20)).to_numpy()
    assert (hist[tv, :7] == 2).all() and (hist[tv, 7] == 2).all()  # every past day: TV at this time


def test_samples_use_30_intervals_up_to_the_current_one(home):
    _, _, iv = home
    X, y, t, _, previous = gad.samples(iv, config=1)
    names, days = gad.CONFIGS[1]
    assert X.shape[1] == gad.WINDOW * len(names) + days + 2
    assert set(y) == {"Sleep", "Cook_Breakfast", "Watch_TV"}
    assert (np.diff(t.astype("int64")) >= 0).all()  # time order, for the time split


def test_strict_hourly_count_ignores_later_intervals():
    t = pd.to_datetime(["2013-03-01 08:00", "2013-03-01 08:10", "2013-03-01 08:20", "2013-03-01 08:30"])
    iv = pd.DataFrame({"start": t, "end": t + pd.Timedelta(seconds=3), "room": "Kitchen", "activity": "Cook"})
    codes = np.zeros(4)
    paper = gad.step_features(iv, codes, strict=False)["hourly_count"].tolist()
    strict = gad.step_features(iv, codes, strict=True)["hourly_count"].tolist()
    assert paper == [4, 4, 4, 4]  # the whole clock hour, later intervals included
    assert strict == [0, 1, 2, 3]  # only what came before


def test_undersample_caps_classes_at_five_times_the_smallest():
    y = np.array(["a"] * 100 + ["b"] * 30 + ["c"] * 5, dtype=object)
    kept = pd.Series(y[gad.undersample(y, np.random.default_rng(0))]).value_counts()
    assert kept.to_dict() == {"a": 25, "b": 25, "c": 5}


@pytest.mark.parametrize("variant", gad.VARIANTS)
def test_every_variant_runs_its_baselines(home, variant):
    _, _, iv = home
    rows = gad.run_home(iv, config=3, variant=variant, models=[])
    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == {"majority", "previous_activity", "history_vote"}
    assert by_model["history_vote"]["accuracy"] > 0.9  # the routine never changes


@pytest.mark.parametrize("model", gad.MODELS)
def test_gad_models_train_when_installed(home, model):  # runs on Kaggle; the laptop has none of them
    pytest.importorskip(model)
    _, _, iv = home
    rows = gad.run_home(iv, config=3, variant="paper", models=[model])
    assert rows[-1]["model"] == model and rows[-1]["accuracy"] > 0.9


def daily_task(days: int = 60) -> Task:
    hours = pd.date_range("2014-01-06", periods=days * 24, freq="h")
    uses = pd.DataFrame({"start": hours[hours.hour == 7] + pd.Timedelta(minutes=5)})
    labels = hourly_labels(uses, hours)
    train, val, test = time_split(labels)
    return Task(1, "Appliance1", "Kettle", "habit", labels, uses, train, val, test, None)


def test_random_split_deals_all_hours_once():
    t = random_split(daily_task())
    together = t.train.index.append(t.val.index).append(t.test.index)
    assert together.is_unique and len(together) == len(t.labels)
    assert t.test.index.min() < t.train.index.max()  # test hours sit between training hours
    assert list(t.labels.index) == list(together)


def test_hourly_inputs_use_only_hours_before():
    t = daily_task()
    h = t.labels.index[24 * 40 + 7]
    before = gad.hourly_inputs(t.labels, None).loc[h]
    changed = t.labels.copy()
    changed[changed.index >= h] = 1 - changed[changed.index >= h]
    after = gad.hourly_inputs(changed, None).loc[h]
    assert np.array_equal(before.to_numpy(), after.to_numpy())
