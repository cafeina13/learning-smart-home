"""CASAS smart-home data (Cook et al., 2013): the comparison article's dataset.

Gad et al. (2025) use 21 single-resident CASAS homes but do not name them. The
public release (Zenodo 15708568, labeled_data.zip, CC BY 4.0) has 30 "hh" homes
of older adults living alone (hh107 and hh121 have two residents). The sizes in
their Table 4 (users 2-6) rise and fall with hh102-hh106, so we take the first
21 single-resident homes: HOMES below. This is inferred, not stated.

Each file is one home, one line per sensor event:

    2013-03-01,00:04:14.280135,Bedroom,ON,Sleep="begin"
    2013-03-01,00:04:15.356898,Bedroom,OFF

The sensor is named after its room (PIR motion: ON/OFF; doors: OPEN/CLOSE). The
optional fifth field annotates activities: X="begin" ... X="end" around a stretch
of events, or a bare X on a single event.

Two views of the same events:

intervals   Gad et al.'s samples: an ON (OPEN) and the next OFF (CLOSE) of the
            same sensor, at least MIN_INTERVAL_S long, labelled with the activity
            under way when it starts (None if none).
tables      our pipeline's uses / covered hours / "appliances", so every existing
            script runs on CASAS unchanged (CAPSTONE_DATASET=casas): each activity
            is an "appliance" switched on when the activity starts, and the
            question becomes "will this activity start in this hour?". The
            sensors are one more row, category "sensor": the motion a real home
            observes, which stands in for "the house's other appliances" (the
            other activities are human annotations, not something a home sees).
"""

import io
import zipfile

import numpy as np
import pandas as pd

URL = "https://zenodo.org/records/15708568/files/labeled_data.zip?download=1"
HOMES = [f"hh{n}" for n in [*range(101, 107), *range(108, 121), 122, 123]]
NOISY = {"Other_Activity", "Entertain_Guests", "Enter_Home"}  # removed by Gad et al. ("ENTER" = Enter_Home)
MIN_INTERVAL_S = 2.0
MIN_ACTIVE_HOURS = 1  # a day counts as recorded if its sensors fired at all (hh124: median 4 active hours a day)


def house_number(home: str) -> int:
    return int(home[2:])  # "hh101" -> 101, an int like REFIT's and UK-DALE's house numbers


def read_events(source) -> pd.DataFrame:
    """One home's events: time, sensor, message, annotation (empty string if none), in time order.
    Read line by line, not as strict CSV: some lines end in a stray comma (hh111:
    'Bedroom,OFF,Sleep="end",'), so everything after the fourth comma is the annotation."""
    text = source.read() if hasattr(source, "read") else open(source, encoding="utf-8", errors="replace").read()
    lines = pd.Series(text.splitlines())
    lines = lines[lines.str.strip() != ""]
    df = lines.str.split(",", n=4, expand=True).reindex(columns=range(5)).fillna("")
    df.columns = ["date", "clock", "sensor", "message", "note"]
    df["note"] = df["note"].str.strip().str.strip(",")
    df["time"] = pd.to_datetime(df["date"].str.strip() + " " + df["clock"].str.strip(), format="ISO8601")
    return df[["time", "sensor", "message", "note"]].sort_values("time", kind="stable").reset_index(drop=True)


def label_events(events: pd.DataFrame) -> pd.Series:
    """The activity under way at each event: begin/end markers open and close an activity (the
    most recently begun one wins when they overlap), a bare annotation labels its own event.
    An event may carry several annotations, comma-separated; the last one names it."""
    open_, labels = [], []
    for note in events["note"]:
        label = None
        for part in filter(None, (x.strip() for x in note.split(","))):
            if part.endswith('="begin"'):
                label = part[:-len('="begin"')]
                open_.append(label)
            elif part.endswith('="end"'):
                label = part[:-len('="end"')]
                if label in open_:
                    open_.remove(label)
            else:
                label = part
        labels.append(label if label is not None else (open_[-1] if open_ else None))
    return pd.Series(labels, index=events.index, dtype=object)


def intervals(events: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Gad et al.'s samples: ON->OFF (OPEN->CLOSE) of one sensor, at least MIN_INTERVAL_S long,
    labelled with the activity at the ON event."""
    rows, started = [], {}
    for i, (t, sensor, msg) in enumerate(zip(events["time"], events["sensor"], events["message"])):
        if msg in ("ON", "OPEN"):
            started.setdefault(sensor, (t, labels.iat[i]))
        elif msg in ("OFF", "CLOSE") and sensor in started:
            t0, label = started.pop(sensor)
            rows.append((t0, t, sensor, label))
    df = pd.DataFrame(rows, columns=["start", "end", "room", "activity"])
    df = df[(df["end"] - df["start"]).dt.total_seconds() >= MIN_INTERVAL_S]
    return df.sort_values("start", kind="stable").reset_index(drop=True)


def activity_starts(events: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Each time an activity starts: an event labelled X whose previous event was not X."""
    previous = labels.shift(1)
    start = labels.notna() & (labels != previous)
    return pd.DataFrame({"activity": labels[start].to_numpy(), "start": events.loc[start, "time"].to_numpy()})


def recorded_hours(events: pd.DataFrame) -> pd.DatetimeIndex:
    """All hours of the days on which the sensors fired (in at least MIN_ACTIVE_HOURS hours).
    A quiet hour on such a day (asleep, out) is a real "no"; a day without data is left out."""
    hours = events["time"].dt.floor("h").drop_duplicates()
    per_day = hours.groupby(hours.dt.normalize()).size()
    days = per_day.index[per_day >= MIN_ACTIVE_HOURS]
    return pd.DatetimeIndex(sorted(d + pd.Timedelta(hours=h) for d in days for h in range(24)))


def tables(home: str, events: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Our pipeline's (uses, covered hours, appliance rows) for one home."""
    house = house_number(home)
    starts = activity_starts(events, labels)
    starts = starts[~starts["activity"].isin(NOISY)]
    motion = events.loc[events["message"].isin(["ON", "OPEN"]), ["time"]].rename(columns={"time": "start"})
    motion["activity"] = "sensors"
    uses = pd.concat([starts, motion], ignore_index=True)
    uses["house"], uses["column"], uses["appliance"] = house, uses["activity"], uses["activity"]
    uses["end"] = uses["start"]
    uses = uses[["house", "column", "appliance", "start", "end"]].sort_values("start", kind="stable")
    hours = recorded_hours(events)
    covered = pd.DataFrame({"house": house, "hour": hours})
    days = max(1, len(hours) // 24)
    rows = []
    for name, group in uses.groupby("column"):
        rows.append({"house": house, "column": name, "appliance": name,
                     "category": "sensor" if name == "sensors" else "habit",
                     "uses": len(group), "days": days, "uses_per_day": round(len(group) / days, 2)})
    return uses.reset_index(drop=True), covered, rows


def read_zip(path, homes=HOMES):
    """(home, events) for each wanted home in labeled_data.zip (files labeled/<home>.csv)."""
    with zipfile.ZipFile(path) as zf:
        for home in homes:
            with zf.open(f"labeled/{home}.csv") as f:
                yield home, read_events(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))


def room_entropy(rooms: pd.Series) -> float:
    """Shannon entropy (bits) of the share of samples in each room."""
    p = rooms.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())
