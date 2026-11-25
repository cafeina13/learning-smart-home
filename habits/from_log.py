"""Read appliance uses back out of the home's own event log.

Each appliance reports "on" and "off" states over MQTT, and the event log keeps
them. Pairing every "on" with the next "off" gives the same uses table that
habits/refit.py builds from REFIT power traces (start, end, duration_s,
energy_wh), so the same routine finder and suggestion code run on both.

Energy per use is the device's rated power times how long it ran; a real home
with metering plugs would report measured energy instead.
"""

import json
from datetime import date, datetime, timedelta

import pandas as pd

from home.eventlog import EventLog

COLUMNS = ["start", "end", "duration_s", "energy_wh"]


def uses_from_log(log: EventLog, device: str, watts: float, only_source: str | None = None) -> pd.DataFrame:
    """Uses of one switch-type device. With only_source, keep only uses started by that source
    (e.g. "resident"), so automations the system itself ran are not learned back as habits."""
    # Commands and states in the order the log received them. Each "on" state is
    # credited to the most recent command for this device. (Matching by timestamp
    # would break: over a real broker the state can be stamped a step later than
    # the command that caused it.)
    rows = log.rows(
        "SELECT ts, type, value, source FROM messages WHERE device = ? AND type IN ('set', 'state') ORDER BY id",
        (device,))

    uses, started, started_by, last_command = [], None, None, None
    for ts, msg_type, value, source in rows:
        if msg_type == "set":
            last_command = source
            continue
        value = json.loads(value)
        t = datetime.fromisoformat(ts)
        if value == "on" and started is None:
            started, started_by = t, last_command
        elif value == "off" and started is not None:
            if only_source is None or started_by == only_source:
                seconds = (t - started).total_seconds()
                uses.append((started, t, seconds, watts * seconds / 3600))
            started = None
    df = pd.DataFrame(uses, columns=COLUMNS)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def days_in_log(log: EventLog) -> list[date]:
    """Every full day the log covers (the first and last, partial days are left out)."""
    (first, last), = log.rows("SELECT MIN(ts), MAX(ts) FROM messages WHERE ts != ''")
    if not first:
        return []
    a = datetime.fromisoformat(first).date() + timedelta(days=1)
    b = datetime.fromisoformat(last).date()
    return [a + timedelta(days=i) for i in range((b - a).days)]
