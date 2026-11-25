"""SQLite log of every message in the home: states, commands, and alerts.

This is the home's memory. The dashboard reads history from it, and the habit
model can later learn from it the same way it learns from REFIT.

One table, one row per message. The common fields get their own columns so
they can be queried; the full payload is kept as JSON so nothing is lost.
"""

import json
import sqlite3
import threading
from pathlib import Path

from home.bus import Bus
from home.events import ROOT, device_id_from_topic

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id      INTEGER PRIMARY KEY,
    ts      TEXT NOT NULL,     -- time inside the home (simulated time in the simulation)
    topic   TEXT NOT NULL,
    type    TEXT NOT NULL,     -- 'state', 'set', or 'alert'
    device  TEXT,              -- '<room>/<name>', NULL for alerts
    value   TEXT,              -- JSON-encoded value
    source  TEXT,              -- who produced it: sim, real, replay, user, rule:<name>
    payload TEXT NOT NULL      -- the full JSON payload
);
CREATE INDEX IF NOT EXISTS messages_device_ts ON messages (device, ts);
"""


class EventLog:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: MqttBus calls handlers from paho's thread.
        # The MQTT thread writes while the web server reads, so access is serialised with a lock.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        # WAL without a disk sync on every commit: ~50x faster on Windows (2 ms -> 0.04 ms
        # per message). The database stays consistent; a power cut can lose only the
        # last moments of messages, which is acceptable for a log.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def attach(self, bus: Bus) -> None:
        bus.subscribe(f"{ROOT}/#", self.record)

    def record(self, topic: str, payload: dict) -> None:
        kind = topic.rsplit("/", 1)[-1]
        msg_type = {"state": "state", "set": "set", "alerts": "alert"}.get(kind, "other")
        with self._lock:
            self._insert(topic, msg_type, payload)

    def _insert(self, topic: str, msg_type: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO messages (ts, topic, type, device, value, source, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                payload.get("ts", ""),
                topic,
                msg_type,
                device_id_from_topic(topic) or payload.get("device"),
                json.dumps(payload.get("value")),
                payload.get("source"),
                json.dumps(payload),
            ),
        )
        self.db.commit()

    def rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self.db.execute(sql, params).fetchall()

    def close(self) -> None:
        self.db.close()
