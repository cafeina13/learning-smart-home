"""The one message format every part of the home uses.

Topics
------
home/<room>/<device>/state   a device reports its current value
home/<room>/<device>/set     someone (user, rule, dashboard) asks a device to change
home/alerts                  controllers raise alerts here

The article (Ozdogan & Das, 2021) publishes the garage smoke level to the topic
"Home/Garage". Here the same reading goes to "home/garage/smoke/state": one
pattern for every device, so the event log and the habit model never need
device-specific code.

Payloads are JSON objects. A state message always carries the same fields, so a
simulated device, a real device, or a replayed dataset row are interchangeable.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

ROOT = "home"
ALERTS_TOPIC = f"{ROOT}/alerts"


def state_topic(device_id: str) -> str:
    # device_id is "<room>/<name>", e.g. "garage/smoke".
    return f"{ROOT}/{device_id}/state"


def set_topic(device_id: str) -> str:
    return f"{ROOT}/{device_id}/set"


def device_id_from_topic(topic: str) -> str | None:
    # "home/garage/smoke/state" -> "garage/smoke"; anything else -> None.
    parts = topic.split("/")
    if len(parts) == 4 and parts[0] == ROOT and parts[3] in ("state", "set"):
        return f"{parts[1]}/{parts[2]}"
    return None


@dataclass
class StateEvent:
    device: str  # "<room>/<name>"
    kind: str  # device type, e.g. "temperature", "lamp"
    value: Any  # number, bool, or short string
    ts: str  # ISO-8601 time of the reading (simulated time in the simulation)
    source: str = "sim"  # "sim", "real", or "replay" (dataset)
    extra: dict = field(default_factory=dict)  # optional device-specific details

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "StateEvent":
        return cls(
            device=payload["device"],
            kind=payload["kind"],
            value=payload["value"],
            ts=payload["ts"],
            source=payload.get("source", "sim"),
            extra=payload.get("extra", {}),
        )


def command(value: Any, ts: datetime, source: str) -> dict:
    # Payload for a .../set topic. `source` says who asked: "user", "rule:<name>", "dashboard".
    return {"value": value, "ts": ts.isoformat(), "source": source}


def encode(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


def decode(raw: bytes) -> dict:
    return json.loads(raw.decode())
