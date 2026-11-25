"""Measure the MQTT traffic: the gap the article itself leaves open.

    .venv\\Scripts\\python.exe scripts\\measure_mqtt.py

Ozdogan & Das (2021) end with: "analysis of network traffic and bandwidth
consumption is not examined. Depending on the number of IoT devices, new
scenarios can be developed." Both parts are measured here, on the local
Mosquitto broker:

A. What a real home sends. The REFIT house-6 demo (the article's devices plus
   the household's 8 appliances, with its accepted suggestions) lives one
   simulated day over Mosquitto. Every message is counted by type and device,
   with its size: payload bytes and MQTT bytes on the wire (QoS 0 PUBLISH =
   2-byte fixed header + 2-byte topic length + topic + payload; TCP/IP headers
   come on top and are not counted).

B. How the broker copes as the home grows. A publisher and a subscriber, in two
   separate processes with their own connections (TCP_NODELAY, like MqttBus),
   exchange messages at rising rates, from a normal home to thousands of
   devices. For each rate: delivery latency (median, 95th, 99th percentile) and
   messages lost. (Timestamps use the system-wide high-resolution counter, so
   they compare across processes.)

Writes data/results/mqtt_traffic.csv and data/results/mqtt_load.csv.
"""

import json
import multiprocessing
import socket
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

from home.bus import MqttBus  # noqa: E402
from home.events import encode  # noqa: E402

RESULTS = ROOT / "data" / "results"
RATES = [1, 10, 100, 1000, 5000, 20000]  # messages per second
SECONDS_PER_RATE = 5


def mqtt_bytes(topic: str, payload_bytes: int) -> int:
    body = 2 + len(topic.encode()) + payload_bytes  # topic length field + topic + payload (QoS 0)
    length_field = 1 if body < 128 else 2 if body < 16384 else 3
    return 1 + length_field + body


# --- A. one day of a real home ----------------------------------------------------------------

def traffic_of_one_day() -> pd.DataFrame:
    from habits.demo import build_refit_demo
    from habits.replay import ReplayResident
    from home.devices import SimClock
    from home.simulation import ArticleHome

    demo = build_refit_demo()
    bus = MqttBus(client_id="capstone-measure-home")
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    def count(topic: str, payload: dict) -> None:
        parts = topic.split("/")
        if parts[1] == "plans":
            kind, device = "plan (smart start)", "/".join(parts[2:4])
        elif parts[1] == "alerts":
            kind, device = "alert", "-"
        else:
            kind, device = {"state": "state (device reports)", "set": "set (commands)"}[parts[3]], "/".join(parts[1:3])
        size = len(encode(payload))
        c = counts[(kind, device)]
        c[0] += 1
        c[1] += size
        c[2] += mqtt_bytes(topic, size)

    bus.subscribe("#", count)
    clock = SimClock(demo.start)
    home = ArticleHome(bus, clock, solar=demo.solar_w)
    devices = {a.device: home.add_appliance(a.device, a.control, a.watts, a.standby_w, a.run_minutes)
               for a in demo.appliances}
    smart_start = {}
    for s in demo.suggestions:  # accept everything, the busiest realistic case
        smart_start.update({s.device: s.setting} if s.setting else {})
        for rule in s.rules:
            home.engine.add(rule)
    resident = ReplayResident(clock, devices, demo.replay_uses, smart_start=smart_start)
    resident.skip_to(demo.start)
    home.announce()
    end = demo.start + timedelta(days=1)
    while clock.now() < end:
        resident.act()
        home.step(10)
    bus.close()
    rows = [{"type": k, "device": d, "messages": c[0], "payload_bytes": c[1], "mqtt_bytes": c[2]}
            for (k, d), c in counts.items()]
    return pd.DataFrame(rows)


# --- B. broker under load ---------------------------------------------------------------------

def _no_delay(client) -> None:
    client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _subscriber(total: int, conn) -> None:
    """Runs in its own process: receive `total` messages (or time out), send back the latencies."""
    latencies, done = [], threading.Event()

    def on_message(client, userdata, msg):
        latencies.append(time.perf_counter() - json.loads(msg.payload)["t"])
        if len(latencies) >= total:
            done.set()

    def on_connect(client, userdata, flags, rc, props):
        _no_delay(client)
        client.subscribe("load/#")
        conn.send("ready")

    sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="capstone-load-sub")
    sub.on_message, sub.on_connect = on_message, on_connect
    sub.connect("127.0.0.1", 1883)
    sub.loop_start()
    done.wait(SECONDS_PER_RATE + 15)
    sub.loop_stop()
    sub.disconnect()
    conn.send(latencies)


def load_test(rate: int, seconds: int = SECONDS_PER_RATE) -> dict:
    total = rate * seconds
    here, there = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_subscriber, args=(total, there))
    proc.start()
    here.recv()  # subscriber is connected and subscribed

    # The publisher writes each message itself (loop_write), without paho's background
    # thread: on Windows that thread takes ~15 ms to wake up for an occasional
    # message, which would be measured as broker latency (measured separately: with
    # the thread, 14-17 ms median at 10-100 msg/s; written directly, 0.2-0.3 ms).
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="capstone-load-pub")
    pub.connect("127.0.0.1", 1883)
    _no_delay(pub)
    # A payload shaped like a sensor state message.
    template = {"device": "living/temperature", "kind": "temperature", "value": 21.5,
                "ts": "2015-02-17T06:30:00", "source": "sim", "extra": {}}
    start = time.perf_counter()
    for i in range(total):
        target = start + i / rate
        while time.perf_counter() < target:  # pace evenly (busy-wait: Windows sleep is ~15 ms)
            pass
        pub.publish(f"load/device{i % 50}/state", json.dumps({**template, "t": time.perf_counter()}))
        pub.loop_write()
    latencies = sorted(here.recv())
    proc.join()
    pub.disconnect()
    q = lambda p: latencies[min(len(latencies) - 1, int(p * len(latencies)))] * 1000 if latencies else float("nan")  # noqa: E731
    return {"rate_per_s": rate, "sent": total, "received": len(latencies), "lost": total - len(latencies),
            "median_ms": round(statistics.median(latencies) * 1000, 3) if latencies else None,
            "p95_ms": round(q(0.95), 3), "p99_ms": round(q(0.99), 3)}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    day = traffic_of_one_day()
    day.to_csv(RESULTS / "mqtt_traffic.csv", index=False)
    by_type = day.groupby("type")[["messages", "payload_bytes", "mqtt_bytes"]].sum().sort_values("messages",
                                                                                                ascending=False)
    total = by_type.sum()
    print("A. One simulated day of REFIT house 6 in the article's home, over Mosquitto\n")
    print(by_type.to_string())
    print(f"\n  total: {int(total['messages'])} messages/day = {total['messages'] / 86400 * 60:.1f} per minute; "
          f"{total['mqtt_bytes'] / 1024:.0f} KiB/day on MQTT = {total['mqtt_bytes'] * 8 / 86400:.0f} bit/s on average; "
          f"mean message {total['mqtt_bytes'] / total['messages']:.0f} bytes")
    top = day.groupby("device")["messages"].sum().sort_values(ascending=False).head(6)
    print("  busiest devices (messages/day):", ", ".join(f"{d} {n}" for d, n in top.items()))

    print("\nB. Broker under load: publisher -> Mosquitto -> subscriber (local, QoS 0)\n")
    load = pd.DataFrame([load_test(r) for r in RATES])
    load.to_csv(RESULTS / "mqtt_load.csv", index=False)
    print(load.to_string(index=False))


if __name__ == "__main__":
    main()
