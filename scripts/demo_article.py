"""Run the article's three scenarios over the real Mosquitto broker and print what happens.

    .venv\\Scripts\\python.exe scripts\\demo_article.py

Simulated time runs fast (one simulated day takes a few seconds). Every message
is also stored in data/events.sqlite.

You can watch the raw MQTT traffic in a second terminal while it runs:
    "C:\\Program Files\\mosquitto\\mosquitto_sub.exe" -h 127.0.0.1 -t "home/#" -v
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from home.bus import MqttBus  # noqa: E402
from home.devices import SimClock  # noqa: E402
from home.eventlog import EventLog  # noqa: E402
from home.events import ALERTS_TOPIC  # noqa: E402
from home.simulation import ArticleHome  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "events.sqlite"
WATCHED = {"living/lamp", "living/window", "living/fan", "hall/door", "garage/door", "garage/car"}


def main() -> None:
    bus = MqttBus(client_id="capstone-demo")
    clock = SimClock(datetime(2026, 10, 12, 7, 0))  # after sunrise
    DB_PATH.unlink(missing_ok=True)
    log = EventLog(DB_PATH)
    log.attach(bus)

    def show(topic, payload):
        t = datetime.fromisoformat(payload["ts"]).strftime("%a %H:%M") if payload.get("ts") else "--"
        if topic == ALERTS_TOPIC:
            print(f"  {t}  ALERT  {payload['device']} = {payload['value']} C (average {payload['average']} C)")
        elif topic.endswith("/state") and payload["device"] in WATCHED:
            print(f"  {t}  {payload['device']:<14} -> {payload['value']}")

    home = ArticleHome(bus, clock)
    home.announce()
    bus.subscribe("home/#", show)  # print only what happens after the starting states

    def run(hours: float):
        for _ in range(int(hours * 360)):  # 10 s steps; each step waits for the broker
            home.step(10)

    run(2)  # the morning passes and the battery charges

    print("\nScenario 1: one room gets much hotter than the others")
    home.physics.extra_heat_c["kitchen/temperature"] = 12
    run(0.1)
    home.physics.extra_heat_c.clear()

    print("\nScenario 2 + 3: the car runs in the garage, smoke rises, the house ventilates")
    home.car.set("on", source="user")
    run(0.5)
    print(f"  smoke now {home.smoke.reading:.1f} (car still running, house open)")
    home.car.set("off", source="user")
    run(1.0)
    print(f"  smoke now {home.smoke.reading:.1f} (car off)")

    print("\nScenario 3: evening, the lamp is on, the sun sets, the solar rule takes over")
    run(6.5)  # to about 17:00
    home.lamp.set("on", source="user")
    run(1.5)  # sunset is at 18:00
    print(f"  battery {home.battery.reading:.0f}%  at {clock.now():%a %H:%M}")

    (count,) = log.rows("SELECT COUNT(*) FROM messages")[0]
    print(f"\n{count} messages logged to {DB_PATH}")
    bus.close()


if __name__ == "__main__":
    main()
