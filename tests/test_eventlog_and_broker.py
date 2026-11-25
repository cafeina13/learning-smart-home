import json
import socket
from datetime import datetime

import pytest

from home.bus import MqttBus
from home.devices import SimClock
from home.eventlog import EventLog
from home.simulation import ArticleHome


def test_event_log_records_states_commands_and_alerts(bus, clock):
    log = EventLog()
    log.attach(bus)
    home = ArticleHome(bus, clock)
    home.lamp.set("on", source="user")
    home.physics.extra_heat_c["hall/temperature"] = 10
    home.run(seconds=60)

    types = {row[0] for row in log.rows("SELECT DISTINCT type FROM messages")}
    assert types == {"state", "set", "alert"}
    (source,) = log.rows("SELECT source FROM messages WHERE type='set' AND device='living/lamp'")[0]
    assert source == "user"
    (value,) = log.rows("SELECT value FROM messages WHERE type='state' AND device='living/lamp'")[0]
    assert json.loads(value) == "on"


def broker_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 1883), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.broker
@pytest.mark.skipif(not broker_running(), reason="Mosquitto is not running on 127.0.0.1:1883")
def test_scenario3_over_real_mosquitto():
    # Same scenario as the in-memory test, but every message goes through the real broker.
    bus = MqttBus(client_id="capstone-test")
    try:
        clock = SimClock(datetime(2026, 10, 12, 12, 0))
        home = ArticleHome(bus, clock)
        log = EventLog()
        log.attach(bus)
        home.announce()
        home.car.set("on")
        home.run(seconds=15 * 60)  # step() flushes the broker, so no sleeping is needed
        assert home.fan.value == 2
        assert home.door.value == "open"
        # Each message is logged exactly once, even though several parts subscribe to it.
        dupes = log.rows("SELECT payload, COUNT(*) c FROM messages GROUP BY payload, topic HAVING c > 1")
        assert dupes == []
    finally:
        bus.close()
