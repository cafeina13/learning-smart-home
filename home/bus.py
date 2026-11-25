"""The message bus: publish a JSON payload to a topic, subscribe to topic patterns.

Two implementations with the same two methods:

- MqttBus talks to a real MQTT broker (Mosquitto on 127.0.0.1 by default).
  This is what the demo runs on, as in the article.
- InMemoryBus delivers messages inside one Python process. Tests use it so
  they run in milliseconds and need no broker.

Topic patterns follow MQTT rules: "+" matches one level, "#" matches the rest.
"""

import socket
import threading
from collections import deque
from typing import Callable, Protocol

import paho.mqtt.client as mqtt
from paho.mqtt.client import topic_matches_sub

from home.events import decode, encode

Handler = Callable[[str, dict], None]  # (topic, payload) -> None


class Bus(Protocol):
    def publish(self, topic: str, payload: dict) -> None: ...

    def subscribe(self, pattern: str, handler: Handler) -> None: ...

    def flush(self) -> None:
        """Return once every message published so far has reached its handlers."""
        ...


class InMemoryBus:
    """Synchronous bus for tests and offline replays.

    Messages published while another message is being handled are queued and
    delivered afterwards, in order. That mirrors a broker (no handler runs inside
    another) and keeps rule chains from recursing.
    """

    def __init__(self):
        self._subs: list[tuple[str, Handler]] = []
        self._queue: deque[tuple[str, dict]] = deque()
        self._delivering = False
        self.published: list[tuple[str, dict]] = []  # every message, for tests

    def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))
        self._queue.append((topic, payload))
        if self._delivering:
            return
        self._delivering = True
        try:
            while self._queue:
                t, p = self._queue.popleft()
                for pattern, handler in list(self._subs):
                    if topic_matches_sub(pattern, t):
                        handler(t, p)
        finally:
            self._delivering = False

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def flush(self) -> None:
        pass  # delivery is already synchronous


class MqttBus:
    """Bus backed by a real MQTT broker.

    The client subscribes to the broker once, to "#", and matches patterns itself.
    Subscribing to each pattern separately would make Mosquitto send one copy
    of a message per overlapping subscription, so handlers would see duplicates.

    paho runs its network loop in a background thread, so handlers are called from
    that thread. A lock makes sure only one handler runs at a time, which is
    the same guarantee InMemoryBus gives.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1883, client_id: str = ""):
        self._subs: list[tuple[str, Handler]] = []
        self._lock = threading.Lock()
        # Every message we publish comes back to us (we subscribe to "#"), so
        # comparing the two counts tells flush() when the broker has caught up.
        # Messages from other clients also count as received; that only matters
        # if another client publishes during a simulation step.
        self._counts = threading.Condition()
        self._sent = 0
        self._received = 0
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._connected = threading.Event()
        self._client.connect(host, port)
        self._client.loop_start()
        if not self._connected.wait(timeout=5):
            raise ConnectionError(f"No MQTT broker answered on {host}:{port}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        # Runs again after every reconnect, so a broker restart does not lose the subscription.
        # TCP_NODELAY: send small MQTT packets at once instead of batching them (Nagle).
        client.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.subscribe("#")
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        payload = decode(msg.payload)
        with self._lock:
            for pattern, handler in list(self._subs):
                if topic_matches_sub(pattern, msg.topic):
                    handler(msg.topic, payload)
        with self._counts:
            self._received += 1
            self._counts.notify_all()

    def publish(self, topic: str, payload: dict) -> None:
        with self._counts:
            self._sent += 1
        self._client.publish(topic, encode(payload))

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def flush(self, timeout: float = 5.0) -> None:
        # A simulation calls this between steps, so simulated time never runs ahead
        # of message delivery (otherwise a rule would react "late" in simulated time).
        # (Waiting on a condition, not polling with sleep: on Windows the shortest
        # sleep is ~15 ms, which would cap the simulation at a few dozen steps a second.)
        with self._counts:
            if not self._counts.wait_for(lambda: self._received >= self._sent, timeout):
                raise TimeoutError("MQTT broker did not deliver all messages in time")

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
