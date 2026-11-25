"""Phone notifications: the right events become messages, once each, in the simulated home."""

import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from home.devices import SimClock
from home.events import ALERTS_TOPIC
from home.notify import NtfySender, PhoneNotifier, nice_name
from home.rules import RuleEngine
from home.simulation import ArticleHome


class Inbox:
    def __init__(self):
        self.messages = []

    def __call__(self, title, message, priority, tags):
        self.messages.append((title, message, priority))

    def titles(self):
        return [t for t, _, _ in self.messages]


def test_names_read_well():
    assert nice_name("kitchen/washing_machine") == "Washing machine (kitchen)"


def test_smoke_in_the_garage_is_one_urgent_message(bus, clock):
    home = ArticleHome(bus, clock)
    inbox = Inbox()
    PhoneNotifier(bus, inbox)
    home.car.set("on", source="user")
    home.run(seconds=15 * 60, dt=60)  # smoke builds up past 10, the article's rule opens everything
    smoke = [m for m in inbox.messages if m[0] == "SMOKE in the garage"]
    assert len(smoke) == 1 and smoke[0][2] == "urgent"


def test_smart_start_plan_moved_and_started():
    from home.bus import InMemoryBus
    bus, clock = InMemoryBus(), SimClock(datetime(2026, 10, 12, 21, 0))
    home = ArticleHome(bus, clock)
    inbox = Inbox()
    PhoneNotifier(bus, inbox)
    dishwasher = home.add_appliance("kitchen/dishwasher", "armable", watts=600, standby_w=1, run_minutes=120)
    dishwasher.set("on", source="rule:someone's automation")  # refused: not loaded and armed
    assert inbox.titles() == ["Not switched: Dishwasher (kitchen)"]
    assert "must be armed first" in inbox.messages[0][1]

    dishwasher.set("armed", source="user", deadline=datetime(2026, 10, 13, 18, 0).isoformat())
    plan = home.scheduler.plans["kitchen/dishwasher"]
    assert inbox.titles()[-1] == "Smart start: Dishwasher (kitchen)"
    assert "Armed: starts at" in inbox.messages[-1][1] and "done before 18:00" in inbox.messages[-1][1]

    second = plan.options[1].start
    home.scheduler.choose("kitchen/dishwasher", second)
    assert inbox.messages[-1][1].startswith(f"Moved: starts at {second:%H:%M}")
    home.run(seconds=(second - clock.now()).total_seconds() + 120, dt=60)
    assert inbox.titles()[-1] == "Started: Dishwasher (kitchen)"


def test_rule_switching_itself_off_is_reported(bus, clock):
    inbox = Inbox()
    PhoneNotifier(bus, inbox)
    bus.publish(ALERTS_TOPIC, {"kind": "rule_disabled", "rule": "cut living/tv 23:00-06:30", "overrides": 3,
                               "ts": clock.now().isoformat()})
    assert inbox.titles() == ["Automation switched off"]
    assert "overrode it 3 times" in inbox.messages[0][1]
    assert RuleEngine.MAX_OVERRIDES == 3


def test_paused_while_skipping_days(bus, clock):
    home = ArticleHome(bus, clock)
    inbox = Inbox()
    notifier = PhoneNotifier(bus, inbox)
    notifier.paused = True
    home.car.set("on", source="user")
    home.run(seconds=15 * 60, dt=60)
    assert inbox.messages == []


def test_sender_posts_to_ntfy_in_the_background():
    received = []
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            received.append((self.headers["Title"], self.headers["Priority"], body))
            self.send_response(200)
            self.end_headers()
            done.set()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    send = NtfySender(f"http://127.0.0.1:{server.server_port}/test-topic")
    send("Too hot: Kitchen (kitchen)", "31.0 °C, the home's average is 22.0 °C", "high", "thermometer")
    assert done.wait(5)
    server.server_close()
    assert received == [("Too hot: Kitchen (kitchen)", "high", "31.0 °C, the home's average is 22.0 °C")]
