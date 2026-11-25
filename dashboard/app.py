"""Web dashboard: watch and control the simulated home, and accept or reject learned automations.

It replaces the article's tablet (scenario 2's MQTT subscriber) and its gateway
web page (scenario 3's rule editor). The server:

- runs the article's home in a background thread, in simulated time
  (`speed` simulated seconds per real second, adjustable from the page),
- talks to it only over the MQTT bus, like any other client,
- offers suggestions; an accepted one becomes live rules (automations, standby
  cuts) or a setting (smart start for a machine),
- shows each armed machine's smart-start options and lets the user pick one,
  and lets the resident choose a tariff (cost-aware smart start) and quiet hours.

Two demos (CAPSTONE_DEMO):

refit      (default when the REFIT data is prepared) a real household, REFIT
           house 6: suggestions learned from its first 80% of days, then its
           other 20% replayed at their real times with that day's real sun
           (habits/demo.py, habits/replay.py). Accepted suggestions act on real
           behaviour the system never learned from.
synthetic  the made-up household and resident (habits/synthetic.py), small and
           fast; the tests use it.

Run with:  .venv\\Scripts\\python.exe scripts\\run_dashboard.py
"""

import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from habits import synthetic
from habits.forest import ForestGuard
from habits.from_log import days_in_log, uses_from_log
from habits.resident import Resident
from habits.suggestions import SuggestionStore, control_of, device_id, suggest_for_appliance
from home.bus import InMemoryBus, MqttBus
from home.devices import Actuator, ArmableAppliance, SimClock
from home.eventlog import EventLog
from home.notify import attach_from_env
from home.simulation import ArticleHome

ROOT = Path(__file__).resolve().parents[1]
# Overridable so tests can use a temporary folder and skip the broker.
DATA = Path(os.environ.get("CAPSTONE_DATA_DIR", ROOT / "data"))
USE_BROKER = os.environ.get("CAPSTONE_BUS", "mqtt") == "mqtt"
REFIT_READY = (ROOT / "data" / "processed" / "refit" / "uses.parquet").exists()
DEMO = os.environ.get("CAPSTONE_DEMO", "refit" if REFIT_READY else "synthetic")
STATIC = Path(__file__).resolve().parent / "static"
STEP_S = 10  # simulated seconds per simulation step

# Synthetic demo: appliances added to the article's home, (watts, minutes per run);
# the coffee machine is the article's own.
APPLIANCES = {
    "Coffee Machine": (1000, 5),
    "Kettle": (2000, 3),
    "Television Site": (90, 200),
    "Washing Machine": (450, 110),
    "Dishwasher": (600, 120),
}


QUIET_HOURS = (22, 7)  # when quiet hours are on, no machine starts from 22:00 to 07:00


class Runtime:
    """Everything the server holds: the home, its bus, the logs, and the simulation thread."""

    def __init__(self, demo: str = DEMO):
        self.bus = self._connect()
        self.lock = threading.Lock()  # one change at a time: simulation step or a request
        self.speed = 600.0  # simulated seconds per real second (10 min/s)
        self.running = True
        self._stop = threading.Event()
        self.demo = demo

        for f in ("events.sqlite", "suggestions.sqlite"):
            (DATA / f).unlink(missing_ok=True)  # every server start is a fresh demo
        self.store = SuggestionStore(DATA / "suggestions.sqlite")
        self.smart_start: dict[str, dict] = {}  # device -> accepted smart-start setting
        self.tariff, self.quiet = "none", False  # the resident's smart-start settings (cost-aware, quiet hours)
        self.switches = {}
        self.guard: ForestGuard | None = None  # the REFIT demo only (its forests need real history)
        self.appliances: list[tuple[str, str, float, float]] = []  # (name, device, watts, standby W)
        if demo == "refit":
            self._build_refit()
        else:
            self._build_synthetic()
        self.log = EventLog(DATA / "events.sqlite")
        self.log.attach(self.bus)
        self.notifier = attach_from_env(self.bus)  # phone notifications when NTFY_URL is set
        self.home.announce()

    def _build_synthetic(self) -> None:
        self.clock = SimClock(datetime(2026, 10, 12, 6, 30))  # a Monday morning, just after sunrise
        self.home = ArticleHome(self.bus, self.clock)
        self.data_source = "synthetic demo household"
        uses, days, standby = synthetic.household()
        for appliance, (watts, minutes) in APPLIANCES.items():
            dev = device_id(appliance)
            self.switches[dev] = self.home.add_appliance(dev, control_of(appliance), watts=watts,
                                                         standby_w=standby.get(appliance, 0.0), run_minutes=minutes)
            self.appliances.append((appliance, dev, watts, standby.get(appliance, 0.0)))
            if appliance in uses:
                for s in suggest_for_appliance(appliance, uses[appliance], days, standby.get(appliance, 0.0),
                                               learned_from="synthetic demo household"):
                    self.store.offer(s)
        self.resident = Resident(self.clock, self.switches, smart_start=self.smart_start)

    def _build_refit(self) -> None:
        from habits.demo import build_refit_demo  # needs the prepared REFIT data
        from habits.replay import ReplayResident

        demo = build_refit_demo()
        self.clock = SimClock(demo.start)
        self.home = ArticleHome(self.bus, self.clock, solar=demo.solar_w)
        self.data_source = (f"real household: {demo.learned_from}; now replaying its "
                            f"{demo.start:%d %b} - {demo.end:%d %b %Y} days, never learned from")
        for a in demo.appliances:
            self.switches[a.device] = self.home.add_appliance(a.device, a.control, watts=a.watts,
                                                              standby_w=a.standby_w, run_minutes=a.run_minutes)
            self.appliances.append((a.name, a.device, a.watts, a.standby_w))
        for s in demo.suggestions:
            self.store.offer(s)
        self.resident = ReplayResident(self.clock, self.switches, demo.replay_uses, smart_start=self.smart_start)
        self.resident.skip_to(demo.start)
        # The habit forests' guard (habits/forest.py): restores a cut plug when a use looks likely this hour.
        self.guard = ForestGuard(self.bus, self.clock, demo.guarded, {a.device for a in demo.appliances})
        # Load-aware smart start: rank start times by the solar left after the house's usual load.
        self.home.scheduler.forecast = lambda t: max(0.0, demo.solar_w(t) - demo.expected_load_w(t))

    def set_smart_start(self, tariff: str | None, quiet: bool | None) -> None:
        """Caller holds self.lock. "none" ranks by solar; a tariff ranks by cost (habits/tariffs.py)."""
        from habits.tariffs import EXPORT, TARIFFS, Prices

        if tariff is not None:
            if tariff != "none" and tariff not in TARIFFS:
                raise ValueError(f"unknown tariff {tariff!r}")
            self.tariff = tariff
            self.home.scheduler.price = None if tariff == "none" else Prices(tariff)
            self.home.scheduler.export_price = EXPORT
        if quiet is not None:
            self.quiet = quiet
            self.home.scheduler.quiet = QUIET_HOURS if quiet else None

    def step(self, dt: float) -> None:
        # Caller holds self.lock.
        self.resident.act()
        self.home.step(dt)
        if self.guard is not None:
            self.guard.tick()
            self.bus.flush()

    def skip(self, days: int) -> None:
        """Live through `days` simulated days as fast as possible (1-minute steps), without
        phone notifications (days of them at once would only be noise)."""
        with self.lock:
            if self.notifier:
                self.notifier.paused = True
            try:
                for _ in range(days * 24 * 60):
                    self.step(60)
            finally:
                if self.notifier:
                    self.bus.flush()
                    self.notifier.paused = False

    def learn_from_log(self) -> dict:
        """Find routines in this home's own event log and offer any new suggestions."""
        days = days_in_log(self.log)
        offered = 0
        for appliance, dev, watts, standby in self.appliances:
            uses = uses_from_log(self.log, dev, watts, only_source="resident")
            for s in suggest_for_appliance(appliance, uses, days, standby,
                                           learned_from=f"this home's event log ({len(days)} days)"):
                offered += self.store.offer(s)
        self.data_source = f"this home's own event log, {len(days)} full days"
        return {"days": len(days), "new_suggestions": offered}

    @staticmethod
    def _connect():
        if not USE_BROKER:
            return InMemoryBus()
        try:
            with socket.create_connection(("127.0.0.1", 1883), timeout=0.5):
                pass
            return MqttBus(client_id="capstone-dashboard")
        except OSError:
            print("Mosquitto is not running: using the in-process bus instead.")
            return InMemoryBus()

    def loop(self) -> None:
        # Keeps simulated time at `speed` x real time: every ~20 ms it runs as many
        # steps as are due. (Sleeping between single steps would not work on
        # Windows, where the shortest sleep is ~15 ms.)
        owed = 0.0  # simulated seconds that are due but not yet stepped
        last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.02)
            now = time.monotonic()
            if self.running:
                owed = min(owed + (now - last) * self.speed, 3600.0)  # never try to catch up more than an hour
                while owed >= STEP_S and not self._stop.is_set():
                    with self.lock:
                        self.step(STEP_S)
                    owed -= STEP_S
            last = now

    def start(self) -> None:
        threading.Thread(target=self.loop, daemon=True, name="simulation").start()

    def stop(self) -> None:
        self._stop.set()
        if isinstance(self.bus, MqttBus):
            self.bus.close()


rt: Runtime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rt
    rt = Runtime()
    rt.start()
    yield
    rt.stop()


app = FastAPI(title="Learning smart home", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    h = rt.home
    with rt.lock:
        devices = [{
            "id": d.id,
            "room": d.room,
            "kind": d.kind,
            "value": d.value,
            "power_w": round(d.power_w(), 1),
            "allowed": list(d.allowed) if isinstance(d, Actuator) else None,
            "control": getattr(d, "control", None),
            "smart_start": rt.smart_start.get(d.id),
        } for d in h.devices]
        plans = [p.to_dict() for p in h.scheduler.plans.values()]
        smart_start_settings = {"tariff": rt.tariff, "quiet": rt.quiet}
        rules = [{"name": r.name, "origin": r.origin, "enabled": r.enabled,
                  "overrides": h.engine.overrides.get(r.name, 0)} for r in h.engine.rules]
        energy = {
            "solar_w": round(h.solar.reading, 1),
            "battery_pct": round(h.battery.reading, 1),
            "load_w": round(h.load_w(), 1),
            "meter": {k: round(v, 3) for k, v in h.meter.items()},
        }
        now = rt.clock.now()
    activity = rt.log.rows(
        "SELECT ts, type, device, value, source, payload FROM messages "
        "WHERE type = 'alert' OR (type = 'set' AND source != 'sim') ORDER BY id DESC LIMIT 25"
    )
    return {
        "time": now.isoformat(),
        "running": rt.running,
        "speed": rt.speed,
        "data_source": rt.data_source,
        "devices": devices,
        "energy": energy,
        "activity": [dict(zip(("ts", "type", "device", "value", "source", "payload"), r)) for r in activity],
        "plans": plans,
        "smart_start_settings": smart_start_settings,
        "rules": rules,
    }


class SetValue(BaseModel):
    value: str | int
    deadline: str | None = None  # ISO time, when arming a machine


@app.post("/api/devices/{room}/{name}")
def set_device(room: str, name: str, body: SetValue):
    device = next((d for d in rt.home.devices if d.id == f"{room}/{name}"), None)
    if not isinstance(device, Actuator):
        raise HTTPException(404, "No controllable device with that id")
    value = body.value
    if isinstance(device.allowed[0], int):
        value = int(value)
    if value not in device.allowed:
        raise HTTPException(400, f"{device.id} accepts {list(device.allowed)}")
    fields = {}
    if value == "armed":
        if not body.deadline:
            raise HTTPException(400, "Arming needs a deadline")
        deadline = datetime.fromisoformat(body.deadline)
        if deadline <= rt.clock.now():
            raise HTTPException(400, "The deadline must be in the future")
        fields["deadline"] = deadline.isoformat()
    with rt.lock:
        device.set(value, source="dashboard", **fields)
        rt.bus.flush()
    return {"ok": True}


@app.get("/api/suggestions")
def suggestions():
    return rt.store.list()


class Decision(BaseModel):
    key: str
    accept: bool


@app.post("/api/suggestions/decide")
def decide(body: Decision):
    accepted = rt.store.decide(body.key, body.accept)
    if body.accept:
        if accepted is None:
            raise HTTPException(409, "Suggestion not found or already decided")
        with rt.lock:
            for rule in accepted.rules:
                rt.home.engine.add(rule)
            if accepted.setting:
                rt.smart_start[accepted.device] = accepted.setting
    return {"ok": True}


class Choice(BaseModel):
    device: str
    start: str  # ISO time of one of the plan's options (or any time that still meets the deadline)


@app.post("/api/plans/choose")
def choose(body: Choice):
    with rt.lock:
        ok = rt.home.scheduler.choose(body.device, datetime.fromisoformat(body.start))
    if not ok:
        raise HTTPException(409, "No plan for that device, or that start would miss the deadline")
    return {"ok": True}


class SmartStartSettings(BaseModel):
    tariff: str | None = None  # "none" (most solar), "flat", "economy7" or "agile"
    quiet: bool | None = None  # no starts in the quiet hours (22:00-07:00)


@app.post("/api/smart-start/settings")
def smart_start_settings(body: SmartStartSettings):
    with rt.lock:
        try:
            rt.set_smart_start(body.tariff, body.quiet)
        except ValueError as err:
            raise HTTPException(400, str(err))
        return {"tariff": rt.tariff, "quiet": rt.quiet}


class SimControl(BaseModel):
    running: bool | None = None
    speed: float | None = None
    skip_days: int | None = None


@app.post("/api/sim")
def sim(body: SimControl):
    if body.running is not None:
        rt.running = body.running
    if body.speed is not None:
        rt.speed = min(max(body.speed, 10.0), 7200.0)
    if body.skip_days:
        rt.skip(min(max(body.skip_days, 1), 28))
    return {"running": rt.running, "speed": rt.speed, "time": rt.clock.now().isoformat()}


@app.post("/api/learn")
def learn():
    return rt.learn_from_log()
