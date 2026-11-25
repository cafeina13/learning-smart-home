import importlib
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAPSTONE_BUS", "memory")
    monkeypatch.setenv("CAPSTONE_DEMO", "synthetic")  # small and fast; the REFIT demo needs the prepared data
    import dashboard.app as app_module
    importlib.reload(app_module)  # pick up the environment
    with TestClient(app_module.app) as c:
        c.post("/api/sim", json={"running": False})  # tests drive time themselves
        yield c, app_module


def test_state_lists_devices_and_energy(client):
    c, _ = client
    s = c.get("/api/state").json()
    ids = {d["id"] for d in s["devices"]}
    assert {"garage/smoke", "living/lamp", "kitchen/kettle", "utility/washing_machine"} <= ids
    assert set(s["energy"]) == {"solar_w", "battery_pct", "load_w", "meter"}
    assert "synthetic" in s["data_source"]


def test_switch_device_from_dashboard(client):
    c, _ = client
    assert c.post("/api/devices/living/lamp", json={"value": "on"}).status_code == 200
    assert c.post("/api/devices/living/fan", json={"value": "2"}).status_code == 200  # fan levels are numbers
    assert c.post("/api/devices/living/lamp", json={"value": "purple"}).status_code == 400
    assert c.post("/api/devices/garage/smoke", json={"value": "on"}).status_code == 404  # sensors cannot be set
    devices = {d["id"]: d["value"] for d in c.get("/api/state").json()["devices"]}
    assert devices["living/lamp"] == "on" and devices["living/fan"] == 2
    sources = {a["source"] for a in c.get("/api/state").json()["activity"]}
    assert "dashboard" in sources


def test_smart_start_from_suggestion_to_a_solar_run(client):
    c, app_module = client
    pending = [s for s in c.get("/api/suggestions").json() if s["status"] == "pending"]
    smart = next(s for s in pending if s["kind"] == "smart_start" and s["device"] == "kitchen/dishwasher")
    assert c.post("/api/suggestions/decide", json={"key": smart["key"], "accept": True}).status_code == 200
    assert c.post("/api/suggestions/decide", json={"key": smart["key"], "accept": True}).status_code == 409

    # The user arms the dishwasher in the evening; arming without a deadline is refused.
    rt = app_module.rt
    with rt.lock:
        while rt.clock.now().hour < 21:
            rt.home.step(60)
    assert c.post("/api/devices/kitchen/dishwasher", json={"value": "armed"}).status_code == 400
    tomorrow_6pm = rt.clock.now().replace(hour=18, minute=0) + timedelta(days=1)
    r = c.post("/api/devices/kitchen/dishwasher", json={"value": "armed", "deadline": tomorrow_6pm.isoformat()})
    assert r.status_code == 200

    plan = c.get("/api/state").json()["plans"][0]
    assert len(plan["options"]) == 3 and plan["options"][0]["solar_share"] > 0.5
    second = plan["options"][1]["start"]
    assert c.post("/api/plans/choose", json={"device": "kitchen/dishwasher", "start": second}).status_code == 200

    with rt.lock:
        while rt.clock.now() < datetime.fromisoformat(second):
            rt.home.step(60)
        rt.home.step(60)
    devices = {d["id"]: d for d in c.get("/api/state").json()["devices"]}
    assert devices["kitchen/dishwasher"]["value"] == "on"
    assert devices["kitchen/dishwasher"]["control"] == "armable"


def test_manual_devices_report_their_control(client):
    c, _ = client
    devices = {d["id"]: d for d in c.get("/api/state").json()["devices"]}
    assert devices["living/television_site"]["control"] == "manual"
    assert devices["kitchen/coffee_machine"]["control"] == "auto"


def test_tariff_and_quiet_hours_from_the_dashboard(client):
    c, _ = client
    assert c.get("/api/state").json()["smart_start_settings"] == {"tariff": "none", "quiet": False}
    assert c.post("/api/smart-start/settings", json={"tariff": "economy7", "quiet": True}).json() == {
        "tariff": "economy7", "quiet": True}
    assert c.post("/api/smart-start/settings", json={"tariff": "cheapest"}).status_code == 400
    now = datetime.fromisoformat(c.get("/api/state").json()["time"])
    deadline = (now + timedelta(days=1)).replace(hour=18, minute=0)
    assert c.post("/api/devices/kitchen/dishwasher", json={"value": "armed", "deadline": deadline.isoformat()}).status_code == 200
    plan = c.get("/api/state").json()["plans"][0]
    assert all("cost_pence" in o for o in plan["options"])
    assert all(not (datetime.fromisoformat(o["start"]).hour >= 22 or datetime.fromisoformat(o["start"]).hour < 7)
               for o in plan["options"])
