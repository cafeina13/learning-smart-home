from home.events import StateEvent, decode, device_id_from_topic, encode, set_topic, state_topic


def test_topics_round_trip():
    assert state_topic("garage/smoke") == "home/garage/smoke/state"
    assert set_topic("living/lamp") == "home/living/lamp/set"
    assert device_id_from_topic("home/garage/smoke/state") == "garage/smoke"
    assert device_id_from_topic("home/alerts") is None


def test_state_event_survives_json():
    event = StateEvent("living/lamp", "lamp", "dim", "2026-10-12T12:00:00", "sim")
    assert StateEvent.from_payload(decode(encode(event.to_payload()))) == event


def test_wildcards_match_like_mqtt(bus):
    got = []
    bus.subscribe("home/+/+/state", lambda t, p: got.append(t))
    bus.subscribe("home/#", lambda t, p: got.append("all:" + t))
    bus.publish("home/living/lamp/state", {"value": "on"})
    bus.publish("home/alerts", {"kind": "x"})
    assert got == ["home/living/lamp/state", "all:home/living/lamp/state", "all:home/alerts"]


def test_messages_published_by_handlers_are_queued_not_nested(bus):
    order = []

    def first(topic, payload):
        order.append("first-start")
        bus.publish("b", {})
        order.append("first-end")

    bus.subscribe("a", first)
    bus.subscribe("b", lambda t, p: order.append("second"))
    bus.publish("a", {})
    assert order == ["first-start", "first-end", "second"]
