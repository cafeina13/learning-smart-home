# The article's home rebuilt in Cisco Packet Tracer

Built by the student on 2026-10-11 in Packet Tracer 8 to show the project
follows Özdoğan & Daş (2021) in the tool they used. It rebuilds the article's
third scenario, the home around the IoT gateway. The article draws each
scenario as a separate topology; the first two (the wired temperature network
sending UDP to a server, and the smoke sensor publishing to a separate MQTT
broker) are reproduced in Python over a real broker instead
(`home/simulation.py`). Everything is in `docs/packet_tracer/`:

| File | What it shows |
| --- | --- |
| `article-home.pkt` | The finished home: open it in Packet Tracer |
| `1-home.png` | The home: Home Gateway, solar panel -> battery -> lamp and coffee maker, smoke detector, fan, door, window, garage door, old car, tablet |
| `2-low-energy.png` | The low-energy rule has fired: lamp dimmed, window open (taken before some devices were moved) |
| `3-smoke-high.png` | The car is smoking: door unlocked, garage door and window open, fan on |
| `4-smoke-clear.png` | Smoke gone: everything closed and locked again |
| `5-tablet-devices.png` | The gateway's device list, all nine things registered |
| `6-smoke-level.png` | The moment the smoke level passes the threshold (0.1035) |
| `rule-conflict-jitter.mp4` | Two rules fighting: the window flips open and closed |
| `smoke-scenario.mp4` | The smoke scenario from start to finish, with the detector's level |

Pictures 1-4 are aligned on the gateway, so flipping between them shows only
what changed.

## How it was built (and what was not obvious)

1. **Place the Home Gateway** (DLC100) and the things: solar panel, battery,
   light, appliance (coffee maker), ceiling fan, door, window, garage door,
   smoke detector, an old car, a tablet PC.
2. **Power each thing on** (its power button on the Physical tab or Alt+click).
   An unpowered thing has a mostly empty Config and no link.
3. **Wire the power**: solar panel -> battery -> light and coffee maker.
4. **Join each thing to the gateway's Wi-Fi** (Config > Wireless0, the
   gateway's SSID). If Config has no wireless settings, add the card first:
   Advanced > I/O Config > Network Adapter > PT-IOT-NM-1W.
5. **Register each thing with the IoT server**: Config > Settings > IoT Server
   = Home Gateway. Being on the Wi-Fi is not enough: without this the
   gateway's device list stays empty.
6. **The old car is not an IoT thing.** It has no network card and needs none:
   it only puts smoke into the simulated air, which the smoke detector measures.
7. **Tablet**: join the Wi-Fi, then Desktop > Web Browser > `192.168.25.1`,
   login admin / admin. Home lists the devices; Conditions holds the rules.

## The rules

| Rule | If | Then |
| --- | --- | --- |
| `low_energy` (the article's Fig. 11) | Match Any: battery available power < 20%; solar panel <= 1 Wh | light Dim; window On |
| `smoke_high` | smoke detector level > 0.1 | door Unlock; garage door On; window On; fan High |
| `smoke_clear` | smoke detector level < 0.02 | door Lock; garage door Off; window Off; fan Off |

Two differences from the article, both forced by Packet Tracer:

- **Smoke thresholds scaled.** The article uses 10 and 2. Packet Tracer's
  detector settles near 0.18 with one car running and never reaches 10, so
  the rules use 0.1 and 0.02: the same 5:1 ratio. Our Python home keeps 10
  and 2 (its smoke model runs 0-100).
- **The door is unlocked, not opened.** A rule can only lock or unlock a door;
  its "open" is a sensor reading (real homes have smart locks, not door
  motors). The window and garage door open as in the article.

To test: hold Alt and click the car to start it (Fast Forward Time speeds up
the smoke). Turn `low_energy` off first, or it fights `smoke_clear` over the
window.

## What it showed about the article's design

- **A rule only acts when something changes.** `low_energy` did nothing when
  created, although both conditions were already true; it fired at the next
  change.
- **The gateway keeps forcing a rule's state** while its condition holds: a
  person who changes the lamp or window by hand is overruled at once.
- **Two rules can fight.** With the battery low (`low_energy`: window open)
  and no smoke (`smoke_clear`: window closed), the window flips open and closed
  forever (`rule-conflict-jitter.mp4`).

Our rule engine acts once, when a condition becomes true; a person may change
the device afterwards, which counts as an override, and three overrides switch
the rule off with an alert. Two conflicting rules each act once and the device
settles (`tests/test_rules.py`, the conflicting-rules test).

## The same rule in our system

`low_energy` in our rule engine (`home/simulation.py`, `article_rules()`):

```json
{
  "name": "low_energy",
  "match": "any",
  "conditions": [
    {"device": "roof/solar", "op": "<=", "value": 1},
    {"device": "utility/battery", "op": "<", "value": 20}
  ],
  "actions": [
    {"device": "living/lamp", "value": "dim"},
    {"device": "living/window", "value": "open"}
  ],
  "origin": "manual"
}
```

Same structure as the gateway's editor: match any/all, conditions, actions.
What the project adds: rules like this can also be learned
(`"origin": "learned"`), every device enforces what a rule may do to it, and
the person can always overrule a rule.
