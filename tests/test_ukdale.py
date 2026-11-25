"""UK-DALE preparation on a tiny fake zip with UK-DALE's layout."""

import io
import zipfile

import numpy as np

from habits.ukdale import appliance_name, houses, labels, prepare_house, read_channel


def fake_zip() -> zipfile.ZipFile:
    start = 1_400_000_000  # May 2014
    t = np.arange(start, start + 3 * 86400, 6)  # three days, every 6 s
    hour = (t - start) // 3600 % 24
    mains = 200 + 100 * (hour >= 18)
    kettle = np.where(((t - start) % 86400 >= 7 * 3600) & ((t - start) % 86400 < 7 * 3600 + 180), 2000, 0)
    tv = np.where(hour >= 20, 90, 8)  # 8 W standby, on in the evening
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("house_7/labels.dat", "1 aggregate\n2 kettle\n3 tv\n4 dish_washer\n")
        for ch, w in ((1, mains + kettle + tv), (2, kettle), (3, tv), (4, np.zeros_like(t))):
            zf.writestr(f"house_7/channel_{ch}.dat", "\n".join(f"{a} {b}" for a, b in zip(t, w)))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_names_become_project_names():
    assert appliance_name("dish_washer") == "Dishwasher"
    assert appliance_name("tv") == "Television"
    assert appliance_name("washing_machine") == "Washing Machine"
    assert appliance_name("htpc") == "Htpc"


def test_layout_is_read():
    zf = fake_zip()
    assert houses(zf) == [7]
    assert labels(zf, 7) == {1: "aggregate", 2: "kettle", 3: "tv", 4: "dish_washer"}
    s = read_channel(zf, 7, 2)
    assert len(s) == 3 * 86400 // 6 and s.max() == 2000


def test_prepared_like_refit():
    uses, covered, rows, agg = prepare_house(fake_zip(), 7, log=lambda *a: None)
    assert set(uses.columns) >= {"house", "column", "appliance", "start", "end", "duration_s", "energy_wh"}
    kettle = uses[uses["appliance"] == "Kettle"]
    assert len(kettle) == 3  # one boil a day
    assert {r["appliance"] for r in rows} == {"Kettle", "Television", "Dishwasher"}
    tv = next(r for r in rows if r["appliance"] == "Television")
    assert tv["idle_level_w"] == 8 and tv["standby_w"] > 5  # the TV's standby is seen
    assert len(covered) >= 70 and agg["aggregate_w"].between(150, 2500).all()
