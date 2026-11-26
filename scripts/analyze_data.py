"""The datasets, described: what they contain, how complete they are, and what the habits look like.

    .venv\\Scripts\\python.exe scripts\\analyze_data.py

Reads the prepared tables of REFIT and UK-DALE (data/processed/) and the CASAS
preparation log from Kaggle (data/results/kaggle_logs/), and writes one page,
data/results/data_analysis.html (charts as SVG, hover a bar or cell for its value), and
the same page as data/results/data_analysis.pdf (printed by a headless Chromium). Nothing is trained; it takes a few seconds.
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy.data import MAX_LOAD_W  # noqa: E402
from habits.dataset import covered_days  # noqa: E402
from habits.evaluate import appliance_type  # noqa: E402

OUT_HTML = ROOT / "data" / "results" / "data_analysis.html"
OUT_PDF = ROOT / "data" / "results" / "data_analysis.pdf"
CASAS_LOG = ROOT / "data" / "results" / "kaggle_logs" / "gad_comparison_2" / "casas_prepare.log"
CASAS_SUMMARY = ROOT / "data" / "results" / "casas" / "casas_summary.json"  # scripts/analyze_casas.py, on Kaggle
DATASETS = {"refit": ("REFIT", "Loughborough area, UK", "8 s"), "ukdale": ("UK-DALE", "London, UK", "6 s")}
PVGIS = {"Loughborough (REFIT)": ROOT / "data" / "raw" / "pvgis" / "loughborough_2013_2015_1kwp.json",
         "London (UK-DALE)": ROOT / "data" / "raw" / "pvgis" / "london_2012_2017_1kwp.json"}
PV_KWP = 3.0  # the panel size the energy results assume (scripts/eval_suggestions.py)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Palette (dataviz skill reference instance, validated): two categorical slots and a blue ramp.
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#f2f6fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def esc(s) -> str:
    return html.escape(str(s))


def nice_ticks(vmax: float, n: int = 4) -> tuple[list[float], float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k) from 0 up to at least vmax, and the axis maximum."""
    raw = vmax / n
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    top = step * np.ceil(vmax / step)
    return list(np.arange(0, top + step / 2, step)), float(top)


def load(ds: str):
    d = ROOT / "data" / "processed" / ds
    uses, covered, apps, agg = (pd.read_parquet(d / f"{n}.parquet")
                                for n in ("uses", "covered_hours", "appliances", "aggregate_hourly"))
    apps["type"] = apps["appliance"].map(appliance_type)
    uses = uses.merge(apps[["house", "column", "category", "type"]], on=["house", "column"], how="left")
    return uses, covered, apps, agg


def per_house(uses, covered, apps, agg) -> pd.DataFrame:
    rows = []
    for house, hours in covered.groupby("house")["hour"]:
        hours = pd.DatetimeIndex(hours)
        span = pd.date_range(hours.min().floor("D"), hours.max().ceil("D"), freq="h", inclusive="left")
        load = agg.loc[agg["house"] == house, "aggregate_w"]
        a = apps[apps["house"] == house]
        rows.append({"house": house, "from": hours.min().date(), "to": hours.max().date(),
                     "days": len(covered_days(hours)), "coverage": len(hours) / len(span),
                     "habit": int((a["category"] == "habit").sum()), "shiftable": int((a["category"] == "shiftable").sum()),
                     "automatic": int((a["category"] == "automatic").sum()),
                     "uses": int(uses[(uses["house"] == house) & (uses["category"] != "automatic")].shape[0]),
                     "mean_w": load[load <= MAX_LOAD_W].mean(), "median_w": load[load <= MAX_LOAD_W].median(),
                     "glitch_hours": int((load > MAX_LOAD_W).sum())})
    return pd.DataFrame(rows)


def pvgis(path: Path) -> tuple[pd.DataFrame, dict]:
    """Hourly PVGIS rows in local time (PV output of 1 kWp in W, irradiance, temperature, wind) and the request."""
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["outputs"]["hourly"]
    utc = pd.to_datetime([r["time"] for r in rows], format="%Y%m%d:%H%M", utc=True).floor("h")
    df = pd.DataFrame({"pv_w": [r["P"] for r in rows], "irradiance": [r["G(i)"] for r in rows],
                       "temperature": [r["T2m"] for r in rows], "wind": [r["WS10m"] for r in rows]},
                      index=utc.tz_convert("Europe/London").tz_localize(None))
    return df.groupby(level=0).mean(), d["inputs"]


def casas_table() -> pd.DataFrame:
    pat = re.compile(r"(hh\d+):\s+(\d+) events, (\S+) to (\S+), (\d+) recorded days,\s+(\d+) intervals \((\d+) labelled\), (\d+) activities")
    rows = [m.groups() for m in map(pat.search, CASAS_LOG.read_text(encoding="utf-8").splitlines()) if m]
    df = pd.DataFrame(rows, columns=["home", "events", "from", "to", "days", "intervals", "labelled", "activities"])
    for c in ("events", "days", "intervals", "labelled", "activities"):
        df[c] = df[c].astype(int)
    return df


# --- SVG charts ---------------------------------------------------------------------------------

def svg(w, h, body, label) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="{esc(label)}" '
            f'style="max-width:100%;height:auto;font-family:Arial,sans-serif">{body}</svg>')


def hbars(items, unit, label, width=640, fmt="{:.0f}", colors=None, x_max=None) -> str:
    """Horizontal bars: items = [(label, value)], one row each, value printed at the bar end."""
    left, row, right = max(150, 12 + 7 * max(len(n) for n, _ in items)), 22, 60
    plot = width - left - right
    ticks, vmax = nice_ticks(x_max or max(v for _, v in items) or 1)
    h = row * len(items) + 30
    out = []
    for t in ticks:
        x = left + plot * t / vmax
        out.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{h - 22}" stroke="{GRID}"/>'
                   f'<text x="{x:.1f}" y="{h - 6}" font-size="11" fill="{MUTED}" text-anchor="middle">'
                   f'{fmt.format(t)}{unit}</text>')
    for i, (name, v) in enumerate(items):
        y = 6 + i * row
        wbar = max(plot * v / vmax, 1)
        color = (colors or {}).get(name, BLUE)
        out.append(f'<text x="{left - 8}" y="{y + 13}" font-size="12" fill="{INK2}" text-anchor="end">{esc(name)}</text>'
                   f'<rect x="{left}" y="{y + 2}" width="{wbar:.1f}" height="{row - 6}" rx="3" fill="{color}">'
                   f'<title>{esc(name)}: {fmt.format(v)}{unit}</title></rect>'
                   f'<text x="{left + wbar + 6:.1f}" y="{y + 13}" font-size="11" fill="{INK}">{fmt.format(v)}{unit}</text>')
    return svg(width, h, "".join(out), label)


def vbars(edges, shares, label, unit_x, marks=(), width=640, height=230, last_open=True) -> str:
    """A histogram: shares per bin (0..1), bin edges in x units; vertical marker lines with labels."""
    left, bottom, top = 46, 40, 18
    plot_w, plot_h = width - left - 16, height - bottom - top
    n = len(shares)
    bw = plot_w / n
    ticks, smax = nice_ticks(max(shares))
    out = []
    for t in ticks:
        y = top + plot_h * (1 - t / smax)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" stroke="{GRID}"/>'
                   f'<text x="{left - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{t:.0%}</text>')
    for i, s in enumerate(shares):
        x = left + i * bw
        hgt = plot_h * s / smax
        lo, hi = edges[i], edges[i + 1]
        name = f"{lo:,.0f}+ {unit_x}" if (last_open and i == n - 1) else f"{lo:,.0f}-{hi:,.0f} {unit_x}"
        out.append(f'<rect x="{x + 1:.1f}" y="{top + plot_h - hgt:.1f}" width="{bw - 2:.1f}" height="{max(hgt, 0.5):.1f}" '
                   f'rx="2" fill="{BLUE}"><title>{esc(name)}: {s:.1%} of hours</title></rect>')
    step = max(1, n // 6)
    for i in range(0, n + 1, step):
        if i < len(edges):
            x = left + i * bw
            txt = f"{edges[i]:,.0f}" + ("+" if last_open and i == n - 1 else "")
            out.append(f'<text x="{x:.1f}" y="{top + plot_h + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{txt}</text>')
    out.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 4}" font-size="11" fill="{INK2}" text-anchor="middle">{esc(unit_x)}</text>')
    for k, (value, text) in enumerate(marks):
        x = left + plot_w * (value - edges[0]) / (edges[-1] - edges[0])
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{ORANGE}" stroke-width="2"'
                   f' stroke-dasharray="{"" if k == 0 else "4 3"}"/>'
                   f'<text x="{x + 5:.1f}" y="{top + 12 + 14 * k}" font-size="11" fill="{INK}">{esc(text)}</text>')
    return svg(width, height, "".join(out), label)


def heatmaps(grids: dict, label: str, width=640, what="switch-ons") -> str:
    """Small multiples of weekday x hour heatmaps, each scaled to its own maximum (blue ramp)."""
    cell_w, cell_h, left, gap = (width - 40) / 24, 13, 40, 34
    block = 7 * cell_h + gap
    out = []
    for b, (name, g) in enumerate(grids.items()):
        y0 = 18 + b * block
        out.append(f'<text x="0" y="{y0 - 5}" font-size="12" font-weight="bold" fill="{INK}">{esc(name)}</text>')
        gmax = g.max() or 1
        for d in range(7):
            out.append(f'<text x="{left - 6}" y="{y0 + d * cell_h + 10}" font-size="10" fill="{MUTED}" text-anchor="end">{WEEKDAYS[d]}</text>')
            for hr in range(24):
                v = g[d, hr] / gmax
                color = RAMP[min(int(v * (len(RAMP) - 1) + 0.5), len(RAMP) - 1)]
                out.append(f'<rect x="{left + hr * cell_w:.1f}" y="{y0 + d * cell_h}" width="{cell_w - 1:.1f}" height="{cell_h - 1}" '
                           f'fill="{color}"><title>{esc(name)}, {WEEKDAYS[d]} {hr:02d}:00: {g[d, hr] / g.sum():.1%} of its {what}'
                           f'</title></rect>')
        for hr in range(0, 24, 3):
            out.append(f'<text x="{left + hr * cell_w + cell_w / 2:.1f}" y="{y0 + 7 * cell_h + 11}" font-size="10" '
                       f'fill="{MUTED}" text-anchor="middle">{hr:02d}</text>')
    h = 18 + len(grids) * block
    x0 = width - 120 - len(RAMP) * 20
    legend = "".join(f'<rect x="{x0 + i * 20}" y="{h - 4}" width="20" height="9" fill="{c}"/>' for i, c in enumerate(RAMP))
    legend += (f'<text x="{x0 - 6}" y="{h + 4}" font-size="10" fill="{MUTED}" text-anchor="end">fewer</text>'
               f'<text x="{x0 + len(RAMP) * 20 + 6}" y="{h + 4}" font-size="10" fill="{MUTED}">more {what}</text>')
    return svg(width, h + 12, "".join(out) + legend, label)


def lines(series: dict, label: str, width=640, height=230, unit="W", xlabels=None, every=3) -> str:
    """Lines over a short x axis (default: the 24 hours of the day); series = {name: (values, color)}, legend top left."""
    left, right, top, bottom = 46, 16, 34, 34
    pw, ph = width - left - right, height - top - bottom
    ticks, vmax = nice_ticks(max(max(v) for v, _ in series.values()))
    out = []
    x = left + 12
    for name, (_, color) in series.items():  # legend: its own row above the plot
        out.append(f'<line x1="{x}" y1="10" x2="{x + 18}" y2="10" stroke="{color}" stroke-width="2"/>'
                   f'<text x="{x + 24}" y="14" font-size="12" fill="{INK}">{esc(name)}</text>')
        x += 40 + 7 * len(name)
    for t in ticks:
        y = top + ph * (1 - t / vmax)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{GRID}"/>'
                   f'<text x="{left - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{t:.0f}</text>')
    xlabels = xlabels or [f"{hr:02d}:00" for hr in range(24)]
    last = len(xlabels) - 1
    for i in range(0, len(xlabels), every):
        out.append(f'<text x="{left + pw * i / last:.1f}" y="{top + ph + 16}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="middle">{esc(xlabels[i])}</text>')
    for name, (vals, color) in series.items():
        pts = " ".join(f"{left + pw * i / last:.1f},{top + ph * (1 - v / vmax):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{left + pw * i / last:.1f}" cy="{top + ph * (1 - v / vmax):.1f}" r="6" fill="transparent">'
                       f'<title>{esc(name)}, {esc(xlabels[i])}: {v:,.0f} {unit}</title></circle>')
    out.append(f'<text x="{left - 6}" y="{top - 10}" font-size="11" fill="{INK2}" text-anchor="end">{unit}</text>')
    return svg(width, height, "".join(out), label)


# --- the page -------------------------------------------------------------------------------------

def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(fmt[c](v) if c in fmt else v)}</td>" for c, v in r.items()) + "</tr>"
                   for _, r in df.iterrows())
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def figure(n: int, title: str, chart: str, note: str) -> str:
    return f'<figure><figcaption><b>Figure {n}.</b> {esc(title)}</figcaption>{chart}<p class="note">{note}</p></figure>'


def main() -> None:
    data = {ds: load(ds) for ds in DATASETS}
    houses = {ds: per_house(*data[ds]) for ds in DATASETS}
    casas = casas_table()

    # Overview table.
    overview = []
    for ds, (name, place, rate) in DATASETS.items():
        h, (uses, _, apps, _) = houses[ds], data[ds]
        overview.append({"Dataset": name, "Homes": len(h), "Place": place, "Period": f"{h['from'].min():%b %Y} - {h['to'].max():%b %Y}",
                         "Readings": f"every {rate}, power (W)", "Days with data": f"{h['days'].sum():,}",
                         "Appliances": f"{len(apps)} ({(apps['category'] != 'automatic').sum()} person-operated)",
                         "Uses found": f"{h['uses'].sum():,}"})
    solar = {name: pvgis(path) for name, path in PVGIS.items()}
    years = [f"{df.index.year.min()}-{df.index.year.max()}" for df, _ in solar.values()]
    overview.append({"Dataset": "PVGIS (EU JRC)", "Homes": "-", "Place": "Loughborough; London",
                     "Period": f"{years[0]}; {years[1]}", "Readings": "hourly solar output and weather (satellite + ERA5)",
                     "Days with data": f"{sum(len(set(df.index.date)) for df, _ in solar.values()):,}", "Appliances": "-",
                     "Uses found": "-"})
    overview.append({"Dataset": "CASAS", "Homes": len(casas), "Place": "USA (CASAS project, Washington State University)",
                     "Period": f"{pd.Timestamp(casas['from'].min()):%b %Y} - {pd.Timestamp(casas['to'].max()):%b %Y}",
                     "Readings": "motion and door sensor events", "Days with data": f"{casas['days'].sum():,}",
                     "Appliances": f"{casas['activities'].min()}-{casas['activities'].max()} labelled activities per home",
                     "Uses found": f"{casas['events'].sum():,} events"})

    # Figure 1: coverage.
    cov = []
    for ds in DATASETS:
        for r in houses[ds].itertuples():
            cov.append((f"{DATASETS[ds][0]} {r.house}", r.coverage * 100))
    fig1 = hbars(cov, "%", "Share of hours with data per home", fmt="{:.0f}", x_max=100,
                 colors={n: (ORANGE if n.startswith("UK-DALE") else BLUE) for n, _ in cov})

    # Figure 2: when appliances are switched on (REFIT).
    uses = data["refit"][0]
    grids = {}
    for t, title in (("kettle", "Kettle"), ("television", "Television"), ("washing machine", "Washing machine"),
                     ("dishwasher", "Dishwasher")):
        s = uses.loc[uses["type"] == t, "start"]
        g = np.zeros((7, 24))
        np.add.at(g, (s.dt.dayofweek.to_numpy(), s.dt.hour.to_numpy()), 1)
        grids[f"{title} ({uses.loc[uses['type'] == t, 'house'].nunique()} homes, {len(s):,} switch-ons)"] = g
    fig2 = heatmaps(grids, "When appliances are switched on, by weekday and hour")

    # Figures 3-4: the hourly load, and how much it changes from one hour to the next (REFIT).
    agg = data["refit"][3].sort_values(["house", "hour"])
    agg = agg[agg["aggregate_w"] <= MAX_LOAD_W]
    w = agg["aggregate_w"].to_numpy()
    edges = list(range(0, 3001, 150)) + [MAX_LOAD_W]
    counts, _ = np.histogram(w, bins=edges)
    shares = counts / counts.sum()
    mean_w, median_w = float(w.mean()), float(np.median(w))
    fig3 = vbars(np.array(edges[:-1] + [3150]), shares, "Distribution of a home's hourly load (REFIT)", "W, hourly mean",
                 marks=[(median_w, f"median {median_w:.0f} W"), (mean_w, f"mean {mean_w:.0f} W")])
    consecutive = agg.groupby("house")["hour"].diff() == pd.Timedelta(hours=1)
    change = agg.groupby("house")["aggregate_w"].diff()[consecutive].abs().to_numpy()
    edges4 = list(range(0, 1501, 75)) + [MAX_LOAD_W]
    c4, _ = np.histogram(change, bins=edges4)
    fig4 = vbars(np.array(edges4[:-1] + [1575]), c4 / c4.sum(), "Change from one hour to the next (REFIT)",
                 "W, |this hour - last hour|", marks=[(float(np.median(change)), f"median {np.median(change):.0f} W")])

    # Figure 5: a typical day, weekdays vs weekends (REFIT).
    hours = pd.DatetimeIndex(agg["hour"])
    weekend = hours.dayofweek >= 5
    prof = {"Weekdays": (agg[~weekend].groupby(hours[~weekend].hour)["aggregate_w"].median().to_numpy(), BLUE),
            "Weekends": (agg[weekend].groupby(hours[weekend].hour)["aggregate_w"].median().to_numpy(), ORANGE)}
    fig5 = lines(prof, "Median load by hour of the day, weekdays and weekends (REFIT)")

    # Figures 6-7: how often each kind of appliance is used, and its standby draw (REFIT).
    apps = data["refit"][2]
    person = apps[apps["category"] != "automatic"]
    kinds = person.groupby("type").filter(lambda g: len(g) >= 4).groupby("type")
    per_day = kinds["uses_per_day"].median().sort_values(ascending=False)
    fig6 = hbars([(f"{t} ({kinds.size()[t]})", v) for t, v in per_day.items()], "", "Uses per day by appliance type",
                 fmt="{:.1f}")
    standby = kinds["standby_w"].median().sort_values(ascending=False)
    fig7 = hbars([(f"{t} ({kinds.size()[t]})", v) for t, v in standby.items()], " W", "Standby power by appliance type",
                 fmt="{:.1f}")
    fridges = apps[apps["category"] == "automatic"]["uses_per_day"].median()

    # Cleaning.
    cleaning = []
    for ds in DATASETS:
        h, (_, covered, apps_ds, agg_ds) = houses[ds], data[ds]
        cleaning.append({"Dataset": DATASETS[ds][0],
                         "Hours kept (>= 300 readings: 40 min REFIT, 30 min UK-DALE)": f"{len(covered):,}",
                         "Hourly totals over 15 kW (logging glitches, dropped)": int(h["glitch_hours"].sum()),
                         "Self-running appliances left out (fridges, freezers, pumps)": int((apps_ds["category"] == "automatic").sum()),
                         "Person-operated appliances analysed": int((apps_ds["category"] != "automatic").sum())})

    pct = lambda v: f"{v:.0%}"  # noqa: E731
    watts = lambda v: f"{v:.0f}"  # noqa: E731
    keep = ["house", "from", "to", "days", "coverage", "habit", "shiftable", "automatic", "uses", "mean_w", "median_w"]
    names = ["House", "From", "To", "Days", "Hours with data", "Habit appl.", "Machines", "Self-running",
             "Uses", "Mean W", "Median W"]
    refit_h, ukdale_h = (houses[ds][keep].set_axis(names, axis=1) for ds in ("refit", "ukdale"))
    casas_v = casas.assign(share=casas["labelled"] / casas["intervals"])[
        ["home", "from", "to", "days", "events", "intervals", "share", "activities"]]
    casas_v.columns = ["Home", "From", "To", "Days", "Sensor events", "Sensor intervals", "Labelled", "Activities"]
    fmt = {"Hours with data": pct, "Mean W": watts, "Median W": watts, "Labelled": pct,
           "Uses": "{:,}".format, "Sensor events": "{:,}".format, "Sensor intervals": "{:,}".format}

    numbers = iter(range(1, 100))

    def fig(title, chart, note):
        return figure(next(numbers), title, chart, note)

    def casas_section() -> str:
        """The CASAS section, built where it stands in the page (so its figures are numbered in order)."""
        if not CASAS_SUMMARY.exists():
            return ""
        cs = json.loads(CASAS_SUMMARY.read_text(encoding="utf-8"))
        cgrids = {f"{a.replace('_', ' ')} ({v['homes']} homes, {v['starts']:,} starts)": np.array(v["grid"])
                  for a, v in cs["starts"].items()}
        cday_all = sorted(cs["per_day"].items(), key=lambda kv: -kv[1])
        cday = cday_all[:15]
        same = pd.DataFrame(cs["same_label"])
        heat = fig("When the most common activities start, by weekday and hour, all homes (each map scaled to its own maximum)",
                   heatmaps(cgrids, "CASAS activity starts by weekday and hour", what="starts"),
                   "The same picture as the appliances: dressing peaks around 7-8 and 21-22, TV in the evening, leaving home "
                   "during the day, sleep around 22-23. Sleep also \"starts\" through the night: the label begins again "
                   "after each night-time toilet visit. Clock-bound routines are what the habit models learn.")
        bars = fig(f"How often each activity starts per recorded day: the {len(cday)} most frequent of the {len(cday_all)} seen in at least 10 homes "
                   f"(median over the homes that have it; homes in brackets)",
                   hbars([(f"{a.replace('_', ' ')} ({cs['per_day_homes'][a]})", v) for a, v in cday], "",
                         "CASAS activity starts per day", fmt="{:.1f}"),
                   "Counted when an activity's label begins, after the three noisy classes (Other, Entertain guests, "
                   "Enter home) are removed.")
        casas_block = f"""
<h2>5. The followed article's data: activities in CASAS</h2>
{heat}
{bars}
<p><b>Why the "previous activity" rule is so strong.</b> In the median home, {same['same_as_previous'].median():.0%}
of labelled sensor intervals have the same activity as the interval before them (range {same['same_as_previous'].min():.0%}
to {same['same_as_previous'].max():.0%} over {len(same)} homes); an activity typically lasts {same['median_run'].median():.0f}
intervals in a row. So repeating the previous label is right most of the time (the rule scored 0.82 accuracy on Gad et
al.'s split and 0.88 on the strict one), and a random split puts the neighbours of almost every test interval into
training: the data behind chapter 9's result.</p>
"""

        return casas_block

    def solar_section(number: int) -> str:
        rows, monthly, colors = [], {}, [BLUE, ORANGE]
        for (name, (df, inputs)), color in zip(solar.items(), colors):
            yearly = df["pv_w"].groupby(df.index.year).sum() / 1000  # kWh per kWp
            full = yearly[df["pv_w"].groupby(df.index.year).size() >= 8700]
            loc, mount = inputs["location"], inputs["mounting_system"]["fixed"]
            rows.append({"Place": name, "Lat, lon": f"{loc['latitude']:.2f}, {loc['longitude']:.2f}",
                         "Years": f"{df.index.year.min()}-{df.index.year.max()}",
                         "Panel": f"1 kWp c-Si, {mount['slope']['value']} deg tilt (optimal), {inputs['pv_module']['system_loss']:.0f}% losses",
                         "Yearly yield": f"{full.min():.0f}-{full.max():.0f} kWh per kWp",
                         "Temperature": f"{df['temperature'].min():.0f} to {df['temperature'].max():.0f} C (mean {df['temperature'].mean():.1f})",
                         "Wind, mean": f"{df['wind'].mean():.1f} m/s"})
            m = df["pv_w"].groupby([df.index.year, df.index.month]).sum() * PV_KWP / 1000
            monthly[name] = (m.groupby(level=1).mean().reindex(range(1, 13)).to_numpy(), color)
        lough = solar["Loughborough (REFIT)"][0]
        june = lough[lough.index.month == 6]["pv_w"] * PV_KWP
        dec = lough[lough.index.month == 12]["pv_w"] * PV_KWP
        day = {"June": (june.groupby(june.index.hour).mean().reindex(range(24)).to_numpy(), BLUE),
               "December": (dec.groupby(dec.index.hour).mean().reindex(range(24)).to_numpy(), ORANGE)}
        daily = lough["pv_w"].groupby(lough.index.date).sum() * PV_KWP / 1000
        dmonth = pd.Series(pd.to_datetime(daily.index).month, index=daily.index)
        jun_days, dec_days = daily[dmonth.to_numpy() == 6], daily[dmonth.to_numpy() == 12]
        m_fig = fig(f"Solar energy per month from a {PV_KWP:g} kWp panel (mean over the years)",
                    lines(monthly, "Monthly solar energy", unit="kWh", xlabels=MONTHS, every=1),
                    f"About {monthly['Loughborough (REFIT)'][0][5] / max(monthly['Loughborough (REFIT)'][0][11], 0.1):.0f} times more "
                    f"in June than in December. The homes had no panels: the energy results are a \"what if\" with this "
                    f"real output for their town and days, which is why smart start helps much more in summer.")
        d_fig = fig(f"Average solar power through the day, Loughborough, {PV_KWP:g} kWp: June and December",
                    lines(day, "Solar power by hour"),
                    f"A June day gives {jun_days.median():.1f} kWh (median; {jun_days.quantile(0.1):.1f}-{jun_days.quantile(0.9):.1f} "
                    f"in 8 days of 10), a December day {dec_days.median():.1f} kWh ({dec_days.quantile(0.1):.1f}-"
                    f"{dec_days.quantile(0.9):.1f}). Cloudy days are in the data: PVGIS gives the real weather of each hour, "
                    f"not a clear-sky curve, so smart start is scored on the sun the homes really had.")
        return f"""
<h2>{number}. Solar and weather (PVGIS)</h2>
<p>PVGIS (the European Commission's Joint Research Centre) gives, for any place and year, the hourly output of a solar
panel, computed from satellite irradiance (SARAH3), and the weather (ERA5). It supplies the solar power for smart
start and the standby estimates, and the temperature, irradiance and wind inputs of the load forecasts.</p>
{table(pd.DataFrame(rows))}
{m_fig}
{d_fig}
"""

    small_change = float((change < 100).mean())
    big_change = float((change >= 1000).mean())
    under_500 = float((w < 500).mean())
    body = f"""
<h1>Dataset analysis</h1>
<p class="lead">Three public datasets, prepared into the same tables: REFIT and UK-DALE (electricity per appliance in
UK homes) for the system and its energy results, and CASAS (labelled activities from motion sensors), the data of the
followed article, Gad et al. (2025); plus PVGIS, the real solar output and weather of the homes' towns. Generated by
<code>scripts/analyze_data.py</code> from the prepared data.</p>

<h2>1. What the data is</h2>
{table(pd.DataFrame(overview))}
<p class="note">"Uses" are the times a person switched an appliance on, found from its power readings: a use starts
when power rises above max(15 W, 3 x its idle level + 10 W), and pauses under 30 minutes (a washing machine filling)
are joined into one use. CASAS events are sensor messages; an interval is one sensor from ON to OFF (at least 2 s).</p>

<h2>2. How the data was cleaned</h2>
{table(pd.DataFrame(cleaning))}
{fig("Share of each home's recording period with data (an hour counts with at least 300 readings: 40 of its 60 minutes in REFIT, 30 in UK-DALE)", fig1,
        f"REFIT homes have data in {houses['refit']['coverage'].min():.0%}-{houses['refit']['coverage'].max():.0%} of their hours "
        f"(median {houses['refit']['coverage'].median():.0%}); the gaps are logger outages, kept out of the analysis rather "
        f"than read as \"nothing used\". UK-DALE house 1 runs for about four years; the other four for months.")}

<h2>3. The homes</h2>
<h3>REFIT, 20 homes</h3>
{table(refit_h, fmt)}
<h3>UK-DALE, 5 homes</h3>
{table(ukdale_h, fmt)}
<h3>CASAS, 21 single-resident homes (the followed article's data)</h3>
{table(casas_v, fmt)}
<p class="note">CASAS homes were inferred from the article's Table 4 sizes (hh107 and hh121 have two residents and are
left out). "Labelled" is the share of sensor intervals that carry an activity label.</p>

<h2>4. Habits exist: when appliances are switched on</h2>
{fig("Switch-ons by weekday and hour, all REFIT homes with that appliance (each map scaled to its own maximum)", fig2,
        "The kettle has sharp morning peaks, the TV a long evening band and later weekend mornings; washing machines and "
        "dishwashers run at many different times. This is why the system automates the regular habits, cuts standby in the "
        "quiet hours, and lets machines be scheduled (smart start) rather than predicted.")}
{fig("How often each kind of appliance is used (median over the homes that have it; number of appliances in brackets)", fig6,
        f"A kettle is used about {per_day['kettle']:.0f} times a day and a TV about {per_day['television']:.0f}; washing "
        f"machines and dishwashers run every day or two. Fridges and freezers, left out, switch on about {fridges:.0f} "
        f"times a day by themselves.")}
{fig("Standby power while not in use (median over homes)", fig7,
        f"Computers ({standby['computer']:.1f} W) and televisions ({standby['television']:.1f} W) draw power around the "
        f"clock while unused: the target of the standby cuts. Kettles draw nothing, so the system suggests nothing for them.")}

{casas_section()}
<h2>{6 if CASAS_SUMMARY.exists() else 5}. The load of a home: why simple forecasts are hard to beat</h2>
{fig("Distribution of a home's hourly load, all REFIT homes (last bar: 3 kW and more)", fig3,
        f"Skewed: {under_500:.0%} of hours are under 500 W, with a long tail of cooking and heating hours. The mean "
        f"({mean_w:.0f} W) sits well above the typical hour (median {median_w:.0f} W). A model trained on squared error is "
        f"pulled towards the mean; one trained on absolute error predicts the typical hour, which is why it forecasts "
        f"better here.")}
{fig("Change in load from one hour to the next (last bar: 1.5 kW and more)", fig4,
        f"In {small_change:.0%} of hours the load changes by less than 100 W, which is why \"same as last hour\" is a strong "
        f"forecast; in {big_change:.0%} it changes by 1 kW or more, up or down, and those swings are what no forecaster sees coming.")}
{fig("Median load by hour of the day", fig5,
        "Low at night, a morning rise, and an evening peak around 18:00-20:00; weekends rise later and stay higher during "
        "the day. The clock explains much of the load, which is why the hour is the feature every model keeps.")}
{solar_section(7 if CASAS_SUMMARY.exists() else 6)}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset analysis</title>
<style>
:root {{ --surface:#fcfcfb; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; }}
body {{ background:var(--surface); color:var(--ink); font:14px/1.45 Arial, sans-serif; margin:0 auto; padding:24px 16px;
        max-width:880px; }}
h1 {{ font-size:24px; margin:0 0 6px; }} h2 {{ font-size:18px; margin:28px 0 8px; }} h3 {{ font-size:14px; margin:16px 0 6px; }}
.lead {{ color:var(--ink2); }} .note {{ color:var(--ink2); font-size:12.5px; margin:6px 0 0; }}
table {{ border-collapse:collapse; width:100%; font-size:11.5px; margin:6px 0; }}
th, td {{ border-bottom:1px solid var(--grid); padding:3px 6px; text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--ink2); font-weight:600; }}
figure {{ margin:18px 0; break-inside:avoid; }} figcaption {{ font-size:13px; margin-bottom:6px; }}
code {{ font-size:12.5px; }}
@page {{ size:A4; margin:14mm; }}
@media print {{ body {{ padding:0; max-width:none; }} h2 {{ break-after:avoid; }} table {{ break-inside:avoid; }} }}
</style></head><body>{body}</body></html>"""
    OUT_HTML.write_text(page, encoding="utf-8")
    print(OUT_HTML.relative_to(ROOT))
    exe = next((b for b in [*sorted(Path.home().glob(".cache/puppeteer/chrome-headless-shell/*/*/chrome-headless-shell.exe"),
                                    reverse=True),
                            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")] if b.exists()), None)
    if exe:
        subprocess.run([str(exe), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                        OUT_HTML.as_uri()], capture_output=True, timeout=120)
        print(OUT_PDF.relative_to(ROOT) if OUT_PDF.exists() else "PDF not written")


if __name__ == "__main__":
    main()
