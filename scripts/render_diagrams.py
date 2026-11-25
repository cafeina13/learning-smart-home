"""Render every Mermaid diagram in docs/diagrams.md to a PNG in docs/figures/, for the report.

    .venv\\Scripts\\python.exe scripts\\render_diagrams.py

Uses a headless Chromium already on the machine (Puppeteer's chrome-headless-shell,
or Microsoft Edge) and the Mermaid library (2.6 MB, fetched once into data/tools/).
Each diagram is drawn twice: once to read its size, once to screenshot it at that
size, at 2x for print. It also reports how large the text prints on an A4
page, and flags figures whose text would be under 8 pt.
"""

import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "diagrams.md"
OUT = ROOT / "docs" / "figures"
MERMAID = ROOT / "data" / "tools" / "mermaid.min.js"
MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
BROWSERS = [
    *sorted(Path.home().glob(".cache/puppeteer/chrome-headless-shell/*/*/chrome-headless-shell.exe"), reverse=True),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
SCALE = 2
PAGE_CM = (16.0, 22.0)  # room for a figure on an A4 page with margins
FONT_PX = 16  # Mermaid's text size
MIN_PT = 8  # smallest text that reads comfortably in print


def printed_text_pt(w: int, h: int) -> float:
    """Text size in points once a w x h px figure is shrunk to fit the page."""
    cm_per_px = min(PAGE_CM[0] / w, PAGE_CM[1] / h)
    return FONT_PX * cm_per_px / 0.03528

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;background:#fff}} #d{{display:inline-block;padding:16px 16px 32px}}</style>
<script src="{mermaid}"></script></head><body><div id="d"></div><script>
mermaid.initialize({{startOnLoad: false, theme: "neutral", fontFamily: "Arial, sans-serif",
                    flowchart: {{htmlLabels: true}}, sequence: {{mirrorActors: false}},
                    securityLevel: "loose"}});
mermaid.render("g", {code}).then(r => {{
  const d = document.getElementById("d"); d.innerHTML = r.svg;
  const svg = d.querySelector("svg"), vb = svg.viewBox.baseVal;  // natural size, not shrunk to the window
  svg.style.maxWidth = "none"; svg.setAttribute("width", vb.width); svg.setAttribute("height", vb.height);
  const b = d.getBoundingClientRect();
  document.title = Math.ceil(b.width) + "x" + Math.ceil(b.height);
}});
</script></body></html>"""


def diagrams() -> list[tuple[str, str]]:
    """(file name, Mermaid code) for each block, named after its section: '1-system-architecture'."""
    blocks, section, count = [], "diagram", {}
    for part in re.split(r"(^## [^\n]*$|```mermaid\n.*?```)", SOURCE.read_text(encoding="utf-8"), flags=re.M | re.S):
        if part.startswith("## "):
            section = re.sub(r"[^a-z0-9]+", "-", part[3:].lower()).strip("-")
        elif part.startswith("```mermaid"):
            count[section] = count.get(section, 0) + 1
            blocks.append((section, part[len("```mermaid\n"):-3]))
    seen, named = {}, []
    for section, code in blocks:  # a section with two diagrams gets -a and -b
        seen[section] = seen.get(section, 0) + 1
        named.append((section + (f"-{'abc'[seen[section] - 1]}" if count[section] > 1 else ""), code))
    return named


def browser() -> Path:
    found = next((b for b in BROWSERS if b.exists()), None)
    if found is None:
        raise SystemExit("no headless Chromium found (Puppeteer's chrome-headless-shell or Edge)")
    return found


def run(exe: Path, page: Path, *args: str) -> str:
    cmd = [str(exe), "--headless", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
           "--virtual-time-budget=10000", *args, page.as_uri()]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120).stdout


def main() -> None:
    if not MERMAID.exists():
        MERMAID.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MERMAID_URL, MERMAID)
    exe = browser()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, code in diagrams():
            page = Path(tmp) / f"{name}.html"
            js_code = "`" + code.replace("\\", "\\\\").replace("`", "\\`") + "`"
            page.write_text(PAGE.format(mermaid=MERMAID.as_uri(), code=js_code), encoding="utf-8")
            title = re.search(r"<title>(\d+)x(\d+)</title>", run(exe, page, "--dump-dom"))
            if not title:
                print(f"{name}: Mermaid could not draw it")
                continue
            w, h = int(title[1]), int(title[2])
            png = OUT / f"{name}.png"
            run(exe, page, f"--window-size={w},{h}", f"--force-device-scale-factor={SCALE}", f"--screenshot={png}")
            pt = printed_text_pt(w, h)
            print(f"{png.relative_to(ROOT)}  {w}x{h} px, text {pt:.1f} pt on A4"
                  + ("" if pt >= MIN_PT else f"  TOO SMALL (keep under ~900 x 1250 px)")
                  if png.exists() else f"{name}: no screenshot")


if __name__ == "__main__":
    main()
