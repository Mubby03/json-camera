"""Freeze the monitor into one self-contained HTML file.

The live page polls a server on the training machine, which is no use from a
phone on the other side of town.  This writes the same page with the status
baked in, so it can be published anywhere static and re-published on a timer.

    python3 monitor/snapshot.py --log out/train_sharp.log --out out/monitor.html
"""

import argparse
import json
import re
from pathlib import Path

from server import DEFAULT_REPO, build_status

HERE = Path(__file__).resolve().parent


def render(repo, log_path, note=""):
    status = build_status(repo, log_path)
    status["snapshot"] = True
    html = (HERE / "index.html").read_text()
    css = (HERE / "style.css").read_text()
    js = (HERE / "app.js").read_text()

    # The page is the body of index.html; the host supplies the document shell.
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S).group(1)
    palette = re.search(r'<body data-palette="([^"]+)"', html).group(1)
    body = body.replace('<script src="app.js"></script>', "")
    body = body.replace(
        "This page polls <code>/api/status</code> every 2s and never writes to the "
        "repository or signals the training process.",
        "This is a snapshot of the run log, re-published on a timer from the training "
        "machine. " + note)

    # One fetch becomes one embedded object; the poll loop then just re-renders it.
    js = js.replace(
        "const response = await fetch('/api/status', { cache: 'no-store' });\n"
        "    render(await response.json());",
        "render(window.__STATUS__);")
    assert "window.__STATUS__" in js, "app.js poll() changed shape; update snapshot.py"

    return (
        "<title>json-camera training</title>\n"
        f"<style>{css}</style>\n"
        f"<script>window.__STATUS__ = {json.dumps(status)};"
        f"document.body.dataset.palette = {json.dumps(palette)};</script>\n"
        f"{body}\n<script>{js}</script>\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(Path(args.repo), args.log, args.note))
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
