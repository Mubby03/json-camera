"""Push the run's status to the web app, so a phone can see it.

    python3 monitor/push.py --log out/train_sharp.log            # every 60s, forever
    python3 monitor/push.py --log out/train_sharp.log --once

Reads the same log the monitor reads, keeps the fields a small screen can use,
and POSTs them with the chat key. Standard library only, like the monitor.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from server import DEFAULT_REPO, build_status

KEY_FILE = Path.home() / ".jsoncam-chat-key"


def slim(status):
    """Only what the chat panel draws. The full monitor keeps the rest."""
    cfg = status.get("config") or {}
    keep = ("lmbda", "hidden", "latent", "batch", "lr", "epochs", "patch", "deadline",
            "steps_per_epoch", "train_patches", "val_patches", "device", "_inferred")
    rows = [{k: r.get(k) for k in ("epoch", "train_psnr", "train_bpp", "val_psnr",
                                   "val_bpp", "val_loss", "seconds", "best")}
            for r in status.get("epochs") or []]
    out = {
        "generated": status.get("generated"),
        "running": status.get("running"),
        "message": status.get("message"),
        "config": {k: cfg[k] for k in keep if k in cfg},
        "epochs": rows,
        "current": status.get("current"),
        "eta": status.get("eta"),
        "best": status.get("best"),
    }
    return out


STARTED = re.compile(r"^\[(\d\d:\d\d:\d\d)\] training until", re.M)


def run_started(log_path):
    """When the trainer was launched, from the run script's own stamp.

    The run ends on a deadline rather than an epoch count, so the phone's
    progress bar needs to know when the clock started.
    """
    if not log_path or not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    found = STARTED.search(text)
    if not found:
        return None
    day = time.localtime(Path(log_path).stat().st_ctime)
    stamp = time.strptime(f"{day.tm_year}-{day.tm_mon}-{day.tm_mday} {found[1]}", "%Y-%m-%d %H:%M:%S")
    return time.strftime("%Y-%m-%dT%H:%M:%S", stamp)


def push(url, key, status):
    body = urllib.parse.urlencode({"status": json.dumps(status), "key": key}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/training", data=body, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--log", default=None)
    ap.add_argument("--url", default="https://json-camera.fly.dev")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    key = KEY_FILE.read_text().strip()
    while True:
        try:
            status = slim(build_status(Path(args.repo), args.log))
            status["run_started"] = run_started(args.log)
            code = push(args.url, key, status)
            print(f"{time.strftime('%H:%M:%S')} pushed ({code})", flush=True)
        except (urllib.error.URLError, OSError) as e:
            # The machine sleeps when idle and takes half a minute to wake; the
            # next push will find it up. Say so and carry on.
            print(f"{time.strftime('%H:%M:%S')} push failed: {e}", file=sys.stderr, flush=True)
        if args.once:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
