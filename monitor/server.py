#!/usr/bin/env python3
"""Live web view of a json-camera training run.

    python3 monitor/server.py

Then open http://localhost:8765.  Standard library only -- any Python 3 works,
and it does not need the project's .venv.

Adapted from the YOLO/VisDrone training monitor.  The layout, charts and ETA
logic carry over unchanged; what differs is the vocabulary.  A detector is
scored with mAP, precision and recall.  A codec has no classes to detect and no
boxes to rank, so there is nothing to average precision over -- it is scored on
*rate* (bits per pixel) against *distortion* (PSNR), and the whole point of the
run is the trade between them.  Showing a bpp under a tile labelled "mAP50"
would be worse than showing nothing, so the tiles say what they mean.

It cannot break the training run.  Two files are opened read-only -- the run log
and scripts/run_training.sh -- and nothing is ever written to the repository or
signalled to the training process.
"""

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent

# `  e4 550/2400  loss    1.528  bpp  0.371  psnr  25.76dB`
STEP = re.compile(
    r"^\s+e(?P<epoch>\d+)\s+(?P<it>\d+)/(?P<its>\d+)\s+"
    r"loss\s+(?P<loss>[-\d.]+)\s+bpp\s+(?P<bpp>[-\d.]+)\s+psnr\s+(?P<psnr>[-\d.]+|inf)dB"
)
# `epoch 3/10  loss    1.652  bpp  0.373  psnr  25.32dB  (1095s)`
EPOCH = re.compile(
    r"^epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"loss\s+(?P<loss>[-\d.]+)\s+bpp\s+(?P<bpp>[-\d.]+)\s+psnr\s+(?P<psnr>[-\d.]+|inf)dB"
    r"\s+\((?P<seconds>[\d.]+)s\)"
)
# `           val  loss    1.647  bpp  0.343  psnr  25.24dB`
VAL = re.compile(
    r"^\s+val\s+loss\s+(?P<loss>[-\d.]+)\s+bpp\s+(?P<bpp>[-\d.]+)\s+psnr\s+(?P<psnr>[-\d.]+|inf)dB"
)
BEST = re.compile(r"new best \((?P<basis>train|val) loss (?P<loss>[-\d.]+)\)")
HEADER_TRAIN = re.compile(r"^(?P<patches>\d+) patches,\s*(?P<steps>\d+) steps/epoch")
HEADER_VAL = re.compile(r"^(?P<patches>\d+) held-out patches")
DEVICE = re.compile(r"^device:\s*(?P<device>\S+)")
# Long-option pairs out of run_training.sh, e.g. `--lmbda 0.0067`.
SH_ARG = re.compile(r"--(?P<key>[a-z-]+)\s+(?P<value>[\w.\-/]+)")


def number(text):
    """Floats, but `inf` stays a string: JSON has no infinity and a PSNR of inf
    is a real state (a bit-exact reconstruction) rather than a parse failure."""
    if text == "inf":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_log(path, limit=8 << 20):
    """The whole log when it is small, otherwise the tail.

    json-camera logs about fifty lines an epoch, so even a very long run stays
    well inside the cap and every epoch row survives to be charted.
    """
    size = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if size > limit:
            handle.seek(size - limit)
            handle.readline()               # drop the partial first line
        return handle.read()


def parse_log(text):
    """-> (epoch rows, current step, run header)."""
    rows, current, header = [], None, {}
    pending = None                          # epoch row awaiting its val line

    for line in text.splitlines():
        found = HEADER_TRAIN.match(line)
        if found:
            header["train_patches"] = int(found["patches"])
            header["steps_per_epoch"] = int(found["steps"])
            continue
        found = HEADER_VAL.match(line)
        if found:
            header["val_patches"] = int(found["patches"])
            continue
        found = DEVICE.match(line)
        if found:
            header["device"] = found["device"]
            continue

        found = EPOCH.match(line)
        if found:
            pending = {
                "epoch": int(found["epoch"]),
                "train_loss": number(found["loss"]),
                "train_bpp": number(found["bpp"]),
                "train_psnr": number(found["psnr"]),
                "seconds": number(found["seconds"]),
            }
            header["epochs_total"] = int(found["epochs"])
            rows.append(pending)
            continue

        found = VAL.match(line)
        if found and pending is not None:
            pending["val_loss"] = number(found["loss"])
            pending["val_bpp"] = number(found["bpp"])
            pending["val_psnr"] = number(found["psnr"])
            continue

        found = BEST.search(line)
        if found and pending is not None:
            pending["best"] = True
            pending["best_basis"] = found["basis"]
            continue

        found = STEP.match(line)
        if found:
            current = {
                "epoch": int(found["epoch"]),
                "iteration": int(found["it"]),
                "iterations": int(found["its"]),
                "loss": number(found["loss"]),
                "bpp": number(found["bpp"]),
                "psnr": number(found["psnr"]),
            }

    # A step line from an epoch that has since finished is stale: the summary
    # is the newer fact, and leaving it would peg the bar at 100% between epochs.
    if current and rows and current["epoch"] <= rows[-1]["epoch"]:
        current = None
    return rows, current, header


def read_run_config(path):
    """Training arguments out of run_training.sh.

    The log records what the run is doing but not what it was asked to do, and
    lambda in particular is the single most important number here -- it is the
    quality knob, and the only thing separating a small file from a big one.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    body = text.split("jsoncam train", 1)[-1] if "jsoncam train" in text else ""
    config = {}
    for found in SH_ARG.finditer(body):
        config[found["key"].replace("-", "_")] = found["value"]
    return config


def training_pid():
    try:
        found = subprocess.run(["pgrep", "-f", "jsoncam train"],
                               capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in found.stdout.split():
        return int(line)
    return None


def checkpoints(repo):
    out = []
    for path in sorted(repo.glob("checkpoints/*.pt")) + sorted(repo.glob("checkpoints/stable/*.pt")):
        if path.name.startswith("_"):
            continue
        stat = path.stat()
        out.append({
            "name": str(path.relative_to(repo)),
            "bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).astimezone()
                                .isoformat(timespec="seconds"),
        })
    return out


def estimate(rows, current, total):
    """Seconds per epoch from the run's own history, then when it finishes.

    Epoch 1 carries cache warm-up and is reliably the slowest, so once a second
    epoch exists it is dropped from the mean rather than dragging the estimate.
    """
    if not rows or not total:
        return None
    times = [row["seconds"] for row in rows if row.get("seconds")]
    if not times:
        return None

    if len(times) > 1:
        later = times[1:]
        per_epoch, basis = sum(later) / len(later), f"mean of {len(later)} epoch(s), excluding warm-up"
    else:
        per_epoch, basis = times[0], "epoch 1 only, includes warm-up"

    done = len(rows)
    within = 0.0
    if current and current.get("iterations"):
        within = current["iteration"] / current["iterations"]
    remaining = max(total - done - within, 0) * per_epoch

    return {
        "seconds_per_epoch": round(per_epoch, 1),
        "basis": basis,
        "epochs_done": done,
        "epochs_total": total,
        "elapsed_seconds": round(sum(times), 1),
        "remaining_seconds": round(remaining, 1),
        "finish": (datetime.now() + timedelta(seconds=remaining))
                  .astimezone().isoformat(timespec="seconds"),
        "fraction_complete": round((done + within) / total, 4),
    }


def build_status(repo, log_path=None):
    log = Path(log_path) if log_path else repo / "out" / "train.log"
    pid = training_pid()
    status = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo": str(repo),
        "running": pid is not None,
        "pid": pid,
        "log": str(log),
        "epochs": [],
        "config": {},
        "current": None,
        "weights": checkpoints(repo),
        "eta": None,
    }
    if not log.exists():
        status["message"] = f"no training log at {log}"
        return status

    rows, current, header = parse_log(read_log(log))
    config = read_run_config(repo / "scripts" / "run_training.sh")
    config.update({k: v for k, v in header.items() if v is not None})

    total = int(config.get("epochs_total") or config.get("epochs") or 0)
    status["epochs"] = rows
    status["current"] = current
    status["config"] = config
    status["eta"] = estimate(rows, current, total)
    status["best"] = next((r["epoch"] for r in reversed(rows) if r.get("best")), None)
    return status


class Handler(SimpleHTTPRequestHandler):
    repo = DEFAULT_REPO
    log_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] != "/api/status":
            return super().do_GET()
        try:
            payload = build_status(self.repo, self.log_path)
        except Exception as error:                 # noqa: BLE001
            # A monitor that dies because one line was half-written is worse than
            # one that reports the problem and is polled again in two seconds.
            payload = {"error": f"{type(error).__name__}: {error}",
                       "running": False, "epochs": [], "weights": []}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                                       # a poll every 2s is not news


SAMPLE = """[19:59:55] training…
device: mps
38400 patches, 2400 steps/epoch
1024 held-out patches for validation
  e1 50/2400  loss   42.907  bpp  2.592  psnr  10.34dB
epoch 1/10  loss    4.704  bpp  0.762  psnr  20.43dB  (1290s)
           val  loss    2.570  bpp  0.341  psnr  22.91dB
           new best (val loss 2.570)
epoch 2/10  loss    2.047  bpp  0.388  psnr  24.19dB  (1108s)
           val  loss    1.922  bpp  0.318  psnr  24.34dB
           new best (val loss 1.922)
  e3 550/2400  loss    1.528  bpp  0.371  psnr  25.76dB
"""


def selfcheck():
    """Parsing checks. No repository or training run required."""
    checks = []

    rows, current, header = parse_log(SAMPLE)
    checks.append(("two epoch rows parsed", len(rows) == 2))
    checks.append(("epoch 1 val bpp", rows[0].get("val_bpp") == 0.341))
    checks.append(("epoch 2 val psnr", rows[1].get("val_psnr") == 24.34))
    checks.append(("best flagged on both", all(r.get("best") for r in rows)))
    checks.append(("epoch seconds", rows[0]["seconds"] == 1290.0))
    checks.append(("header steps/epoch", header["steps_per_epoch"] == 2400))
    checks.append(("header val patches", header["val_patches"] == 1024))
    checks.append(("header total epochs", header["epochs_total"] == 10))
    checks.append(("device", header["device"] == "mps"))
    checks.append(("current is the live epoch 3", current and current["epoch"] == 3))
    checks.append(("current iteration", current and current["iteration"] == 550))

    # A step line older than the newest epoch summary must not be shown as live.
    stale, stale_current, _ = parse_log(SAMPLE.rsplit("  e3", 1)[0])
    checks.append(("stale step line dropped", stale_current is None))

    # A half-written final line must not crash or invent a row.
    torn, _, _ = parse_log(SAMPLE + "epoch 3/10  loss    1.4")
    checks.append(("torn line ignored", len(torn) == 2))

    # inf PSNR is a real state, not a parse failure.
    inf_rows, _, _ = parse_log(
        "epoch 1/2  loss 0.0  bpp 0.1  psnr infdB  (10s)\n")
    checks.append(("inf psnr survives", inf_rows and inf_rows[0]["train_psnr"] is None))

    eta = estimate([{"seconds": 1290.0}, {"seconds": 1108.0}, {"seconds": 1095.0}],
                   {"iteration": 1200, "iterations": 2400}, 10)
    checks.append(("eta excludes warm-up epoch", eta["seconds_per_epoch"] == 1101.5))
    checks.append(("eta counts partial epoch", eta["fraction_complete"] == 0.35))

    width = max(len(name) for name, _ in checks)
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(width)}")
        ok = ok and bool(passed)
    print(f"\n{'all checks passed' if ok else 'FAILURES above'} ({len(checks)} checks)")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--log", type=Path, default=None,
                        help="training log to read (default: <repo>/out/train.log)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(selfcheck())

    Handler.repo = args.repo.resolve()
    Handler.log_path = args.log
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"json-camera training monitor  ->  http://localhost:{args.port}")
    print(f"  repo {Handler.repo}")
    print(f"  log  {args.log or Handler.repo / 'out' / 'train.log'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
