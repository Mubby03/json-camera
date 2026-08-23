#!/usr/bin/env python3
"""A way to reach Claude from a phone while it works on this repo.

    python3 scripts/bridge.py watch        # block until a new instruction arrives
    python3 scripts/bridge.py say "done"   # reply into the thread
    python3 scripts/bridge.py history      # what has been said so far

The channel is a GitHub issue on a private repo, chosen because `gh` is already
authenticated on this machine, so there is nothing new to sign into and no secret
to store.  Comment from the GitHub mobile app and it lands here.

**Why `watch` blocks rather than polls in the background.**  Claude does not run
continuously; it wakes when a task it started finishes.  So the way to reach it
is to give it a task that does not finish until you say something.  `watch` is
that task: it long-polls, blocks, and exits the moment a new comment appears.
Claude is woken by the exit and reads what you wrote.

That means the loop is: Claude arms `watch`, you comment, Claude wakes, does the
work, replies with `say`, and arms `watch` again.  If nothing is armed, your
comment simply waits in the thread until something is.

State is a single file holding the id of the last comment already handled, so a
restart never replays old instructions and never silently skips one.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = os.environ.get("JSONCAM_BRIDGE_REPO", "Mubby03/json-camera-control")
ISSUE = os.environ.get("JSONCAM_BRIDGE_ISSUE", "1")
STATE = Path(__file__).resolve().parent.parent / ".bridge-state.json"
POLL_SECONDS = int(os.environ.get("JSONCAM_BRIDGE_POLL", "20"))


def gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "gh failed")
    return r.stdout


def comments():
    """Every comment on the channel, oldest first."""
    raw = gh("api", f"repos/{REPO}/issues/{ISSUE}/comments",
             "--paginate", "-q",
             '.[] | {id, user: .user.login, at: .created_at, body}')
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_seen():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text()).get("last_id", 0)
        except (ValueError, OSError):
            pass
    return 0


def save_seen(comment_id):
    # Written only after a message has been handed over, so a crash mid-handover
    # replays the instruction rather than losing it. A repeat is recoverable; a
    # dropped instruction looks like Claude ignored you.
    STATE.write_text(json.dumps({"last_id": comment_id, "at": time.time()}))


def cmd_watch(args):
    seen = load_seen()
    me = gh("api", "user", "-q", ".login").strip()
    deadline = time.time() + args.timeout if args.timeout else None
    print(f"watching {REPO}#{ISSUE} for new comments (last handled id {seen})", flush=True)

    while True:
        try:
            fresh = [c for c in comments() if c["id"] > seen and c["user"] == me]
        except Exception as e:                       # a blip must not end the watch
            print(f"  poll failed, retrying: {e}", flush=True)
            fresh = []

        if fresh:
            c = fresh[0]
            body = c["body"].strip()
            # `coconut` is an optional marker; strip it so the instruction reads clean.
            if body.lower().startswith("coconut"):
                body = body[len("coconut"):].lstrip(" :,-\n")
            print("\n=== NEW INSTRUCTION ===", flush=True)
            print(f"from {c['user']} at {c['at']}", flush=True)
            print(body, flush=True)
            print("=== END ===", flush=True)
            save_seen(c["id"])
            return 0

        if deadline and time.time() > deadline:
            print("no new instruction before the timeout", flush=True)
            return 2
        time.sleep(POLL_SECONDS)


def cmd_say(args):
    gh("issue", "comment", ISSUE, "--repo", REPO, "--body", args.text)
    print(f"replied on {REPO}#{ISSUE}")
    # Our own reply must not read back as an instruction on the next watch.
    latest = comments()
    if latest:
        save_seen(latest[-1]["id"])
    return 0


def cmd_history(args):
    for c in comments()[-args.n:]:
        print(f"--- {c['user']} at {c['at']} (id {c['id']})")
        print(c["body"].strip())
        print()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("watch", help="block until a new instruction arrives")
    p.add_argument("--timeout", type=int, default=0, help="seconds, 0 means forever")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("say", help="reply into the thread")
    p.add_argument("text")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("history", help="show recent messages")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(fn=cmd_history)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
