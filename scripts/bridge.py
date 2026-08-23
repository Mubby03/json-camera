#!/usr/bin/env python3
"""A way to reach Claude from a phone while it works on this repo.

    python3 scripts/bridge.py watch        # block until a new instruction arrives
    python3 scripts/bridge.py say "done"   # reply into the thread
    python3 scripts/bridge.py history      # what has been said so far

The channel is a chat app at /chat on the deployed site, installable to a phone
home screen as a PWA.  A GitHub issue was the first version and worked, but it is
not a chat app: this one has bubbles, a keyboard that behaves, and it opens in one
tap without loading a whole client.

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

BASE = os.environ.get("JSONCAM_CHAT_URL", "https://json-camera.fly.dev").rstrip("/")
STATE = Path(__file__).resolve().parent.parent / ".bridge-state.json"


def key():
    k = os.environ.get("JSONCAM_CHAT_KEY", "")
    if not k:
        path = Path.home() / ".jsoncam-chat-key"
        if path.exists():
            k = path.read_text().strip()
    if not k:
        raise SystemExit(
            "No chat key. Put it in ~/.jsoncam-chat-key or set JSONCAM_CHAT_KEY.")
    return k


def fetch(after=0, wait=0):
    """Messages after `after`. `wait` holds the connection open server side."""
    import urllib.parse
    import urllib.request

    q = urllib.parse.urlencode({"after": after, "key": key(), "wait": wait})
    with urllib.request.urlopen(f"{BASE}/api/chat?{q}", timeout=wait + 30) as r:
        return json.loads(r.read().decode("utf-8"))["messages"]


def post(text, who="claude"):
    import urllib.parse
    import urllib.request

    data = urllib.parse.urlencode({"text": text, "who": who, "key": key()}).encode()
    req = urllib.request.Request(f"{BASE}/api/chat", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# Bumped whenever the id space changes. Without this, state written by an older
# transport is read as if it meant the same thing: GitHub comment ids run into
# the billions while chat ids start at 1, so every real message looked stale and
# the watcher waited forever with nothing visibly wrong.
CHANNEL = f"chat@{BASE}"


def load_seen():
    if STATE.exists():
        try:
            data = json.loads(STATE.read_text())
            if data.get("channel") == CHANNEL:
                return data.get("last_id", 0)
        except (ValueError, OSError):
            pass
    return 0


def save_seen(message_id):
    # Written only after a message has been handed over, so a crash mid-handover
    # replays it rather than losing it. A repeat is recoverable; a dropped
    # message is indistinguishable from being ignored.
    STATE.write_text(json.dumps(
        {"channel": CHANNEL, "last_id": message_id, "at": time.time()}))


def cmd_watch(args):
    seen = load_seen()
    deadline = time.time() + args.timeout if args.timeout else None
    print(f"watching {BASE}/chat (last handled id {seen})", flush=True)

    while True:
        try:
            # A long poll rather than a tight loop: the server holds the request
            # open, so a message arrives in about a second instead of after a
            # sleep, and the machine is not woken pointlessly in between.
            fresh = [m for m in fetch(after=seen, wait=45) if m["who"] != "claude"]
        except Exception as e:                       # a blip must not end the watch
            print(f"  poll failed, retrying: {e}", flush=True)
            fresh = []
            time.sleep(5)

        if fresh:
            m = fresh[0]
            text = m["text"].strip()
            # `coconut` is an optional marker; strip it so the instruction reads clean.
            if text.lower().startswith("coconut"):
                text = text[len("coconut"):].lstrip(" :,-\n")
            when = time.strftime("%H:%M:%S", time.localtime(m["at"]))
            print("\n=== NEW MESSAGE ===", flush=True)
            print(f"from {m['who']} at {when}", flush=True)
            print(text, flush=True)
            print("=== END ===", flush=True)
            save_seen(m["id"])
            return 0

        if deadline and time.time() > deadline:
            print("nothing arrived before the timeout", flush=True)
            return 2


def cmd_say(args):
    m = post(args.text)
    save_seen(max(load_seen(), m["id"]))   # our own reply is not an instruction
    print(f"sent as message {m['id']}")
    return 0


def cmd_history(args):
    for m in fetch()[-args.n:]:
        when = time.strftime("%d %b %H:%M", time.localtime(m["at"]))
        print(f"--- {m['who']} at {when} (id {m['id']})")
        print(m["text"].strip())
        print()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("watch", help="block until a new message arrives")
    p.add_argument("--timeout", type=int, default=0, help="seconds, 0 means forever")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("say", help="send a message to the phone")
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
