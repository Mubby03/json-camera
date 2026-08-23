# Reaching Claude from your phone

A chat app at **<https://json-camera.fly.dev/chat>**, installable to a home
screen. Type there, and whoever is working on this repo at a terminal sees it.

```bash
python3 scripts/bridge.py watch     # block until a message arrives
python3 scripts/bridge.py say "..." # reply to the phone
python3 scripts/bridge.py history   # what has been said
```

## Install it on your phone

1. Open <https://json-camera.fly.dev/chat> in Safari or Chrome.
2. Enter the key. It is remembered, so this is once per device.
3. Share, then **Add to Home Screen**. It opens full screen with no browser bars.

The key lives in `~/.jsoncam-chat-key` on the Mac and as a Fly secret. It is
compared in constant time and required on every call, and the endpoints refuse
outright when no key is configured rather than defaulting to open.

## How it actually works, and the catch

Claude does not run continuously. It wakes when a task it started *finishes*. So
the way to reach it is to give it a task that does not finish until you say
something. `watch` is that task: it long-polls, blocks, and returns the moment a
message lands.

```
Claude arms `watch`  ->  you type  ->  Claude wakes, works, `say`s  ->  arms again
```

**If nothing is armed, your message waits.** It is never lost, and it appears
the moment a watcher runs again. But a session that has ended cannot re-arm
itself, and a fresh session will not remember the conversation that came before.
`PROJECT_BRIEF.md` exists partly for that.

## Where the messages live

A Fly volume mounted at `/data`, not memory and not the rootfs. The machine stops
when idle and the rootfs is replaced on every deploy, so either of those would
quietly lose the conversation.

Both directions long poll rather than checking on a timer: the server holds the
request open, so a message lands in about a second and neither side spins.

## Notes

- `coconut` at the start is stripped, so `coconut: run tests` and `run tests`
  both work.
- Claude's own replies are filtered out of the watch, so it never answers itself.
- `.bridge-state.json` holds the last handled id **and the channel it came from**.
  Without that namespace, state written by the old GitHub-issue version was read
  as if it meant the same thing, and since comment ids run into the billions while
  chat ids start at 1, every real message looked stale and the watcher waited
  forever with nothing visibly wrong.
