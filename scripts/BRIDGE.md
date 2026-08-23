# Reaching Claude from your phone

A private GitHub issue is the channel. Comment on it from the GitHub mobile app
and Claude, working in the CLI on the Mac, wakes up and reads it.

**Channel:** <https://github.com/Mubby03/json-camera-control/issues/1>

```bash
python3 scripts/bridge.py watch     # block until an instruction arrives
python3 scripts/bridge.py say "..." # reply into the thread
python3 scripts/bridge.py history   # what has been said
```

## Why a GitHub issue and not the Us app

The Us app was the first idea and it does not work: there is no `.env` on this
machine, so there are no Supabase credentials and nothing can read or write those
notes. `gh` is already authenticated, so this channel needed no new secret and no
new account. The repo is private, so the messages are not public.

If the Us app credentials ever land on the machine, the same `watch` and `say`
shape would port over; only the transport changes.

## How it actually works, and the catch

Claude does not run continuously. It wakes when a task it started finishes. So
the way to reach it is to give it a task that **does not finish until you say
something**. That is what `watch` is: it long-polls the issue, blocks, and exits
the moment a new comment appears. The exit is what wakes Claude.

The loop:

```
Claude arms `watch`  ->  you comment  ->  Claude wakes, works, `say`s  ->  arms again
```

**The catch worth knowing:** if nothing is armed, your comment just waits in the
thread. It is not lost, but nothing happens until a watcher is running again.
Claude re-arms after each instruction, but a session that has ended cannot re-arm
itself, and a brand new session will not remember this conversation.

`PROJECT_BRIEF.md` exists partly for that: it is the context a fresh session needs.

## Notes

- `coconut` at the start of a comment is stripped, so both `coconut: run tests`
  and `run tests` work.
- Only comments from the authenticated account are treated as instructions, so
  Claude's own replies do not read back as new work.
- `.bridge-state.json` holds the id of the last handled comment. It is written
  *after* handover, so a crash mid-handover repeats an instruction rather than
  dropping it. A repeat is recoverable; a silently dropped instruction looks like
  being ignored.
