# Training Monitor

A live web view of a json-camera training run.

```bash
python3 monitor/server.py
```

Then open **http://localhost:8765**.

Any Python 3 works — the system one is fine. There is nothing to install and it
does not need the project's `.venv`.

## Files

| File | What it is |
|---|---|
| `index.html` | the page |
| `style.css` | tokens, layout, light/dark |
| `app.js` | polling, charts, tooltips — no libraries |
| `server.py` | static files + `/api/status`, standard library only |

## What it shows

- **Two progress bars** — epochs done out of the total, and steps done inside
  the current epoch, with a finish time computed from the run's own measured
  epoch times rather than a guess.
- **Four stat tiles** — validation PSNR, bpp and loss, plus the compression
  ratio against raw RGB, each with the change since the previous epoch.
- **Three charts** — quality (PSNR), rate (bits per pixel), and the
  rate-distortion loss, training and validation on each. Hover for a crosshair
  and per-epoch values.
- **A table** of every epoch, and which ones took `best.pt`.
- **Run configuration and checkpoints**, including when `jc.best.pt` last moved.

## Why there is no mAP

This started as the YOLO11m / VisDrone monitor and the chart machinery is
unchanged, but the metrics could not carry over. A detector is scored with mAP,
precision and recall — all of which need class labels and boxes to rank. A
codec has neither. It is scored on **rate** against **distortion**: how many
bits a picture cost, and how wrong it came back. Putting a bpp under a tile
labelled "mAP50" would be worse than showing nothing, so the tiles say what
they mean.

One consequence worth knowing about: on a detector every metric improves
upwards, so the original could colour any rise green. Here only PSNR does. Bits
per pixel and loss improve by **falling**, so each series declares which
direction is better and the deltas are coloured from that rather than from the
sign.

## Reading it

**Rate should stay roughly flat.** `lambda` pins where the codec sits on the
rate-distortion curve, so a healthy run holds its bpp near constant while PSNR
climbs. A validation bpp that keeps sliding is not good news — it means the
model is buying its loss by throwing the picture away.

**Validation PSNR reads higher than training PSNR.** That is expected, not a
fault: training quantises by adding uniform noise, which is a deliberately
pessimistic stand-in, while validation rounds for real and skips augmentation.

## It cannot break the training run

The server opens two files read-only — the run log and `scripts/run_training.sh`
— and never writes to the repository or signals the training process. Stopping
the server, closing the page or leaving it open overnight all have exactly the
same effect on training, which is none.

Two details worth knowing:

- A log line caught **mid-write** is skipped rather than guessed at. It lands on
  the next poll two seconds later.
- A step line left over from an epoch that has since finished is **dropped**
  rather than shown as live, so the epoch bar does not sit pinned at 100% during
  validation.

## Options

```bash
python3 monitor/server.py --port 9000               # a different port
python3 monitor/server.py --repo /path/to/checkout  # a different checkout
python3 monitor/server.py --log out/other.log       # a different run log
python3 monitor/server.py --selfcheck               # parsing checks, no server needed
```

`--selfcheck` runs sixteen checks covering every log line format, the stale step
line, a torn final line, an infinite PSNR, and the ETA logic. No repository or
training run required.
