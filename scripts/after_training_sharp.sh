#!/bin/bash
# Waits for the sharp run to finish, then does the post-training pass unattended:
# slim the best checkpoint, benchmark it against JPEG and against both shipped
# models on the same 12 photographs, run the tests. Writes a Markdown report.
set -u
cd /Users/mubaraq/dev/json-camera
NEW=checkpoints/stable/jc-sharp.pt

echo "# Sharp run (lambda 0.05, hidden 128 / latent 192, multi-scale crops)"
echo
echo "waiting for training to finish..."
while pgrep -f 'jsoncam train' >/dev/null 2>&1; do sleep 60; done
echo "finished at $(date '+%Y-%m-%d %H:%M')"
echo
echo '## Per-epoch'
echo '```'
grep -E '^epoch|val  |ema  |new best|deadline' out/train_sharp.log
echo '```'

echo
echo '## Slimming the best checkpoint'
echo '```'
mkdir -p checkpoints/stable
.venv/bin/jsoncam export checkpoints/jc-sharp.best.pt -o $NEW 2>&1 | grep -viE 'warn'
echo '```'

for ck in $NEW jsoncam/models/jc-hq.pt jsoncam/models/jc-final.pt; do
  echo
  echo "## $ck against JPEG at matched size, 12 held-out photographs"
  echo '```'
  .venv/bin/python scripts/benchmark.py -c $ck --limit 12 2>&1 | grep -viE 'warn'
  echo '```'
done

echo
echo '## Tests'
echo '```'
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -3
echo '```'
echo
echo "done $(date '+%H:%M')"
