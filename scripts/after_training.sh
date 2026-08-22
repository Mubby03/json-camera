#!/bin/bash
# Waits for the run to finish, then does the whole post-training pass unattended.
set -u
cd /Users/mubaraq/dev/json-camera

echo "# High-quality run (lambda 0.05)"
echo
echo "waiting for training to finish..."
while pgrep -f 'jsoncam train' >/dev/null 2>&1; do sleep 60; done
echo "finished at $(date '+%Y-%m-%d %H:%M')"
echo
echo '## Per-epoch'
echo '```'
grep -E '^epoch|val  |new best' out/train_hq.log
echo '```'

echo
echo '## Slimming the best checkpoint'
echo '```'
.venv/bin/jsoncam export checkpoints/jc-hq.best.pt -o checkpoints/stable/jc-hq.pt 2>&1 | grep -viE 'warn'
echo '```'

echo
echo '## Against JPEG at matched size, 12 held-out photographs'
echo '```'
.venv/bin/python scripts/benchmark.py -c checkpoints/stable/jc-hq.pt --limit 12 2>&1 | grep -viE 'warn'
echo '```'

echo
echo '## The old model, same benchmark, for comparison'
echo '```'
.venv/bin/python scripts/benchmark.py -c checkpoints/stable/jc-final.pt --limit 12 2>&1 | grep -viE 'warn' | tail -8
echo '```'

echo
echo '## Tests'
echo '```'
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -3
echo '```'
echo
echo "done $(date '+%H:%M')"
