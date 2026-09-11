#!/bin/bash
# The long run: a bigger network, fresh multi-scale crops, and a wall-clock
# deadline instead of an epoch count.
#
# Train set: DIV2K train (800) + Flickr2K (2650), each stored at 1.0 / 0.75 /
#            0.5 / 0.35 of its size, so a 2K photograph also appears as a
#            ~1080p and a ~720p one. Crops are cut fresh every step.
# Val set:   DIV2K valid (0801-0900), fixed 256px patches, never trained on.
#
#   nohup caffeinate -dims scripts/run_training_sharp.sh > out/train_sharp.log 2>&1 &
#
# caffeinate matters: a laptop that sleeps at 2am ends the run at 2am.
set -e
cd ~/dev/json-camera
export PYTHONUNBUFFERED=1

DEADLINE="${DEADLINE:-2026-09-12T14:20}"
OUT="${OUT:-checkpoints/jc-sharp.pt}"

FOLDERS="data/DIV2K_train_HR"
[ -d data/Flickr2K ] && FOLDERS="$FOLDERS data/Flickr2K"

if [ ! -s data/images.npy.index.npz ]; then
  echo "[$(date +%H:%M:%S)] building multi-scale image cache from: $FOLDERS"
  .venv/bin/jsoncam prepare-images --images $FOLDERS --out data/images.npy --workers 8
fi
if [ ! -s data/val_patches.npy ]; then
  echo "[$(date +%H:%M:%S)] building val patch cache…"
  .venv/bin/jsoncam prepare --images data/DIV2K_valid_HR --out data/val_patches.npy \
      --patch 256 --per-image 16
fi

# workers=0 on purpose: the cropper does ~2000 crops/s single-threaded against
# ~22 images/s the GPU consumes, and the spawn-based worker pool wedges on this
# machine. 2000 steps at ~1.35 steps/s is a ~25 minute epoch; the deadline, not
# --epochs, decides when it ends, and the learning rate anneals to the deadline.
# RESUME=1 picks the last epoch checkpoint back up after a crash or a reboot.
RESUME_ARGS=""
[ -n "${RESUME:-}" ] && [ -s "$OUT" ] && RESUME_ARGS="--resume $OUT"

echo "[$(date +%H:%M:%S)] training until $DEADLINE $RESUME_ARGS"
.venv/bin/jsoncam train $RESUME_ARGS \
    --image-cache data/images.npy --patch 256 --steps-per-epoch 2000 \
    --val-cache data/val_patches.npy \
    --out "$OUT" \
    --hidden 128 --latent 192 \
    --lmbda 0.05 \
    --batch 16 --lr 1e-4 --warmup 500 --ema 0.9995 \
    --epochs 32 --deadline "$DEADLINE" --workers 0 --log-every 50
echo "[$(date +%H:%M:%S)] DONE"
