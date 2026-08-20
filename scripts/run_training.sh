#!/bin/bash
# Build the patch caches (if needed), then train.
#
# Train set: DIV2K_train_HR   (800 images, 0001-0800)
# Val set:   data/val_images  (64 images from DIV2K valid, 0804-0900) -- disjoint,
#            so the "best" checkpoint is chosen on data the model never trained on.
set -e
cd ~/dev/json-camera
export PYTHONUNBUFFERED=1

if [ ! -s data/patches.npy ]; then
  echo "[$(date +%H:%M:%S)] building train patch cache…"
  .venv/bin/jsoncam prepare --images data/DIV2K_train_HR --out data/patches.npy \
      --patch 256 --per-image 48
fi
if [ ! -s data/val_patches.npy ]; then
  echo "[$(date +%H:%M:%S)] building val patch cache…"
  .venv/bin/jsoncam prepare --images data/val_images --out data/val_patches.npy \
      --patch 256 --per-image 16
fi

# workers=0 on purpose: a single-threaded loader does 46 batches/s off the memmap,
# ~20x more than the ~2.3 steps/s the GPU can consume, and the spawn-based worker
# pool wedges on this machine.  Extra processes would only add risk here.
# 38,400 patches / batch 16 = 2,400 steps/epoch.  At ~2.3 steps/s on an M1 Pro
# that is ~17 min/epoch, so 10 epochs is roughly a 3 hour run.
echo "[$(date +%H:%M:%S)] training…"
.venv/bin/jsoncam train \
    --cache data/patches.npy \
    --val-cache data/val_patches.npy \
    --out checkpoints/jc.pt \
    --hidden 96 --latent 128 \
    --lmbda 0.0067 \
    --batch 16 --lr 1e-4 \
    --epochs 10 --workers 0 --log-every 50
echo "[$(date +%H:%M:%S)] DONE"
