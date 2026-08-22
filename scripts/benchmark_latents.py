"""Reproduce the compressed-domain training claim on your own machine.

    python scripts/benchmark_latents.py --images data/val_images

Trains the same small network two ways, on pixels and on latents, and reports
throughput for both.  A claim like "9x faster" is worth nothing unless the
person reading it can run it, so this prints the numbers it measured rather
than the numbers I measured.

It also reports disk size against JPEG, because the storage saving and the
speed saving are separate claims and both should be checked.
"""

import argparse
import io
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def backbone(in_ch, first_stride, classes=1000):
    """Deliberately plain, and identical either way apart from the input stem.
    A fancier network would make the comparison about the network."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=first_stride, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, classes),
    )


def throughput(model, x, steps=8):
    opt = torch.optim.SGD(model.parameters(), 1e-3)
    for _ in range(2):                                  # warm up
        opt.zero_grad(); model(x).sum().backward(); opt.step()
    t = time.time()
    for _ in range(steps):
        opt.zero_grad(); model(x).sum().backward(); opt.step()
    dt = (time.time() - t) / steps
    return x.shape[0] / dt, dt * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/val_images")
    ap.add_argument("-c", "--checkpoint", default="checkpoints/stable/jc-final.pt")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from jsoncam.dataset import LatentDataset, prepare_dataset

    tmp = Path(tempfile.mkdtemp(prefix="jcl-bench-"))
    try:
        shard = tmp / "bench.jcl"
        print(f"encoding up to {args.limit} images at {args.size}x{args.size}...")
        prepare_dataset(args.images, shard, checkpoint=args.checkpoint,
                        size=args.size, limit=args.limit, progress=False)
        ds = LatentDataset(shard, checkpoint=args.checkpoint)
        C, H, W = ds.latent_shape

        # --- storage -----------------------------------------------------
        src = sorted(p for p in Path(args.images).rglob("*")
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.limit]
        jpeg_total = 0
        for p in src:
            b = io.BytesIO()
            Image.open(p).convert("RGB").resize((args.size, args.size)).save(b, "JPEG", quality=90)
            jpeg_total += b.tell()
        shard_size = os.path.getsize(shard)

        print(f"\n  STORAGE for {len(ds)} images at {args.size}x{args.size}")
        print(f"    as JPEG q90          {jpeg_total/1024:9.1f} KB")
        print(f"    as latents (.jcl)    {shard_size/1024:9.1f} KB   "
              f"{jpeg_total/max(1,shard_size):.1f}x smaller")

        # --- throughput --------------------------------------------------
        pix = torch.randn(args.batch, 3, args.size, args.size)
        lat = torch.randn(args.batch, C, H, W)
        pix_ips, pix_ms = throughput(backbone(3, 2), pix)
        lat_ips, lat_ms = throughput(backbone(C, 1), lat)

        print(f"\n  TRAINING THROUGHPUT, batch {args.batch}, {args.threads} threads")
        print(f"    on pixels   3 x {args.size} x {args.size:<4} {pix_ms:8.1f} ms/step {pix_ips:8.0f} img/s")
        print(f"    on latents  {C} x {H} x {W:<6} {lat_ms:8.1f} ms/step {lat_ips:8.0f} img/s")
        print(f"    speedup     {lat_ips/pix_ips:.1f}x   "
              f"(input tensor is {3*args.size*args.size/(C*H*W):.1f}x smaller)")

        # --- loading -----------------------------------------------------
        t = time.time()
        for i in range(min(len(ds), 12)):
            ds[i]
        per = (time.time() - t) / min(len(ds), 12) * 1000
        print(f"\n  LOADING")
        print(f"    unpack one latent    {per:8.2f} ms   "
              f"-> {1000/per:.0f} img/s per worker")
        verdict = ("faster than the model can consume, so decoding is free"
                   if 1000 / per * 4 > lat_ips else
                   "SLOWER than the model consumes; raise num_workers")
        print(f"    with 4 workers       {4000/per:8.0f} img/s   {verdict}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
