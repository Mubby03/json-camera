"""Training data: random crops, cached as a flat uint8 memmap.

Decoding a 2K JPEG for every sample would make training data-bound, so we pay
that cost once up front and extract patches into a single memmap file.  After
that a batch is just a memory read and the GPU stays busy.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append(os.path.join(root, f))
    return out


def build_patch_cache(folder, out_path, patch=256, per_image=24, seed=0, limit=None):
    """Extract random patches from every image into one uint8 memmap."""
    paths = list_images(folder)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found under {folder}")

    rng = random.Random(seed)
    total = len(paths) * per_image
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(total, patch, patch, 3)
    )

    n = 0
    for pi, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:  # skip unreadable files rather than kill the run
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        W, H = img.size
        if W < patch or H < patch:
            # Upscaling a small image would teach the net to compress blur.
            continue
        a = np.asarray(img)
        for _ in range(per_image):
            x = rng.randrange(0, W - patch + 1)
            y = rng.randrange(0, H - patch + 1)
            mm[n] = a[y : y + patch, x : x + patch]
            n += 1
        if (pi + 1) % 25 == 0:
            print(f"  {pi + 1}/{len(paths)} images -> {n} patches")

    mm.flush()
    del mm
    # Trim to what we actually wrote.  Rewriting means pulling the whole cache
    # through RAM, which for a big set is several GB -- so only do it when some
    # images were actually skipped and the tail really is unwritten padding.
    if n < total:
        arr = np.load(out_path, mmap_mode="r")[:n]
        np.save(out_path, np.ascontiguousarray(arr))
    print(f"wrote {n} patches of {patch}x{patch} to {out_path} "
          f"({os.path.getsize(out_path)/1e9:.1f} GB)")
    return n


class PatchDataset(Dataset):
    def __init__(self, cache_path, augment=True):
        self.data = np.load(cache_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        a = np.asarray(self.data[i])
        if self.augment:
            if random.random() < 0.5:
                a = a[:, ::-1]
            if random.random() < 0.5:
                a = a[::-1]
            k = random.randrange(4)
            if k:
                a = np.rot90(a, k)
        a = np.ascontiguousarray(a)
        return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)
