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


# --------------------------------------------------------------------------
# multi-scale image cache: whole images at several scales, cropped online
#
# The patch cache above fixes the crops once; a long run then sees the same
# 38,400 windows over and over.  This keeps whole images instead, each stored
# at several scales, and cuts a fresh random window every time it is asked.
# A 2K photograph at 0.35 is about 720p, so the scale tiers are also how the
# model learns that detail comes in different sizes: a native-resolution crop
# of a 4K frame is smooth per pixel, a 720p crop is dense.

SCALES = (1.0, 0.75, 0.5, 0.35)
SCALE_WEIGHTS = (0.40, 0.25, 0.20, 0.15)


def _probe(p):
    try:
        with Image.open(p) as im:
            return im.size
    except Exception as e:
        print(f"  skip {os.path.basename(p)}: {e}")
        return None


def _render_entry(job):
    """Worker: decode one image, resize to one scale, write into the memmap."""
    cache_path, path, scale, offset, w, h = job
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        # Antialiased downscale, so a 0.35 tier is an honest 720p photograph
        # and not an aliased one.
        img = img.resize((w, h), Image.LANCZOS)
    a = np.asarray(img, dtype=np.uint8)
    assert a.shape == (h, w, 3), (a.shape, (h, w))
    mm = np.load(cache_path, mmap_mode="r+")
    mm[offset : offset + a.size] = a.reshape(-1)
    mm.flush()
    del mm
    return a.size


def build_image_cache(folders, out_path, scales=SCALES, patch=256, limit=None, workers=None):
    """Store every image under `folders` at every scale, in one flat memmap.

    Writes `out_path` (a 1-D uint8 .npy) and `out_path + '.index.npz'`
    describing where each (image, scale) rendering lives.
    """
    from multiprocessing import Pool

    if isinstance(folders, str):
        folders = [folders]
    paths = []
    for f in folders:
        paths.extend(list_images(f))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found under {folders}")

    entries = []  # (path, scale, offset, w, h)
    offset = 0
    for p in paths:
        size = _probe(p)
        if size is None:
            continue
        W, H = size
        for s in scales:
            w, h = (W, H) if s == 1.0 else (round(W * s), round(H * s))
            if w < patch or h < patch:
                continue  # never upscale, never crop past the edge
            entries.append((p, s, offset, w, h))
            offset += w * h * 3
    print(f"{len(paths)} images -> {len(entries)} renderings, {offset/1e9:.1f} GB")

    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(offset,))
    del mm
    jobs = [(out_path, p, s, o, w, h) for (p, s, o, w, h) in entries]
    done = 0
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render_entry, jobs, chunksize=4)):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} renderings")
    np.savez(out_path + ".index.npz",
             path=np.array([e[0] for e in entries]),
             scale=np.array([e[1] for e in entries], dtype=np.float32),
             offset=np.array([e[2] for e in entries], dtype=np.int64),
             w=np.array([e[3] for e in entries], dtype=np.int64),
             h=np.array([e[4] for e in entries], dtype=np.int64))
    print(f"wrote {len(entries)} renderings to {out_path} ({os.path.getsize(out_path)/1e9:.1f} GB)")
    return len(entries)


class ImageCropDataset(Dataset):
    """Random `patch`-sized windows out of a multi-scale image cache.

    An "epoch" is whatever `length` says: the sampler is random, so there is
    no natural end to the data, and the trainer decides how many steps make a
    checkpoint-and-validate cycle.
    """

    def __init__(self, cache_path, patch=256, length=32000, augment=True,
                 scale_weights=None, seed=None):
        self.cache_path, self.patch, self.length, self.augment = cache_path, patch, length, augment
        idx = np.load(cache_path + ".index.npz")
        self.scale = idx["scale"]; self.offset = idx["offset"]
        self.w = idx["w"]; self.h = idx["h"]
        self.n_images = len(set(idx["path"].tolist()))
        # Pick a scale tier first, then an image uniformly within it, so the
        # tier mix is what SCALE_WEIGHTS says rather than whatever the image
        # sizes happen to make it.
        # Stored as float32, so 0.35 comes back as 0.34999999; compare rounded.
        self.scale = np.round(self.scale.astype(np.float64), 4)
        tiers = sorted(set(self.scale.tolist()), reverse=True)
        weights = scale_weights or {round(s, 4): w for s, w in zip(SCALES, SCALE_WEIGHTS)}
        self.tiers = [(np.flatnonzero(self.scale == s), weights[s]) for s in tiers]
        self._data = None
        self._rng = random.Random(seed)

    @property
    def data(self):
        if self._data is None:  # opened lazily, so a worker gets its own handle
            self._data = np.load(self.cache_path, mmap_mode="r")
        return self._data

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_data"] = None
        return d

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = self._rng
        members, _ = rng.choices(self.tiers, weights=[w for _, w in self.tiers])[0]
        e = int(members[rng.randrange(len(members))])
        w, h, p = int(self.w[e]), int(self.h[e]), self.patch
        x = rng.randrange(0, w - p + 1)
        y = rng.randrange(0, h - p + 1)
        img = self.data[self.offset[e] : self.offset[e] + w * h * 3].reshape(h, w, 3)
        a = np.asarray(img[y : y + p, x : x + p])
        if self.augment:
            if rng.random() < 0.5:
                a = a[:, ::-1]
            if rng.random() < 0.5:
                a = a[::-1]
            k = rng.randrange(4)
            if k:
                a = np.rot90(a, k)
        a = np.array(a, dtype=np.uint8, order="C")  # a real copy: the memmap is read-only
        return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)
