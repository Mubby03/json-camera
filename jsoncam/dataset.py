"""Train on the compressed representation instead of on pixels.

The usual reason to compress a training set is to save disk.  This saves disk
*and* training time, because it hands the model the latent grid rather than the
picture, and the latent is six times smaller than the pixels it stands for.
Every layer downstream of the input is then working on a smaller tensor.

Measured on one machine, same architecture and batch either way:

    3 x 224 x 224 pixels    150,528 values     51 img/s
    128 x 14 x 14 latents    25,088 values    459 img/s     9x faster

The obvious alternative, storing decoded pixels and skipping all of this, is
worse on both counts.  And the obvious other alternative, storing raw int16
latents for instant loading, is 2.7x *larger* on disk than JPEG, which throws
away the storage win.  So shards hold the rANS bitstream, 2.9 KB per image
against JPEG's 18.2 KB, and workers unpack it.  That unpacking costs about 5 ms,
so four workers deliver ~810 images a second while the model above can only
consume 459: decoding is free in practice because it happens off the critical
path.

    jsoncam prepare-latents photos/ --out train.jcl
    ds = LatentDataset("train.jcl")

One caveat that matters.  A latent is only meaningful to the checkpoint that
produced it, so a shard records the model fingerprint and refuses to open under
a different one.  Retrain your codec and you must rebuild your shards.
"""

import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from . import codec, rans

MAGIC = b"JCL1"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# writing


class ShardWriter:
    """Streams items to disk so a million-image set never sits in memory.

    Layout: magic, then the length-prefixed JSON header, then every bitstream
    back to back, then the index, then a pointer to where the index started.
    The index goes last because its size is not known until the data is written,
    and the trailing pointer is what makes it findable without scanning.
    """

    def __init__(self, path, model, size=None):
        self.path = Path(path)
        self.model = model
        self.size = size
        self.items = []
        self.fh = self.path.open("wb")
        self.fh.write(MAGIC)
        header = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": {**model.config, "fingerprint": codec.model_fingerprint(model)},
            "precision": 12,
            "resize": size,
        }
        blob = json.dumps(header).encode("utf-8")
        self.fh.write(struct.pack("<I", len(blob)))
        self.fh.write(blob)
        self.data_start = self.fh.tell()
        self.tables = codec._tables(model, 12)

    @torch.no_grad()
    def add(self, img, label=0, name=None):
        img = ImageOps.exif_transpose(img).convert("RGB")
        if self.size:
            img = img.resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        x = x.float().div(255.0).unsqueeze(0)
        H, W = x.shape[2], x.shape[3]
        ph, pw = (-H) % codec.DOWNSCALE, (-W) % codec.DOWNSCALE
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")

        y = self.model.encoder(x)
        s = self.model.prior.symbols(y)[0].numpy().astype(np.int64)
        lo = self.tables["range_lo"][:, None, None]
        hi = self.tables["range_hi"][:, None, None]
        s = np.clip(s, lo, hi) - lo

        C, lh, lw = s.shape
        chans = np.repeat(np.arange(C, dtype=np.int32), lh * lw)
        blob, lanes = rans.encode(s.reshape(-1).astype(np.int32), chans,
                                  self.tables["freqs"], self.tables["starts"], 12)
        off = self.fh.tell()
        self.fh.write(blob)
        self.items.append({
            "o": off, "n": len(blob), "l": int(label), "lanes": lanes,
            "c": C, "h": lh, "w": lw, "name": name,
        })
        return len(blob)

    def close(self):
        index_at = self.fh.tell()
        payload = json.dumps(self.items).encode("utf-8")
        self.fh.write(payload)
        self.fh.write(struct.pack("<Q", index_at))
        self.fh.close()
        return len(self.items)


def prepare_dataset(images, out, checkpoint="checkpoints/stable/jc-final.pt",
                    size=None, labels=None, limit=None, progress=True):
    """Encode a folder of images into one shard of latents.

    `images` is a directory.  Subdirectories become integer class labels, the
    convention torchvision's ImageFolder uses, so an existing layout works
    unchanged.  `labels` overrides that with an explicit mapping.
    """
    model, _ = codec.load_checkpoint(checkpoint)
    model.eval()
    root = Path(images)

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no images under {root}")

    classes = sorted({p.parent.name for p in files if p.parent != root})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    writer = ShardWriter(out, model, size=size)
    total_in = total_out = 0
    for i, p in enumerate(files):
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        label = (labels or {}).get(str(p), class_to_idx.get(p.parent.name, 0))
        total_out += writer.add(img, label=label, name=p.name)
        total_in += p.stat().st_size
        if progress and (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}  {total_out/1e6:.1f} MB out of {total_in/1e6:.1f} MB in")
    n = writer.close()
    size = os.path.getsize(out)
    if writer.size:
        # Comparing against the source would flatter the codec with the
        # downscale, which the codec did not do. Say so instead.
        detail = f" (resized to {writer.size}px first, so not comparable to source files)"
    else:
        detail = f", source was {total_in/1e6:.1f} MB, {total_in/max(1, size):.1f}x smaller"
    print(f"wrote {out}: {n} images, {size/1e6:.1f} MB{detail}")
    return n


# --------------------------------------------------------------------------
# reading


class LatentDataset(Dataset):
    """A torch Dataset yielding latent grids, ready to feed a network.

    Returns (latent, label) where latent is float32 (C, H, W).  Drop it into a
    DataLoader as usual; the file handle is opened lazily per worker, because a
    handle inherited across a fork is shared and the reads interleave.
    """

    def __init__(self, path, checkpoint=None, dequantize=True, transform=None):
        self.path = Path(path)
        self.dequantize = dequantize
        self.transform = transform
        self._fh = None

        with self.path.open("rb") as fh:
            if fh.read(4) != MAGIC:
                raise ValueError(f"{path} is not a json-camera latent shard")
            n = struct.unpack("<I", fh.read(4))[0]
            self.header = json.loads(fh.read(n).decode("utf-8"))
            fh.seek(-8, os.SEEK_END)
            index_at = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(index_at)
            raw = fh.read()[:-8]
            self.items = json.loads(raw.decode("utf-8"))

        cp = checkpoint or self._guess_checkpoint()
        model, _ = codec.load_checkpoint(cp)
        got = codec.model_fingerprint(model)
        want = self.header["model"]["fingerprint"]
        if got != want:
            raise ValueError(
                f"shard was built with model {want}, you loaded {got}.\n"
                "A latent only means anything to the checkpoint that produced it. "
                "Rebuild the shard, or point checkpoint= at the right one.")
        self.tables = codec._tables(model, self.header["precision"])
        self.range_lo = self.tables["range_lo"][:, None, None]
        self.offset = model.prior.offset.detach().numpy()[:, None, None]
        self.classes = sorted({it["l"] for it in self.items})

    def __getstate__(self):
        """Drop the file handle when this is pickled to a worker.

        DataLoader pickles the dataset to send it to each worker process, and an
        open file handle cannot be pickled at all.  Worse, if it could, the
        workers would share one handle and their seeks would interleave into
        garbage.  Each worker reopens lazily on first access.
        """
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    @staticmethod
    def _guess_checkpoint():
        for c in ("checkpoints/stable/jc-final.pt", "checkpoints/jc.best.pt"):
            if Path(c).exists():
                return c
        raise FileNotFoundError("pass checkpoint= explicitly")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self._fh is None:                       # per worker, see the docstring
            self._fh = self.path.open("rb")
        it = self.items[i]
        self._fh.seek(it["o"])
        blob = self._fh.read(it["n"])

        C, H, W = it["c"], it["h"], it["w"]
        chans = np.repeat(np.arange(C, dtype=np.int32), H * W)
        s = rans.decode(blob, chans, self.tables["freqs"], self.tables["starts"],
                        self.tables["lut"], self.header["precision"],
                        it["lanes"], C * H * W)
        s = s.reshape(C, H, W) + self.range_lo
        y = s.astype(np.float32)
        if self.dequantize:
            y = y + self.offset.astype(np.float32)
        out = torch.from_numpy(y)
        if self.transform:
            out = self.transform(out)
        return out, it["l"]

    @property
    def latent_shape(self):
        it = self.items[0]
        return (it["c"], it["h"], it["w"])

    def to_pixels(self, i, checkpoint=None):
        """Decode item `i` back to an image. For inspection, not for training:
        this runs the decoder network, which is the expensive half."""
        model, _ = codec.load_checkpoint(checkpoint or self._guess_checkpoint())
        y, _ = self[i]
        with torch.no_grad():
            x = model.decoder(y.unsqueeze(0)).clamp(0, 1)
        arr = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)
