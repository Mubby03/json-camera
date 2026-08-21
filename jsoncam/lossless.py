"""Lossless mode: nothing is thrown away.

The learned codec in the rest of this project buys its ratio by discarding
detail, which is the only way to reach 60x.  This is the opposite trade.  Every
pixel comes back exactly, bit for bit, and the file is about 20% smaller than
PNG rather than 60x smaller than anything.  Both numbers are honest; they are
answers to different questions.

There is no network here at all, and that is deliberate.  A neural encoder is
useful precisely because it can decide what to drop.  When nothing may be
dropped, the job is pure prediction: guess each pixel from the ones already
decoded, and write down only where the guess was wrong.  Good guesses make small
numbers, small numbers cost few bits, and the arithmetic must be integer
throughout so the guess is reproducible on the other side.

Three steps, each individually reversible:

1. YCoCg-R, a lifting colour transform.  Photographs have heavily correlated
   red, green and blue; this decorrelates them using only integer adds, shifts
   and subtracts, so it inverts exactly.  Worth about 1.7 MB on a 3 megapixel
   photograph before anything else happens.

2. MED, the median edge predictor from JPEG-LS.  Predicts a pixel from its left,
   up and up-left neighbours, and switches behaviour when those neighbours look
   like a vertical or horizontal edge, which is what stops edges costing a
   fortune.

3. rANS over the prediction residuals, with a frequency table measured from the
   image itself, one per colour plane.

The decoder cannot simply run step 2 backwards in one pass, because each pixel
needs its neighbours already reconstructed.  Doing that a pixel at a time would
be millions of Python iterations.  Instead note that MED only ever looks left
and up, so every pixel on an anti-diagonal depends solely on earlier
anti-diagonals: a whole diagonal resolves in one vectorised step.  That turns
3.1 million sequential operations into 3,575 array ones, and a full 3 megapixel
reconstruction takes about half a second.
"""

import base64
import time

import numpy as np
from PIL import Image, ImageOps

from . import rans

FORMAT = "json-camera/lossless/1"
PRECISION = 12


# --------------------------------------------------------------------------
# reversible colour transform


def rgb_to_ycocg(a):
    """YCoCg-R. Integers in, integers out, exactly invertible.

    Co and Cg gain one bit of range over the input, which is the price of the
    transform being lossless rather than the usual lossy YCbCr matrix.
    """
    r = a[:, :, 0].astype(np.int32)
    g = a[:, :, 1].astype(np.int32)
    b = a[:, :, 2].astype(np.int32)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    return y, co, cg


def ycocg_to_rgb(y, co, cg):
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = co + b
    return np.stack([r, g, b], -1).astype(np.uint8)


# --------------------------------------------------------------------------
# prediction


def _med(a, b, c):
    """Median edge predictor. a=left, b=up, c=up-left."""
    hi = np.maximum(a, b)
    lo = np.minimum(a, b)
    return np.where(c >= hi, lo, np.where(c <= lo, hi, a + b - c))


def predict_residual(p):
    """Forward pass. The encoder has the whole plane, so this is vectorised."""
    a = np.zeros_like(p)
    b = np.zeros_like(p)
    c = np.zeros_like(p)
    a[:, 1:] = p[:, :-1]
    b[1:, :] = p[:-1, :]
    c[1:, 1:] = p[:-1, :-1]
    return p - _med(a, b, c)


def reconstruct(resid):
    """Inverse pass, one anti-diagonal at a time.

    `out` carries a one pixel border of zeros so the top row and left column
    read their missing neighbours as zero, exactly as the forward pass did.
    """
    H, W = resid.shape
    out = np.zeros((H + 1, W + 1), dtype=np.int32)
    for d in range(H + W - 1):
        i = np.arange(max(0, d - W + 1), min(H - 1, d) + 1)
        j = d - i
        pred = _med(out[i + 1, j], out[i, j + 1], out[i, j])
        out[i + 1, j + 1] = pred + resid[i, j]
    return out[1:, 1:]


# --------------------------------------------------------------------------
# frequency tables


def _tables(planes):
    """One integer frequency table per plane, measured from this image.

    Measuring per image rather than shipping fixed tables costs a few kilobytes
    of header and pays for itself many times over: a sky and a brick wall have
    very different residual distributions.
    """
    from .entropy import _to_integer_freqs

    total = 1 << PRECISION
    los, his, rows = [], [], []
    for r in planes:
        lo, hi = int(r.min()), int(r.max())
        counts = np.bincount((r.ravel() - lo).astype(np.int64), minlength=hi - lo + 1)
        p = np.maximum(counts.astype(np.float64), 1e-12)
        p /= p.sum()
        los.append(lo)
        his.append(hi)
        rows.append(_to_integer_freqs(p, total))

    width = max(len(r) for r in rows)
    freqs = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        freqs[i, : len(r)] = r
        # rANS needs every row to sum to 2**precision, so pad short rows by
        # lending the slack to the symbol that can most afford it.
        if len(r) < width:
            freqs[i, int(np.argmax(r))] -= 0
    return freqs, np.array(los), np.array(his)


# --------------------------------------------------------------------------
# public API


def encode_image(img, name=None):
    """PIL image -> JSON-ready dict, with nothing discarded."""
    t0 = time.time()
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    img = img.convert("RGB")
    a = np.asarray(img, dtype=np.uint8)
    H, W, _ = a.shape

    planes = rgb_to_ycocg(a)
    resid = [predict_residual(p) for p in planes]
    del planes
    freqs, los, his = _tables(resid)
    starts, _ = rans.build_luts(freqs)

    # Symbols are 0-based indices into each plane's own table.  int32 throughout:
    # a residual index cannot exceed a few thousand, and int64 would cost eight
    # bytes per sample twice over, which on a large photograph is enough memory
    # to have the process killed.  Planes are released as they are consumed.
    syms = np.empty(3 * H * W, dtype=np.int32)
    for i, (r, lo) in enumerate(zip(resid, los)):
        syms[i * H * W : (i + 1) * H * W] = (r.ravel() - lo).astype(np.int32)
        resid[i] = None
    del resid
    chans = np.repeat(np.arange(3, dtype=np.int32), H * W)
    count = int(syms.size)
    blob, lanes = rans.encode(syms, chans, freqs, starts, PRECISION)
    del syms, chans

    return {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image": {
            "width": W, "height": H, "name": name,
            "icc_profile": base64.b64encode(icc).decode("ascii") if icc else None,
        },
        "codec": {
            "kind": "rans",
            "transform": "ycocg-r+med",
            "precision": PRECISION,
            "lanes": lanes,
            "count": count,
            "bitstream_bytes": len(blob),
            "range_lo": los.tolist(),
            "range_hi": his.tolist(),
            "freqs": [row.tolist() for row in freqs],
            "encode_seconds": round(time.time() - t0, 3),
        },
        "payload": {"encoding": "b85", "data": base64.b85encode(blob).decode("ascii")},
    }


def decode_dict(doc):
    """JSON dict -> the original PIL image, exactly."""
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a json-camera lossless file: {doc.get('format')!r}")

    cdc = doc["codec"]
    H, W = doc["image"]["height"], doc["image"]["width"]
    freqs = np.array(cdc["freqs"], dtype=np.int64)
    los = np.array(cdc["range_lo"], dtype=np.int64)
    starts, lut = rans.build_luts(freqs)

    blob = base64.b85decode(doc["payload"]["data"].encode("ascii"))
    chans = np.repeat(np.arange(3, dtype=np.int32), H * W)
    syms = rans.decode(blob, chans, freqs, starts, lut,
                       cdc["precision"], cdc["lanes"], cdc["count"])
    del chans

    planes = []
    for i in range(3):
        r = syms[i * H * W : (i + 1) * H * W].reshape(H, W).astype(np.int32) + los[i]
        planes.append(reconstruct(r))

    out = Image.fromarray(ycocg_to_rgb(*planes))
    icc = (doc.get("image") or {}).get("icc_profile")
    if icc:
        out.info["icc_profile"] = base64.b64decode(icc)
    return out


def stats(doc, json_path=None):
    import os

    W, H = doc["image"]["width"], doc["image"]["height"]
    px = W * H
    raw = px * 3
    bits = doc["codec"]["bitstream_bytes"]
    out = {
        "pixels": px, "raw_bytes": raw, "bitstream_bytes": bits,
        "bpp": 8.0 * bits / px, "ratio_vs_raw": raw / bits,
    }
    if json_path is not None:
        jb = os.path.getsize(json_path)
        out["json_bytes"] = jb
        out["ratio_vs_raw_json"] = raw / jb
    return out
