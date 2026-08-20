"""The JSON container: image <-> .json.

Layout of the file is a small readable header plus one packed payload string.
The header is metadata you can eyeball; the payload is the rANS bitstream,
which is incompressible by construction (if it still had structure, the
entropy coder would have squeezed it out).

Note on the text tax: JSON can only hold text, so the bitstream has to be
ASCII-armoured.  base85 costs 25% over the raw bits, base64 costs 33%.  That
overhead is the price of the format being JSON, not a flaw in the compression
-- `stats()` reports both numbers so you can see it.
"""

import base64
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import rans
from .model import JSONCamera

DOWNSCALE = 16          # four stride-2 convolutions
TILE = 512              # pixels per tile, keeps peak memory flat
MARGIN = 96             # context around each tile so seams do not show

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------
# packing


def _pack(blob, encoding):
    if encoding == "b85":
        return base64.b85encode(blob).decode("ascii")
    if encoding == "b64":
        return base64.b64encode(blob).decode("ascii")
    raise ValueError(f"unknown encoding {encoding!r}")


def _unpack(text, encoding):
    if encoding == "b85":
        return base64.b85decode(text.encode("ascii"))
    if encoding == "b64":
        return base64.b64decode(text.encode("ascii"))
    raise ValueError(f"unknown encoding {encoding!r}")


# --------------------------------------------------------------------------
# tiled application of the conv nets


def _tiled_encode(model, x, tile=TILE, margin=MARGIN, device="cpu"):
    """Run the encoder over a big image without materialising huge activations.

    Each tile is fed with `margin` pixels of genuine neighbouring image so the
    convolutions see the same context they would in a single pass; the extra
    latents are then discarded.  That is what keeps tile boundaries invisible.
    """
    _, _, H, W = x.shape
    lh, lw = H // DOWNSCALE, W // DOWNSCALE
    out = torch.zeros(1, model.latent, lh, lw)

    for i in range(0, H, tile):
        for j in range(0, W, tile):
            i1, j1 = min(i + tile, H), min(j + tile, W)
            a = max(0, i - margin)
            b = min(H, i1 + margin)
            c = max(0, j - margin)
            d = min(W, j1 + margin)
            patch = x[:, :, a:b, c:d].to(device)
            y = model.encoder(patch).cpu()
            oi, oj = (i - a) // DOWNSCALE, (j - c) // DOWNSCALE
            h, w = (i1 - i) // DOWNSCALE, (j1 - j) // DOWNSCALE
            out[:, :, i // DOWNSCALE : i // DOWNSCALE + h, j // DOWNSCALE : j // DOWNSCALE + w] = \
                y[:, :, oi : oi + h, oj : oj + w]
    return out


def _tiled_decode(model, y, tile=TILE, margin=MARGIN, device="cpu"):
    """Mirror of _tiled_encode, working in latent coordinates."""
    _, _, lh, lw = y.shape
    H, W = lh * DOWNSCALE, lw * DOWNSCALE
    lt, lm = tile // DOWNSCALE, margin // DOWNSCALE
    out = torch.zeros(1, 3, H, W)

    for i in range(0, lh, lt):
        for j in range(0, lw, lt):
            i1, j1 = min(i + lt, lh), min(j + lt, lw)
            a, b = max(0, i - lm), min(lh, i1 + lm)
            c, d = max(0, j - lm), min(lw, j1 + lm)
            patch = y[:, :, a:b, c:d].to(device)
            x = model.decoder(patch).cpu()
            oi, oj = (i - a) * DOWNSCALE, (j - c) * DOWNSCALE
            h, w = (i1 - i) * DOWNSCALE, (j1 - j) * DOWNSCALE
            out[:, :, i * DOWNSCALE : i * DOWNSCALE + h, j * DOWNSCALE : j * DOWNSCALE + w] = \
                x[:, :, oi : oi + h, oj : oj + w]
    return out


# --------------------------------------------------------------------------
# tables


def _tables(model, precision=12):
    """Frequency tables for the range coder.

    Always computed on CPU: encoder and decoder MUST agree bit for bit, and
    float kernels are not guaranteed identical across CPU/MPS/CUDA.  A
    one-ULP difference here would desync the coder and corrupt the image.
    """
    prior = copy.deepcopy(model.prior).to("cpu").eval()
    t = prior.build_tables(precision=precision)
    starts, lut = rans.build_luts(t["freqs"])
    t["starts"], t["lut"] = starts, lut
    return t


def model_fingerprint(model):
    """Decode is only valid with the exact weights that encoded."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# public API


@torch.no_grad()
def encode_image(model, img, encoding="b85", precision=12, device="cpu", tile=TILE):
    """PIL image -> JSON-ready dict."""
    model.eval()
    model.to(device)
    prior_cpu = copy.deepcopy(model.prior).cpu().eval()
    img = img.convert("RGB")
    W, H = img.size
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Pad up to a multiple of the downscale factor; replicate rather than zero
    # so the border does not invent a hard black edge for the net to spend bits on.
    ph = (-H) % DOWNSCALE
    pw = (-W) % DOWNSCALE
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")

    y = _tiled_encode(model, x, tile=tile, device=device)
    s = prior_cpu.symbols(y)[0].numpy().astype(np.int64)  # (C, lh, lw)

    t = _tables(model, precision)
    lo = t["range_lo"][:, None, None]
    hi = t["range_hi"][:, None, None]
    clipped = int(np.sum((s < lo) | (s > hi)))
    s = np.clip(s, lo, hi) - lo  # shift to 0-based table indices

    C, lh, lw = s.shape
    chans = np.repeat(np.arange(C, dtype=np.int64), lh * lw)
    blob, lanes = rans.encode(s.reshape(-1), chans, t["freqs"], t["starts"], precision)

    payload = _pack(blob, encoding)
    return {
        "format": "json-camera/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {**model.config, "fingerprint": model_fingerprint(model)},
        "image": {"width": W, "height": H},
        "latent": {"channels": C, "height": lh, "width": lw},
        "codec": {
            "kind": "rans",
            "precision": precision,
            "lanes": lanes,
            "count": int(s.size),
            "bitstream_bytes": len(blob),
            "clipped_symbols": clipped,
        },
        "payload": {"encoding": encoding, "data": payload},
    }


@torch.no_grad()
def decode_dict(model, doc, device="cpu", tile=TILE, strict=True):
    """JSON dict -> PIL image."""
    model.eval()
    model.to(device)
    prior_cpu = copy.deepcopy(model.prior).cpu().eval()
    if doc.get("format") != "json-camera/1":
        raise ValueError(f"not a json-camera file: {doc.get('format')!r}")

    fp = doc["model"].get("fingerprint")
    mine = model_fingerprint(model)
    if strict and fp and fp != mine:
        raise ValueError(
            f"checkpoint mismatch: file was encoded with model {fp}, you loaded {mine}.\n"
            "The weights ARE the codebook -- decoding needs the exact same ones."
        )

    cdc = doc["codec"]
    precision = cdc["precision"]
    C = doc["latent"]["channels"]
    lh, lw = doc["latent"]["height"], doc["latent"]["width"]

    t = _tables(model, precision)
    blob = _unpack(doc["payload"]["data"], doc["payload"]["encoding"])
    chans = np.repeat(np.arange(C, dtype=np.int64), lh * lw)
    s = rans.decode(blob, chans, t["freqs"], t["starts"], t["lut"],
                    precision, cdc["lanes"], cdc["count"])
    s = s.reshape(C, lh, lw) + t["range_lo"][:, None, None]

    y = prior_cpu.dequantize(torch.from_numpy(s).unsqueeze(0))
    x = _tiled_decode(model, y, tile=tile, device=device)

    H, W = doc["image"]["height"], doc["image"]["width"]
    x = x[:, :, :H, :W].clamp(0, 1)
    arr = (x[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def write_json(doc, path):
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def stats(doc, json_path=None):
    """Where the bytes actually went."""
    W, H = doc["image"]["width"], doc["image"]["height"]
    px = W * H
    raw = px * 3
    bits = doc["codec"]["bitstream_bytes"]
    out = {
        "pixels": px,
        "raw_bytes": raw,
        "bitstream_bytes": bits,
        "bpp": 8.0 * bits / px,
        "ratio_vs_raw": raw / bits,
    }
    if json_path is not None:
        import os

        jb = os.path.getsize(json_path)
        out["json_bytes"] = jb
        out["text_overhead_pct"] = 100.0 * (jb - bits) / bits
        out["ratio_vs_raw_json"] = raw / jb
    return out


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {"hidden": 128, "latent": 192})
    model = JSONCamera(**cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
