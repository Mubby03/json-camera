"""json-camera: a learned image codec, and a way to train on its output.

Three things live here, and they answer different questions.

**Lossy codec.**  A convolutional encoder, a learned entropy model and a range
coder.  Reaches roughly 60x by deciding what to discard, and beats JPEG at a
matched file size.

    import jsoncam
    doc = jsoncam.encode("photo.jpg")
    jsoncam.decode(doc, "restored.png")

**Lossless codec.**  No network at all: a reversible colour transform, a
predictor, and the same range coder.  Returns the picture bit for bit, about 20%
under PNG.  Use it when nothing may be thrown away.

    doc = jsoncam.encode_lossless("photo.png")

**Compressed-domain training.**  The interesting one.  Instead of decoding to
pixels, hand a model the latent grid directly.  It is six times smaller than the
pixels it stands for, so training steps run about nine times faster, and the
dataset on disk is about six times smaller than the same images as JPEG.

    jsoncam.prepare_dataset("photos/", "train.jcl")
    ds = jsoncam.LatentDataset("train.jcl")

A latent only means anything to the checkpoint that produced it.  Shards record
a model fingerprint and refuse to open under a different one, so retraining the
codec means rebuilding the shards.
"""

from pathlib import Path

__version__ = "0.2.0"

__all__ = [
    "encode", "decode", "encode_lossless", "decode_lossless",
    "load_checkpoint", "stats", "prepare_dataset", "LatentDataset",
    "JSONCamera", "psnr", "ms_ssim", "__version__",
]

DEFAULT_CHECKPOINT = "checkpoints/stable/jc-final.pt"


def _as_image(src):
    from PIL import Image

    return src if hasattr(src, "size") and hasattr(src, "convert") else Image.open(src)


def _resolve(checkpoint):
    if checkpoint:
        return checkpoint
    for c in (DEFAULT_CHECKPOINT, "checkpoints/jc.best.pt"):
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        "No checkpoint found. Pass checkpoint=..., or train one with `jsoncam train`.")


def encode(image, out=None, checkpoint=None, encoding="b85", device="cpu"):
    """Compress an image with the learned codec.

    `image` is a path or a PIL image.  Returns the container dict; also writes it
    if `out` is given.
    """
    from . import codec

    model, _ = codec.load_checkpoint(_resolve(checkpoint))
    img = _as_image(image)
    name = Path(image).name if isinstance(image, (str, Path)) else None
    doc = codec.encode_image(model, img, encoding=encoding, device=device, name=name)
    if out:
        codec.write_json(doc, out)
    return doc


def decode(doc, out=None, checkpoint=None, device="cpu"):
    """Rebuild an image. Accepts a container dict or a path to one.

    Detects lossless files and routes them itself, so callers do not have to
    know which mode produced the file.
    """
    from . import codec, lossless

    if isinstance(doc, (str, Path)):
        doc = codec.read_json(doc)
    if doc.get("format") == lossless.FORMAT:
        img = lossless.decode_dict(doc)
    else:
        model, _ = codec.load_checkpoint(_resolve(checkpoint))
        img = codec.decode_dict(model, doc, device=device)
    if out:
        img.save(out, icc_profile=img.info.get("icc_profile"))
    return img


def encode_lossless(image, out=None):
    """Compress with nothing discarded. Needs no checkpoint."""
    from . import codec, lossless

    name = Path(image).name if isinstance(image, (str, Path)) else None
    doc = lossless.encode_image(_as_image(image), name=name)
    if out:
        codec.write_json(doc, out)
    return doc


def decode_lossless(doc, out=None):
    """Rebuild a lossless file, bit for bit."""
    from . import codec, lossless

    if isinstance(doc, (str, Path)):
        doc = codec.read_json(doc)
    img = lossless.decode_dict(doc)
    if out:
        img.save(out, icc_profile=img.info.get("icc_profile"))
    return img


def stats(doc, json_path=None):
    """Where the bytes went, for either format."""
    from . import codec, lossless

    if doc.get("format") == lossless.FORMAT:
        return lossless.stats(doc, json_path)
    return codec.stats(doc, json_path)


def psnr(a, b):
    """Peak signal to noise ratio between two images, in dB."""
    import math

    import numpy as np

    x = np.asarray(_as_image(a).convert("RGB"), np.float64)
    y = np.asarray(_as_image(b).convert("RGB"), np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def ms_ssim(a, b):
    """Multi-scale SSIM between two images, 0 to 1."""
    from .metrics import from_images

    return from_images(_as_image(a).convert("RGB"), _as_image(b).convert("RGB"))


def __getattr__(name):
    # Deferred so `import jsoncam` stays fast and does not pull in torch until
    # something actually needs it.
    if name == "prepare_dataset":
        from .dataset import prepare_dataset

        return prepare_dataset
    if name == "LatentDataset":
        from .dataset import LatentDataset

        return LatentDataset
    if name == "JSONCamera":
        from .model import JSONCamera

        return JSONCamera
    if name == "load_checkpoint":
        from .codec import load_checkpoint

        return load_checkpoint
    raise AttributeError(f"module 'jsoncam' has no attribute {name!r}")
