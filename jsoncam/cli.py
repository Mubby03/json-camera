"""Command line: prepare / train / encode / decode / eval."""

import argparse
import io
import math
import os
import sys

import numpy as np
from PIL import Image

from . import codec, lossless
from .metrics import from_images as _ms_ssim, ms_ssim_db


def _psnr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024.0


def cmd_prepare(args):
    from .data import build_patch_cache

    build_patch_cache(args.images, args.out, patch=args.patch,
                      per_image=args.per_image, limit=args.limit)


def cmd_train(args, extra):
    from . import train

    train.main(extra)


def cmd_prepare_latents(args):
    """Encode a folder of images into one shard of latents for training."""
    from .dataset import prepare_dataset

    prepare_dataset(args.images, args.out, checkpoint=args.checkpoint,
                    size=args.size, limit=args.limit)


def cmd_export(args):
    """Strip a training checkpoint down to just what decoding needs.

    Training checkpoints carry Adam's optimiser state, which is ~2x the size of
    the weights and completely useless for inference.
    """
    import torch

    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    slim = {"model": ck["model"], "config": ck["config"],
            "lmbda": ck.get("lmbda"), "metrics": ck.get("metrics"),
            "epoch": ck.get("epoch")}
    torch.save(slim, args.output)
    before, after = os.path.getsize(args.input), os.path.getsize(args.output)
    model, _ = codec.load_checkpoint(args.output)
    print(f"{args.input} ({_human(before)})  ->  {args.output} ({_human(after)})")
    print(f"  config      {ck['config']}")
    print(f"  fingerprint {codec.model_fingerprint(model)}")
    if ck.get("metrics"):
        print(f"  metrics     {ck['metrics']}")


def cmd_encode(args):
    img = Image.open(args.input)
    if args.lossless:
        return _encode_lossless(args, img)
    model, ck = codec.load_checkpoint(args.checkpoint)
    doc = codec.encode_image(model, img, encoding=args.encoding,
                             precision=args.precision, device=args.device,
                             name=os.path.basename(args.input))
    out = args.output or os.path.splitext(args.input)[0] + ".json"
    codec.write_json(doc, out)
    s = codec.stats(doc, out)
    print(f"{args.input}  ->  {out}")
    print(f"  image        {doc['image']['width']}x{doc['image']['height']}  "
          f"({_human(s['raw_bytes'])} raw RGB)")
    print(f"  latent grid  {doc['latent']['channels']}x{doc['latent']['height']}x{doc['latent']['width']}"
          f"  ({doc['codec']['count']:,} numbers)")
    print(f"  bitstream    {_human(s['bitstream_bytes'])}   {s['bpp']:.4f} bpp")
    print(f"  json file    {_human(s['json_bytes'])}   (+{s['text_overhead_pct']:.0f}% text armour)")
    print(f"  ratio        {s['ratio_vs_raw_json']:.0f}x vs raw, "
          f"{os.path.getsize(args.input)/s['json_bytes']:.2f}x vs the source file")
    if doc["codec"]["clipped_symbols"]:
        print(f"  note: {doc['codec']['clipped_symbols']} symbols hit the table edge "
              f"(harmless, slight quality loss)")


def _encode_lossless(args, img):
    doc = lossless.encode_image(img, name=os.path.basename(args.input))
    out = args.output or os.path.splitext(args.input)[0] + ".json"
    codec.write_json(doc, out)
    s = lossless.stats(doc, out)
    png = io.BytesIO()
    img.convert("RGB").save(png, "PNG", optimize=True)
    print(f"{args.input}  ->  {out}   (lossless, nothing discarded)")
    print(f"  image        {doc['image']['width']}x{doc['image']['height']}  "
          f"({_human(s['raw_bytes'])} raw RGB)")
    print(f"  bitstream    {_human(s['bitstream_bytes'])}   {s['bpp']:.4f} bpp   "
          f"{100 * (1 - s['bitstream_bytes'] / png.tell()):+.0f}% vs PNG ({_human(png.tell())})")
    print(f"  json file    {_human(s['json_bytes'])}   (+25% text armour, which "
          f"cancels the win: the container costs what the coder saves)")


def cmd_decode(args):
    doc = codec.read_json(args.input)
    if doc.get("format") == lossless.FORMAT:
        img = lossless.decode_dict(doc)
        stem = os.path.splitext(doc.get("image", {}).get("name")
                                or os.path.basename(args.input))[0]
        out = args.output or os.path.join(os.path.dirname(args.input) or ".", stem + ".png")
        img.save(out, icc_profile=img.info.get("icc_profile"))
        print(f"{args.input}  ->  {out}   ({img.size[0]}x{img.size[1]}, bit exact)")
        return
    model, ck = codec.load_checkpoint(args.checkpoint)
    img = codec.decode_dict(model, doc, device=args.device, strict=not args.force)
    # Prefer the name the picture went in with over the name of the .json.
    stem = os.path.splitext(doc.get("image", {}).get("name") or os.path.basename(args.input))[0]
    out = args.output or os.path.join(os.path.dirname(args.input) or ".", stem + ".png")
    img.save(out, icc_profile=img.info.get("icc_profile"))
    print(f"{args.input}  ->  {out}   ({img.size[0]}x{img.size[1]})")


def cmd_eval(args):
    """Encode, decode, and score it honestly against JPEG at the same size."""
    model, ck = codec.load_checkpoint(args.checkpoint)
    src = Image.open(args.input).convert("RGB")

    doc = codec.encode_image(model, src, encoding=args.encoding, device=args.device)
    tmp = args.output or os.path.join("out", os.path.basename(args.input) + ".json")
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    codec.write_json(doc, tmp)
    rec = codec.decode_dict(model, doc, device=args.device)
    s = codec.stats(doc, tmp)

    ours_psnr = _psnr(src, rec)
    ours_ms = _ms_ssim(src, rec)
    png = os.path.splitext(tmp)[0] + ".decoded.png"
    rec.save(png)

    # Binary-search JPEG quality for the closest file size, then compare PSNR.
    target = s["json_bytes"] if args.vs_json else s["bitstream_bytes"]
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        src.save(buf, "JPEG", quality=q)
        n = buf.tell()
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (q, n, buf.getvalue())
        if n < target:
            lo = q + 1
        else:
            hi = q - 1
    jq, jn, jbytes = best
    jrec = Image.open(io.BytesIO(jbytes)).convert("RGB")
    jpeg_psnr = _psnr(src, jrec)
    jpeg_ms = _ms_ssim(src, jrec)

    print(f"\n{args.input}   {src.size[0]}x{src.size[1]}")
    print(f"  {'':<14} {'size':>10}  {'rate':>10}   {'PSNR':>8}   {'MS-SSIM':>16}")
    print(f"  {'json-camera':<14} {_human(s['json_bytes']):>10}  "
          f"{s['bpp']:.4f} bpp   {ours_psnr:5.2f} dB   "
          f"{ours_ms:.4f} ({ms_ssim_db(ours_ms):5.2f} dB)")
    print(f"  {'JPEG q=' + str(jq):<14} {_human(jn):>10}  "
          f"{8*jn/s['pixels']:.4f} bpp   {jpeg_psnr:5.2f} dB   "
          f"{jpeg_ms:.4f} ({ms_ssim_db(jpeg_ms):5.2f} dB)")
    d = ours_psnr - jpeg_psnr
    dm = ms_ssim_db(ours_ms) - ms_ssim_db(jpeg_ms)
    print(f"  -> PSNR    {abs(d):.2f} dB {'better' if d > 0 else 'worse'} than JPEG at matched size")
    print(f"  -> MS-SSIM {abs(dm):.2f} dB {'better' if dm > 0 else 'worse'} than JPEG at matched size")
    print(f"  wrote {tmp} and {png}")


def build_parser():
    ap = argparse.ArgumentParser("jsoncam", description="Learned image codec that writes JSON.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="extract training patches from an image folder")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="data/patches.npy")
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--per-image", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train a model (see `jsoncam train --help`)")
    p.set_defaults(fn=None)

    p = sub.add_parser("prepare-latents",
                       help="encode a folder into latents, for compressed-domain training")
    p.add_argument("images")
    p.add_argument("--out", required=True, help="output .jcl shard")
    p.add_argument("-c", "--checkpoint", default="checkpoints/stable/jc-final.pt")
    p.add_argument("--size", type=int, default=None,
                   help="resize to NxN first, as a training pipeline usually would")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_prepare_latents)

    p = sub.add_parser("export", help="strip optimiser state for shipping")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_export)

    for name, fn, needs_out in (("encode", cmd_encode, True), ("decode", cmd_decode, True),
                                ("eval", cmd_eval, True)):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("-o", "--output", default=None)
        p.add_argument("-c", "--checkpoint", default="checkpoints/jc.best.pt")
        p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
        if name in ("encode", "eval"):
            p.add_argument("--encoding", default="b85", choices=["b85", "b64"])
        if name == "encode":
            p.add_argument("--precision", type=int, default=12)
            p.add_argument("--lossless", action="store_true",
                           help="discard nothing; no model needed, about 20%% under PNG")
        if name == "decode":
            p.add_argument("--force", action="store_true",
                           help="decode even if the checkpoint fingerprint differs")
        if name == "eval":
            p.add_argument("--vs-json", action="store_true",
                           help="match JPEG to the .json size instead of the raw bitstream")
        p.set_defaults(fn=fn)

    return ap


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "train":
        from . import train
        return train.main(argv[1:])
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
