"""Command line: prepare / train / encode / decode / eval / convert / restore."""

import argparse
import base64
import io
import math
import os
import pathlib
import sys

import numpy as np
from PIL import Image

from . import codec, formats, lossless, meta
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


def cmd_prepare_images(args):
    """Whole images at several scales, for crops cut fresh during training."""
    from .data import build_image_cache

    build_image_cache(args.images, args.out, patch=args.patch, limit=args.limit,
                      workers=args.workers)


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
                             name=os.path.basename(args.input),
                             preview=not args.no_preview, exif=not args.no_exif,
                             gps=not args.no_gps)
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
    if s.get("preview_bytes"):
        print(f"  preview      {_human(s['preview_bytes'])}   "
              f"(+{s['preview_pct']:.0f}%, so the file can be looked at without a checkpoint)")
    print(f"  ratio        {s['ratio_vs_raw_json']:.0f}x vs raw, "
          f"{os.path.getsize(args.input)/s['json_bytes']:.2f}x vs the source file")
    if doc["codec"]["clipped_symbols"]:
        print(f"  note: {doc['codec']['clipped_symbols']} symbols hit the table edge "
              f"(harmless, slight quality loss)")


def _encode_lossless(args, img):
    doc = lossless.encode_image(img, name=os.path.basename(args.input),
                                preview=not args.no_preview, exif=not args.no_exif,
                                gps=not args.no_gps)
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


# --------------------------------------------------------------------------
# bulk: a folder in, a folder out
#
# One photograph at a time is a demo. A library is thousands, and the two have
# different failure modes: a batch has to survive the one corrupt file in the
# middle, has to be resumable after the laptop sleeps, and has to leave the
# originals alone. Both commands below skip work already done, so re-running
# after an interruption picks up where it stopped instead of starting again.

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}


def _walk(root, suffixes):
    root = pathlib.Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("."))


def cmd_convert(args):
    """Encode every image under a folder, mirroring the tree into --out."""
    src = pathlib.Path(args.images)
    dst = pathlib.Path(args.out)
    files = _walk(src, IMAGE_SUFFIXES)
    if not files:
        print(f"no images under {src}")
        return 1

    model = None
    if not args.lossless:
        model, _ = codec.load_checkpoint(args.checkpoint)

    done = skipped = failed = 0
    src_bytes = out_bytes = 0
    for i, path in enumerate(files, 1):
        rel = path.relative_to(src) if src.is_dir() else pathlib.Path(path.name)
        target = dst / rel.with_suffix(".json")
        if target.exists() and not args.force:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(path)
            img.load()
            if args.max_side and max(img.size) > args.max_side:
                img.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            if args.lossless:
                doc = lossless.encode_image(img, name=path.name, preview=not args.no_preview,
                                            exif=not args.no_exif, gps=not args.no_gps)
            else:
                doc = codec.encode_image(model, img, device=args.device, name=path.name,
                                         preview=not args.no_preview, exif=not args.no_exif,
                                         gps=not args.no_gps)
            codec.write_json(doc, target)
        except Exception as error:
            # One unreadable file must not end a run of three thousand.
            failed += 1
            print(f"  [{i}/{len(files)}] {rel}: {type(error).__name__}: {error}")
            continue
        done += 1
        src_bytes += path.stat().st_size
        out_bytes += target.stat().st_size
        if not args.quiet:
            print(f"  [{i}/{len(files)}] {rel}  {_human(path.stat().st_size)} -> "
                  f"{_human(target.stat().st_size)}")

    print(f"\n{done} converted, {skipped} already done, {failed} failed  ->  {dst}")
    if done:
        print(f"{_human(src_bytes)} of source became {_human(out_bytes)}  "
              f"({src_bytes / max(out_bytes, 1):.1f}x smaller)")
    if not args.no_preview:
        print("Every file carries its own thumbnail, so `jsoncam restore --previews` "
              "can build a contact sheet without a checkpoint.")
    return 0


def _write_exif(doc):
    """Put the capture date, camera and location back on a restored JPEG.

    A round trip that returns the pixels but loses the date has still destroyed
    the library, because a folder of photographs sorted by nothing is not a
    library. JPEG can carry this; PNG has nowhere sensible to put it, which is
    why the file time below is set either way.
    """
    info = doc.get("meta") or {}
    exif = Image.Exif()
    if (camera := info.get("camera")):
        make, _, model_name = camera.partition(" ")
        exif[271] = make
        exif[272] = model_name or camera
    if (captured := info.get("captured_at")):
        stamp = captured.replace("-", ":", 2).replace("T", " ")
        exif[306] = stamp
        exif.get_ifd(0x8769)[36867] = stamp
    if (lens := info.get("lens")):
        exif.get_ifd(0x8769)[42036] = lens
    if (place := info.get("gps")):
        gps = exif.get_ifd(0x8825)
        for value, ref_tag, val_tag, positive, negative in (
                (place["lat"], 1, 2, "N", "S"), (place["lon"], 3, 4, "E", "W")):
            gps[ref_tag] = positive if value >= 0 else negative
            value = abs(value)
            degrees = int(value)
            minutes = int((value - degrees) * 60)
            seconds = round((value - degrees - minutes / 60) * 3600, 4)
            # Plain numbers, not (numerator, denominator) pairs: Pillow does the
            # rational packing itself and chokes on a tuple of tuples.
            gps[val_tag] = (float(degrees), float(minutes), float(seconds))
    return exif.tobytes() if len(exif) else None


def _touch(path, doc):
    """Set the file's modified time to when the shutter opened."""
    captured = (doc.get("meta") or {}).get("captured_at")
    if not captured:
        return
    try:
        import datetime

        when = datetime.datetime.fromisoformat(captured).timestamp()
        os.utime(path, (when, when))
    except (ValueError, OSError):
        pass


def cmd_restore(args):
    """Decode a folder of .json files back into ordinary pictures.

    This is the exit. A format nobody can leave is a place to lose photographs,
    not a place to keep them, so it stays a single command and it never needs an
    explanation. `--previews` is the version that works even when the checkpoint
    is gone: it pulls the embedded thumbnails straight out of the headers, which
    is a lossy rescue but a rescue.
    """
    src = pathlib.Path(args.library)
    dst = pathlib.Path(args.out)
    files = _walk(src, {".json"})
    if not files:
        print(f"no .json files under {src}")
        return 1

    model = None
    done = skipped = failed = 0
    for i, path in enumerate(files, 1):
        rel = path.relative_to(src) if src.is_dir() else pathlib.Path(path.name)
        try:
            doc = codec.read_json(path)
        except Exception as error:
            failed += 1
            print(f"  [{i}/{len(files)}] {rel}: not readable JSON ({error})")
            continue
        if doc.get("format") not in ("json-camera/1", lossless.FORMAT):
            skipped += 1
            continue

        stem = os.path.splitext((doc.get("image") or {}).get("name") or rel.name)[0]
        suffix = ".jpg" if args.format == "jpeg" else ".png"
        target = dst / rel.parent / (stem + suffix)
        if target.exists() and not args.force:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            if args.previews:
                blob, _ = meta.proxy_bytes(doc)
                if not blob:
                    skipped += 1
                    continue
                img = Image.open(io.BytesIO(blob))
                img.load()
            elif doc["format"] == lossless.FORMAT:
                img = lossless.decode_dict(doc)
            else:
                if model is None:
                    model, _ = codec.load_checkpoint(args.checkpoint)
                img = codec.decode_dict(model, doc, device=args.device,
                                        strict=not args.force)
        except Exception as error:
            failed += 1
            print(f"  [{i}/{len(files)}] {rel}: {type(error).__name__}: {error}")
            continue

        save = {}
        if (profile := (doc.get("image") or {}).get("icc_profile")):
            # The picture's colours only mean what the camera meant if its
            # profile travels with them, so carry it back out too.
            save["icc_profile"] = base64.b64decode(profile)
        if args.format == "jpeg":
            save["quality"] = args.quality
            if (raw_exif := _write_exif(doc)):
                save["exif"] = raw_exif
            img = img.convert("RGB")
        img.save(target, **save)
        _touch(target, doc)
        done += 1
        if not args.quiet:
            note = " (preview only)" if args.previews else ""
            print(f"  [{i}/{len(files)}] {rel}  ->  {target.name}  "
                  f"{img.size[0]}x{img.size[1]}{note}")

    print(f"\n{done} restored, {skipped} skipped, {failed} failed  ->  {dst}")
    return 0


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


def _privacy_flags(p):
    """The three switches that decide what rides along beside the pixels.

    They are opt-out rather than opt-in because the defaults are what a person
    converting their own library wants, and the one that actually matters is
    --no-gps: a photograph carries the coordinates of the place it was taken,
    and that survives every copy of the file made afterwards.
    """
    p.add_argument("--no-preview", action="store_true",
                   help="do not embed a thumbnail (saves ~15%%, costs the ability to look at it)")
    p.add_argument("--no-exif", action="store_true",
                   help="drop the capture date, camera and location entirely")
    p.add_argument("--no-gps", action="store_true",
                   help="keep the date and camera, drop where it was taken")


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

    p = sub.add_parser("prepare-images",
                       help="store whole images at several scales; train crops them live")
    p.add_argument("--images", required=True, nargs="+", help="one or more folders")
    p.add_argument("--out", default="data/images.npy")
    p.add_argument("--patch", type=int, default=256, help="drop renderings smaller than this")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(fn=cmd_prepare_images)

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

    p = sub.add_parser("convert", help="encode a whole folder of images, resumably")
    p.add_argument("images", help="folder of photographs, or one file")
    p.add_argument("--out", required=True, help="where the .json files go")
    p.add_argument("-c", "--checkpoint", default="checkpoints/stable/jc-final.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--lossless", action="store_true", help="discard nothing; no checkpoint needed")
    p.add_argument("--max-side", type=int, default=None,
                   help="shrink anything longer than this before encoding")
    p.add_argument("--force", action="store_true", help="redo files that are already converted")
    p.add_argument("--quiet", action="store_true")
    _privacy_flags(p)
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("restore", help="decode a folder of .json back into pictures")
    p.add_argument("library", help="folder of .json files, or one file")
    p.add_argument("--out", required=True, help="where the pictures go")
    p.add_argument("-c", "--checkpoint", default="checkpoints/stable/jc-final.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--format", default="jpeg", choices=["jpeg", "png"],
                   help="jpeg carries the capture date and location back out; png does not")
    p.add_argument("--quality", type=int, default=92, help="jpeg quality (default 92)")
    p.add_argument("--previews", action="store_true",
                   help="pull the embedded thumbnails instead of decoding; needs no checkpoint")
    p.add_argument("--force", action="store_true",
                   help="overwrite, and decode even on a fingerprint mismatch")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_restore)

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
            _privacy_flags(p)
        if name == "decode":
            p.add_argument("--force", action="store_true",
                           help="decode even if the checkpoint fingerprint differs")
        if name == "eval":
            p.add_argument("--vs-json", action="store_true",
                           help="match JPEG to the .json size instead of the raw bitstream")
        p.set_defaults(fn=fn)

    return ap


def main():
    # An iPhone's photographs are HEIC, so a bulk convert of anything copied off
    # a phone needs this before it can open the first file.
    formats.enable_heif()
    argv = sys.argv[1:]
    if argv and argv[0] == "train":
        from . import train
        return train.main(argv[1:])
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
