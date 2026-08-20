"""Score a checkpoint against JPEG over a folder of held-out images.

One image proves nothing -- codecs win on some content and lose on others, and
a single flattering photo is how a codec gets oversold.  This runs the whole
set and reports the spread as well as the mean.
"""

import argparse, io, math, os, statistics, sys, time
import numpy as np
from PIL import Image
from jsoncam import codec
from jsoncam.metrics import from_images as ms_ssim, ms_ssim_db


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def jpeg_at(img, target):
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); n = buf.tell()
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (q, n, buf.getvalue())
        if n < target: lo = q + 1
        else: hi = q - 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/val_images")
    ap.add_argument("-c", "--checkpoint", default="checkpoints/stable/jc-final.pt")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--vs-json", action="store_true",
                    help="match JPEG to the .json size instead of the raw bitstream")
    args = ap.parse_args()

    model, _ = codec.load_checkpoint(args.checkpoint)
    paths = sorted(p for p in os.listdir(args.images) if p.endswith(".png"))[: args.limit]
    rows = []
    print(f"{len(paths)} images, checkpoint {args.checkpoint}\n")
    print(f"{'image':<12}{'bpp':>8}{'ours dB':>10}{'jpeg dB':>10}{'dPSNR':>8}"
          f"{'ours MS':>10}{'jpeg MS':>10}{'dMS dB':>9}")

    for name in paths:
        src = Image.open(os.path.join(args.images, name)).convert("RGB")
        t0 = time.time()
        doc = codec.encode_image(model, src, device="cpu")
        enc = time.time() - t0
        t0 = time.time()
        rec = codec.decode_dict(model, doc, device="cpu")
        dec = time.time() - t0

        bits = doc["codec"]["bitstream_bytes"]
        px = src.size[0] * src.size[1]
        bpp = 8.0 * bits / px
        json_bytes = int(bits * 1.25)          # b85 armour, no temp file needed
        target = json_bytes if args.vs_json else bits
        jq, jn, jb = jpeg_at(src, target)
        jrec = Image.open(io.BytesIO(jb)).convert("RGB")

        op, jp = psnr(src, rec), psnr(src, jrec)
        om, jm = ms_ssim(src, rec), ms_ssim(src, jrec)
        rows.append((bpp, op, jp, ms_ssim_db(om), ms_ssim_db(jm), enc, dec, px))
        print(f"{name:<12}{bpp:>8.4f}{op:>10.2f}{jp:>10.2f}{op-jp:>+8.2f}"
              f"{om:>10.4f}{jm:>10.4f}{ms_ssim_db(om)-ms_ssim_db(jm):>+9.2f}")

    dp = [r[1] - r[2] for r in rows]
    dm = [r[3] - r[4] for r in rows]
    mp = sum(r[7] for r in rows) / len(rows) / 1e6
    print(f"\n  mean bpp        {statistics.mean(r[0] for r in rows):.4f}")
    print(f"  mean PSNR       ours {statistics.mean(r[1] for r in rows):.2f} dB   "
          f"jpeg {statistics.mean(r[2] for r in rows):.2f} dB")
    print(f"  mean MS-SSIM dB ours {statistics.mean(r[3] for r in rows):.2f} dB   "
          f"jpeg {statistics.mean(r[4] for r in rows):.2f} dB")
    print(f"\n  PSNR   vs JPEG  {statistics.mean(dp):+.2f} dB mean, "
          f"range {min(dp):+.2f} to {max(dp):+.2f}, wins {sum(d>0 for d in dp)}/{len(dp)}")
    print(f"  MSSSIM vs JPEG  {statistics.mean(dm):+.2f} dB mean, "
          f"range {min(dm):+.2f} to {max(dm):+.2f}, wins {sum(d>0 for d in dm)}/{len(dm)}")
    print(f"\n  speed           {statistics.mean(r[5] for r in rows):.1f}s encode / "
          f"{statistics.mean(r[6] for r in rows):.1f}s decode  ({mp:.1f} MP average, CPU)")


if __name__ == "__main__":
    main()
