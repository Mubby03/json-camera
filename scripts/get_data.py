"""Fetch DIV2K to train on.

The canonical ETH host (data.vision.ee.ethz.ch) is frequently throttled to a
few KB/s, which makes the 3.5 GB train set effectively undownloadable.  We
pull the same archive from a HuggingFace mirror instead, which sustains
~8 MB/s.  Falls back to ETH if the mirror is gone.

    python scripts/get_data.py            # 800-image train set (3.5 GB)
    python scripts/get_data.py --valid    # 100-image valid set only
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

MIRROR = "https://huggingface.co/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip"
ETH = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
ETH_VALID = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def hook(t0):
    def inner(count, block, total):
        got = count * block
        el = max(1e-9, time.time() - t0)
        pct = min(100.0, 100.0 * got / total) if total > 0 else 0.0
        sys.stdout.write(
            f"\r  {pct:5.1f}%  {got/1e9:.2f}/{total/1e9:.2f} GB  "
            f"{got/el/1e6:5.1f} MB/s  eta {max(0,(total-got))/max(1,got/el)/60:4.1f}m"
        )
        sys.stdout.flush()
    return inner


def fetch(url, zp):
    print(f"fetching {url}")
    urllib.request.urlretrieve(url, zp, hook(time.time()))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", action="store_true", help="fetch the 100-image valid set instead")
    ap.add_argument("--out", default=None, help="flatten PNGs into this folder")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    name = "DIV2K_valid_HR" if args.valid else "DIV2K_train_HR"
    zp = os.path.join(DEST, name + ".zip")
    target = os.path.join(DEST, name)

    if os.path.isdir(target) and len(os.listdir(target)) > 10:
        print(f"already have {target} ({len(os.listdir(target))} files)")
    else:
        if not os.path.exists(zp):
            urls = [ETH_VALID] if args.valid else [MIRROR, ETH]
            for i, u in enumerate(urls):
                try:
                    fetch(u, zp)
                    break
                except Exception as e:
                    print(f"  failed ({e})")
                    if i == len(urls) - 1:
                        raise
        print("extracting…")
        with zipfile.ZipFile(zp) as z:
            z.extractall(DEST)
        os.remove(zp)

    # The mirror nests the PNGs one level deeper; normalise either shape.
    if not os.path.isdir(target):
        for root, dirs, files in os.walk(DEST):
            if os.path.basename(root) == name:
                target = root
                break
    pngs = [f for f in os.listdir(target) if f.endswith(".png")]
    print(f"done -> {target}  ({len(pngs)} images)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for f in pngs:
            d = os.path.join(args.out, f)
            if not os.path.exists(d):
                shutil.copy2(os.path.join(target, f), d)
        print(f"flattened {len(pngs)} -> {args.out}")


if __name__ == "__main__":
    main()
