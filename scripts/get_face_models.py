"""Fetch the two face models. About 38 MB, once.

They are not in the repository because 37 MB of weights does not belong in git,
and the feature reports itself unavailable without them rather than breaking
anything. Both come from OpenCV's own model zoo and are Apache-2.0.

    python scripts/get_face_models.py

The Dockerfile runs this at build time so the deployed image carries them.

**Pinned by content hash, not by commit.**  A commit pin says where the bytes
came from; a hash pin says what they are, which is the thing that actually
matters here.  If either model silently changed, every embedding already stored
would stop being comparable to new ones, and a library that had learned who six
people were would re-cluster them into strangers on the next deploy.  So a
mismatch fails the build rather than producing a subtly broken feature.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = (
    ("yunet.onnx",
     f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     232589,
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("sface.onnx",
     f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     38696353,
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "json-camera/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/faces")
    out.mkdir(parents=True, exist_ok=True)
    failed = False

    for name, url, size, digest in MODELS:
        target = out / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f"  {name:<12} already here and verified")
            continue

        try:
            blob = fetch(url)
        except Exception as error:
            print(f"  {name:<12} FAILED to download: {error}")
            failed = True
            continue

        got = hashlib.sha256(blob).hexdigest()
        if got != digest:
            # Refuse rather than write it. A wrong model is worse than no model:
            # the feature would appear to work and group the wrong people.
            print(f"  {name:<12} REFUSED: expected sha256 {digest[:16]}..., "
                  f"got {got[:16]}... ({len(blob):,} bytes, expected {size:,}).")
            print(f"  {'':<12} The upstream model changed. Do not ship this: existing "
                  f"face groupings would no longer match. Check opencv_zoo and update "
                  f"the hash deliberately.")
            failed = True
            continue

        target.write_bytes(blob)
        print(f"  {name:<12} {len(blob):,} bytes  verified {got[:16]}...")

    if failed:
        return 1
    print(f"\nface models ready in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
