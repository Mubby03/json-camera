"""Push the web app to a Hugging Face Space.

    python scripts/deploy_space.py --repo yourname/json-camera

Ships the docker SDK rather than gradio, so the Space runs the real site: the
landing page, the compressor and the decompressor, exactly as they run locally.

A staging directory is assembled rather than pushing the repo as it stands,
because the working tree carries 11 GB of training data and fat checkpoints that
have no business in a Space. Upload goes over the HTTP API, so there is no
git-lfs to install.

Auth, in order of preference:
    huggingface-cli login          (once, stored in ~/.cache/huggingface)
    --token hf_xxx                 (or the HF_TOKEN env var)
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """---
title: json-camera
emoji: 📷
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A neural network that compresses photographs into JSON text.
---

# json-camera

A learned image codec that stores a photograph as a JSON file.

An image goes in, a convolutional network crushes it to a small grid of integers,
a learned entropy model prices those integers in bits, a range coder packs them,
and the result lands in a `.json`. A second network reads that grid back and
rebuilds the picture.

Nothing about the transform is hand designed. Encoder, decoder and entropy model
are trained jointly against a loss that counts the output file size in bits:

```
loss = lambda * distortion + rate
```

Measured over held out DIV2K photographs at about 0.31 bits per pixel, this
checkpoint beats JPEG at a matched file size by 2.21 dB PSNR and 3.46 dB MS-SSIM,
winning on 12 of 12 images.

## Two things to be straight about

**This is compression, not encryption.** The payload looks unreadable, but there
is no key. Anyone holding the checkpoint can read it.

**The weights are part of the file format.** A `.json` is only decodable by the
exact model that encoded it, which is why every file carries a fingerprint and
decoding refuses on a mismatch.

Built by [mubby.space](https://mubby.space). Source: https://github.com/{gh}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. mubbyt/json-camera")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoints to ship (default: checkpoints/stable/*.pt)")
    ap.add_argument("--github", default="Mubby03/json-camera")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not upload")
    args = ap.parse_args()

    cks = args.checkpoints or sorted(glob.glob(os.path.join(ROOT, "checkpoints", "stable", "*.pt")))
    if not cks:
        sys.exit("No checkpoints to ship. Train one, then:\n"
                 "  jsoncam export checkpoints/jc.best.pt -o checkpoints/stable/jc-final.pt")

    stage = tempfile.mkdtemp(prefix="jsoncam-space-")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(os.path.join(ROOT, "jsoncam"), os.path.join(stage, "jsoncam"), ignore=ignore)
    shutil.copytree(os.path.join(ROOT, "web"), os.path.join(stage, "web"), ignore=ignore)
    shutil.copy(os.path.join(ROOT, "Dockerfile"), stage)
    shutil.copy(os.path.join(ROOT, "requirements-space.txt"), stage)

    # Mirrors the repo layout so the one Dockerfile works here and on Cloud Run.
    os.makedirs(os.path.join(stage, "checkpoints", "stable"))
    total = 0
    for c in cks:
        # A training checkpoint carries Adam state at roughly twice the weight
        # size and is useless for inference, so refuse to ship one by accident.
        import torch

        ck = torch.load(c, map_location="cpu", weights_only=False)
        if "opt" in ck:
            sys.exit(f"{c} still has optimiser state. Slim it first:\n"
                     f"  jsoncam export {c} -o checkpoints/stable/{os.path.basename(c)}")
        shutil.copy(c, os.path.join(stage, "checkpoints", "stable"))
        total += os.path.getsize(c)
        print(f"  ship {os.path.basename(c)}  ({os.path.getsize(c)/1e6:.1f} MB)")

    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(README.format(gh=args.github))

    print(f"staged {stage}  ({total/1e6:.1f} MB of weights)")
    if args.dry_run:
        for root, _, files in os.walk(stage):
            for name in sorted(files):
                rel = os.path.relpath(os.path.join(root, name), stage)
                print(f"    {rel}")
        print("dry run, nothing uploaded")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")

    api = HfApi(token=args.token)
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not authenticated. Run `huggingface-cli login`, or pass --token / set HF_TOKEN.\n"
                 "Create a token with WRITE access at https://huggingface.co/settings/tokens")
    print(f"authenticated as {who}")

    api.create_repo(args.repo, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=args.repo, repo_type="space",
                      commit_message="deploy json-camera")
    print(f"\nbuilding at https://huggingface.co/spaces/{args.repo}")
    print("First build takes a few minutes while torch installs.")


if __name__ == "__main__":
    main()
