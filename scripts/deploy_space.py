"""Push the demo to a Hugging Face Space.

Assembles a clean staging directory (the repo has training data and fat
checkpoints we do not want to upload) and pushes it with the HTTP API, so no
git-lfs install is needed.

    python scripts/deploy_space.py --repo yourname/json-camera

Auth, in order of preference:
    huggingface-cli login          (once, stored in ~/.cache/huggingface)
    --token hf_xxx                 (or the HF_TOKEN env var)
"""

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """---
title: json-camera
emoji: 📷
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: {sdk}
app_file: app.py
pinned: false
license: mit
short_description: A neural network that compresses photographs into JSON text.
---

# json-camera

A learned image codec that stores a photograph as a JSON file.

An image goes in, a convolutional network crushes it to a small grid of integers,
a learned entropy model turns those integers into the fewest bits they can honestly
be written in, and the result lands in a `.json`. A second network reads that grid
back and rebuilds the picture.

Nothing about the transform is hand-designed. The network learns what to keep and
what to throw away, trained against a loss that counts the output file's size in bits:

```
loss = lambda * distortion + rate
```

See the **How it works** tab in the app.

## Two things to be straight about

**This is compression, not encryption.** It looks unreadable, but there is no key.
Anyone with the checkpoint reads it.

**The weights are part of the file format.** A `.json` is only decodable by the exact
model that encoded it, which is why each file carries a fingerprint.

Source: https://github.com/{gh}
"""

SPACE_REQS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pillow
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/json-camera")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoints to ship (default: checkpoints/*.slim.pt)")
    ap.add_argument("--github", default="yourname/json-camera")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not upload")
    args = ap.parse_args()

    try:
        import gradio
        from huggingface_hub import HfApi
    except ImportError as e:
        sys.exit(f"missing dependency: {e}. pip install gradio huggingface_hub")

    cks = args.checkpoints
    if not cks:
        import glob
        cks = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.slim.pt")))
    if not cks:
        sys.exit("No checkpoints to ship. Train one, then:\n"
                 "  jsoncam export checkpoints/jc.best.pt -o checkpoints/jc.slim.pt")

    stage = tempfile.mkdtemp(prefix="jsoncam-space-")
    shutil.copy(os.path.join(ROOT, "app.py"), stage)
    shutil.copytree(os.path.join(ROOT, "jsoncam"), os.path.join(stage, "jsoncam"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(stage, "checkpoints"))
    total = 0
    for c in cks:
        shutil.copy(c, os.path.join(stage, "checkpoints"))
        total += os.path.getsize(c)
        print(f"  ship {os.path.basename(c)}  ({os.path.getsize(c)/1e6:.1f} MB)")

    with open(os.path.join(stage, "requirements.txt"), "w") as f:
        f.write(SPACE_REQS)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(README.format(sdk=gradio.__version__, gh=args.github))

    print(f"staged {stage}  ({total/1e6:.1f} MB of weights)")
    if args.dry_run:
        print("dry run -- not uploading")
        return

    api = HfApi(token=args.token)
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not authenticated. Run `huggingface-cli login`, or pass --token / set HF_TOKEN.\n"
                 "Create a token with WRITE access at https://huggingface.co/settings/tokens")
    print(f"authenticated as {who}")

    api.create_repo(args.repo, repo_type="space", space_sdk="gradio",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=args.repo, repo_type="space",
                      commit_message="deploy json-camera")
    print(f"\nlive shortly at https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
