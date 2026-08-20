"""Gradio interface for json-camera.

Runs locally (`python app.py`) and unchanged on a Hugging Face Space.

Any checkpoint dropped into MODEL_DIR shows up in the quality dropdown
automatically, so adding a new lambda later needs no code change.
"""

import glob
import io
import json
import math
import os
import tempfile
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from jsoncam import codec

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

MODEL_DIR = os.environ.get("JSONCAM_MODELS", "checkpoints")
MAX_SIDE = int(os.environ.get("JSONCAM_MAX_SIDE", "3840"))
_CACHE = {}


# --------------------------------------------------------------------------


def discover_models():
    """Find usable checkpoints, best (highest lambda) first."""
    found = []
    for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt"))):
        if os.path.basename(p).startswith("_"):
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if "model" not in ck or "config" not in ck:
                continue
        except Exception:
            continue
        lm = ck.get("lmbda")
        m = ck.get("metrics") or {}
        bits = f"~{m['bpp']:.2f} bpp" if m.get("bpp") else "?"
        q = f"lambda={lm:g}" if lm else "unknown lambda"
        found.append((f"{os.path.basename(p)}  ({q}, {bits})", p, lm or 0.0))
    found.sort(key=lambda t: -t[2])
    return [(label, path) for label, path, _ in found]


def load_model(path):
    if path not in _CACHE:
        _CACHE[path] = codec.load_checkpoint(path)[0]
    return _CACHE[path]


def _psnr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{int(n)} B" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0


def _jpeg_at_size(img, target):
    """Best JPEG that fits in `target` bytes -- the honest baseline."""
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        n = buf.tell()
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (q, n, buf.getvalue())
        if n < target:
            lo = q + 1
        else:
            hi = q - 1
    return best


# --------------------------------------------------------------------------


def do_encode(image, model_path, encoding, compare_jpeg, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    if not model_path:
        raise gr.Error("No trained checkpoint available yet.")

    img = image.convert("RGB")
    note = ""
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        new = (max(1, int(img.width * s)), max(1, int(img.height * s)))
        img = img.resize(new, Image.LANCZOS)
        note = f"\n> Resized to {new[0]}x{new[1]} (this demo caps the long side at {MAX_SIDE}px).\n"

    model = load_model(model_path)

    progress(0.15, desc="Encoding…")
    t0 = time.time()
    doc = codec.encode_image(model, img, encoding=encoding, device="cpu")
    t_enc = time.time() - t0

    tmp = os.path.join(tempfile.mkdtemp(), "image.json")
    codec.write_json(doc, tmp)
    s = codec.stats(doc, tmp)

    progress(0.6, desc="Decoding back…")
    t0 = time.time()
    rec = codec.decode_dict(model, doc, device="cpu")
    t_dec = time.time() - t0

    q = _psnr(img, rec)
    rows = [
        ("Original (raw RGB)", _human(s["raw_bytes"]), ""),
        ("Latent grid", f"{doc['latent']['channels']}x{doc['latent']['height']}x{doc['latent']['width']}",
         f"{doc['codec']['count']:,} numbers"),
        ("**Bitstream**", f"**{_human(s['bitstream_bytes'])}**", f"{s['bpp']:.4f} bpp"),
        ("**JSON file**", f"**{_human(s['json_bytes'])}**",
         f"+{s['text_overhead_pct']:.0f}% text armour"),
        ("Compression vs raw", f"{s['ratio_vs_raw_json']:.0f}x", ""),
        ("Quality (PSNR)", f"{q:.2f} dB", ""),
        ("Time", f"{t_enc:.1f}s encode / {t_dec:.1f}s decode", ""),
    ]
    md = f"### Results{note}\n\n| | | |\n|---|---|---|\n"
    md += "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)

    if compare_jpeg:
        progress(0.85, desc="Comparing with JPEG…")
        jq, jn, jb = _jpeg_at_size(img, s["json_bytes"])
        jp = _psnr(img, Image.open(io.BytesIO(jb)).convert("RGB"))
        d = q - jp
        word = "**better**" if d > 0 else "worse"
        md += (f"\n\n### Against JPEG at the same file size\n\n"
               f"| codec | size | quality |\n|---|---|---|\n"
               f"| json-camera | {_human(s['json_bytes'])} | {q:.2f} dB |\n"
               f"| JPEG q={jq} | {_human(jn)} | {jp:.2f} dB |\n\n"
               f"json-camera is **{abs(d):.2f} dB {word}** here.")

    header = {k: v for k, v in doc.items() if k != "payload"}
    header["payload"] = {
        "encoding": doc["payload"]["encoding"],
        "data": doc["payload"]["data"][:96] + f"… ({len(doc['payload']['data']):,} chars total)",
    }
    return rec, md, tmp, json.dumps(header, indent=2)


def do_decode(file_obj, model_path):
    if file_obj is None:
        raise gr.Error("Upload a .json file first.")
    if not model_path:
        raise gr.Error("No trained checkpoint available yet.")
    try:
        doc = codec.read_json(file_obj.name if hasattr(file_obj, "name") else file_obj)
    except Exception as e:
        raise gr.Error(f"Could not read that as JSON: {e}")

    model = load_model(model_path)
    try:
        img = codec.decode_dict(model, doc, device="cpu")
    except ValueError as e:
        # Almost always the fingerprint guard -- worth explaining, not just failing.
        raise gr.Error(str(e))

    info = (f"**{doc['image']['width']}x{doc['image']['height']}**  ·  "
            f"encoded {doc.get('created', '?')}  ·  "
            f"model `{doc['model'].get('fingerprint', '?')}`  ·  "
            f"{_human(doc['codec']['bitstream_bytes'])} of bitstream")
    return img, info


# --------------------------------------------------------------------------

EXPLAINER = """
## What is actually happening

**1 — Images are numbers.** A 4K photo is 3840x2160x3, about 25 MB raw. Neighbouring
pixels are nearly always similar, so most of those bytes are redundant.

**2 — Strided convolutions squeeze it.** A convolution slides a small window over the
image; `stride=2` makes it hop two pixels at a time, halving each dimension. Four in a
row take 3840x2160 down to a 240x135 grid. Unlike max-pooling or mean-pooling, the
window's weights are **learned** — training finds a rule better than "take the mean",
because it is optimised for one thing only: being reconstructable later.

**3 — Quantisation.** The grid is rounded to whole numbers. This is the lossy step and
where most of the saving comes from. Rounding is not differentiable, so during training
we add uniform noise instead — a smooth stand-in that lets gradients through.

**4 — The entropy model counts the bits.** A second network learns, per channel, how
values in that channel are distributed. Knowing a value's probability tells you its exact
cost: `-log2(p)` bits. Common values cost a fraction of a bit.

This is the piece that makes the whole thing work. Without it you are just saving an
array. With it, the network gets a **differentiable estimate of the output file size**
and can be trained to minimise it directly:

```
loss = lambda * (how wrong the picture is) + (how many bits it cost)
```

Those two terms fight each other, and `lambda` decides who wins. It is the quality knob,
and the only difference between a 200 KB model and a 2 MB one.

**5 — rANS packs the bits.** A range coder writes the symbol stream at essentially the
theoretical limit (measured: within 1.6% of Shannon entropy).

**6 — JSON.** The bitstream is base85-armoured into a text payload with a readable header.

---

## Three honest caveats

**This is compression, not encryption.** It looks unreadable, but there is no key and no
secret. Anyone with the checkpoint can read it. Do not use it to hide anything.

**The model weights are part of the file format.** The decoder cannot rebuild anything
without the exact weights that encoded it — which is why every file carries a
`fingerprint` and decoding refuses on a mismatch. The checkpoint is a shared codebook,
like the Huffman tables baked into every JPEG decoder.

**JSON costs 25%.** Text holds only ~6.1 bits per character in base85 (base64 is worse at
33%). That overhead is the price of the container being JSON, not a flaw in the codec —
both numbers are always shown above so you can see the tax.
"""


def build_ui():
    models = discover_models()
    choices = [(lbl, p) for lbl, p in models]
    default = choices[0][1] if choices else None

    with gr.Blocks(title="json-camera") as demo:
        gr.Markdown("# json-camera\n"
                    "### A neural network that compresses photographs into JSON text.")
        if not choices:
            gr.Markdown("> **No trained checkpoint found.** Train one with "
                        "`jsoncam train`, then export it into `checkpoints/`.")

        model_dd = gr.Dropdown(choices=choices, value=default, label="Model / quality level",
                               info="Higher lambda = better image, bigger file.")

        with gr.Tab("Encode"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Input image", height=340)
                    with gr.Row():
                        enc_dd = gr.Radio(["b85", "b64"], value="b85", label="Text encoding",
                                          info="b85 costs +25%, b64 costs +33%")
                    cmp_cb = gr.Checkbox(value=True, label="Compare against JPEG at the same size")
                    go = gr.Button("Encode to JSON", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Reconstructed from the JSON", height=340)
                    out_file = gr.File(label="Download .json")
            out_md = gr.Markdown()
            with gr.Accordion("Peek inside the JSON", open=False):
                out_json = gr.Code(language="json", label="Header (payload truncated)")
            go.click(do_encode, [inp, model_dd, enc_dd, cmp_cb],
                     [out_img, out_md, out_file, out_json])

        with gr.Tab("Decode"):
            gr.Markdown("Upload a `.json` produced by this app. It must be decoded with "
                        "the **same model** that encoded it.")
            with gr.Row():
                dec_in = gr.File(label=".json file", file_types=[".json"])
                dec_out = gr.Image(label="Decoded image", height=340)
            dec_info = gr.Markdown()
            gr.Button("Decode", variant="primary").click(
                do_decode, [dec_in, model_dd], [dec_out, dec_info])

        with gr.Tab("How it works"):
            gr.Markdown(EXPLAINER)

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft(),
                      server_name=os.environ.get("HOST", "127.0.0.1"),
                      server_port=int(os.environ.get("PORT", "7860")))
