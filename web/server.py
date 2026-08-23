"""Web app for json-camera: a landing page, a compressor and a decompressor.

    .venv/bin/python web/server.py

Then open http://localhost:8000.

Results are held in a temp directory keyed by a random id rather than returned
inline.  A 2 MP photograph makes a .json of a few hundred kilobytes and two
preview images on top of that, and pushing all of it through one JSON response
would stall the page on the parse.  The browser gets a small summary and then
pulls the heavy parts as ordinary image and file requests, which also means the
download carries a real Content-Disposition filename instead of a blob URL.
"""

import io
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from jsoncam import codec, lossless
from jsoncam.metrics import from_images as ms_ssim, ms_ssim_db

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = Path(os.environ.get("JSONCAM_MODELS", ROOT / "checkpoints" / "stable"))
MAX_SIDE = int(os.environ.get("JSONCAM_MAX_SIDE", "3840"))
MAX_UPLOAD = int(os.environ.get("JSONCAM_MAX_UPLOAD", str(40 * 1024 * 1024)))
# Lossless peaks at roughly 150 MB of working memory per megapixel, measured, so
# a 13 megapixel photograph needs about 1.9 GB.  The machine has 2 GB, and going
# over does not fail politely: the kernel kills the process and every other
# request in flight dies with it.  Refuse clearly instead.
MAX_LOSSLESS_MP = float(os.environ.get("JSONCAM_MAX_LOSSLESS_MP", "10"))
STORE = Path(tempfile.mkdtemp(prefix="jsoncam-web-"))
STORE_TTL = 3600

# Use the whole machine. Requests are already serialised by the concurrency
# limit in front of this, so holding cores back only makes each one slower.
torch.set_num_threads(max(1, os.cpu_count() or 1))

app = FastAPI(title="json-camera", docs_url=None, redoc_url=None)
_models = {}


# --------------------------------------------------------------------------
# models


def discover_models():
    found = []
    for path in sorted(MODEL_DIR.glob("*.pt")):
        if path.name.startswith("_"):
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if "model" not in ck or "config" not in ck:
            continue
        metrics = ck.get("metrics") or {}
        # Report the held-out bpp when the checkpoint carries one.  The training
        # figure is measured against the noise proxy and reads high, so quoting
        # it on a quality picker would misdescribe every model in the list.
        bpp = metrics.get("val_bpp") or metrics.get("bpp")
        psnr = metrics.get("val_psnr") or metrics.get("psnr")
        found.append({
            "id": path.stem,
            "path": str(path),
            "lmbda": ck.get("lmbda"),
            "bpp": bpp,
            "psnr": psnr,
            "label": path.stem,
        })
    found.sort(key=lambda m: (m["bpp"] or 0))

    # Describe each model by what it actually does, not by its filename. With
    # more than one on the curve the honest framing is the trade between them:
    # the cheapest genuinely beats JPEG at a matched size, the dearest looks
    # better but no longer does. Saying only the flattering half would be a lie
    # of omission on whichever model was left out.
    if len(found) > 1:
        found[0]["note"] = "smallest files, beats JPEG at matched size"
        found[-1]["note"] = "sharpest picture, larger files"
    for m in found:
        bits = []
        if m["bpp"]:
            bits.append(f"{m['bpp']:.2f} bpp")
        if m["psnr"]:
            bits.append(f"{m['psnr']:.1f} dB")
        if m.get("note"):
            bits.append(m["note"])
        m["label"] = " · ".join(bits) or m["id"]
    return found


def load_model(model_id):
    for meta in discover_models():
        if meta["id"] == model_id:
            if meta["path"] not in _models:
                _models[meta["path"]] = codec.load_checkpoint(meta["path"])[0]
            return _models[meta["path"]]
    raise HTTPException(404, f"no such model: {model_id}")


def default_model_id():
    models = discover_models()
    if not models:
        raise HTTPException(503, "no trained checkpoint is available")
    return models[0]["id"]


# --------------------------------------------------------------------------
# result store


def sweep():
    """Drop anything older than the TTL. Called on write, so it needs no timer."""
    cutoff = time.time() - STORE_TTL
    for entry in STORE.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def new_slot():
    sweep()
    slot = STORE / secrets.token_urlsafe(12)
    slot.mkdir(parents=True, exist_ok=True)
    return slot


def slot_path(job, name):
    # `job` arrives from the URL, so it must never be able to walk out of STORE.
    if not job.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "bad id")
    path = (STORE / job / name).resolve()
    if not str(path).startswith(str(STORE.resolve())) or not path.exists():
        raise HTTPException(404, "expired or unknown result")
    return path


# --------------------------------------------------------------------------
# helpers


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def psnr(a, b):
    import math

    import numpy as np

    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return None if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def jpeg_at_size(img, target):
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


def safe_stem(name):
    """The upload's basename, stripped of anything that could steer a path."""
    stem = Path(name or "image").name
    stem = os.path.splitext(stem)[0].strip() or "image"
    return "".join(c for c in stem if c not in '/\\:*?"<>|').strip() or "image"


# --------------------------------------------------------------------------
# api


@app.get("/api/models")
def api_models():
    models = discover_models()
    return {"models": models, "default": models[0]["id"] if models else None,
            "max_side": MAX_SIDE}


@app.post("/api/compress")
async def api_compress(
    file: UploadFile = File(...),
    model_id: str = Form(None),
    encoding: str = Form("b85"),
    compare: str = Form("true"),
    mode: str = Form("lossy"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "that file was empty")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file is larger than {human(MAX_UPLOAD)}")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "that does not look like an image we can read")

    # The size cap exists because the neural codec is seconds per megapixel. It
    # must never apply to lossless, where resizing would discard the very thing
    # the mode promises to keep. Prediction and entropy coding are fast enough
    # to take the image at full resolution.
    note = None
    if str(mode).lower() != "lossless" and img.mode in ("RGBA", "LA", "PA"):
        note = ("This image has transparency, and the learned codec has three input "
                "channels, so the alpha channel will be dropped. Lossless mode keeps it.")
    if str(mode).lower() != "lossless" and max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / max(img.size)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)
        resized = f"Resized to {size[0]} by {size[1]}. This demo caps the long side at {MAX_SIDE} pixels."
        note = f"{note} {resized}" if note else resized

    stem = safe_stem(file.filename)
    if str(mode).lower() == "lossless":
        mp = img.width * img.height / 1e6
        if mp > MAX_LOSSLESS_MP:
            raise HTTPException(413, (
                f"That image is {mp:.1f} megapixels and lossless mode is capped at "
                f"{MAX_LOSSLESS_MP:.0f} here. Lossless cannot resize it for you, because "
                f"resizing is exactly the thing this mode promises not to do. Use lossy "
                f"for an image this size, or run it locally with "
                f"`jsoncam encode photo.png --lossless`, which has no cap."))
        return _compress_lossless(file, img, raw, stem, note)

    model_id = model_id or default_model_id()
    model = load_model(model_id)

    t0 = time.time()
    doc = codec.encode_image(model, img, encoding=encoding, device="cpu",
                             name=Path(file.filename or "image").name)
    encode_seconds = time.time() - t0

    slot = new_slot()
    json_name = f"{stem}.json"
    json_path = slot / "payload.json"
    codec.write_json(doc, json_path)

    t0 = time.time()
    rec = codec.decode_dict(model, doc, device="cpu")
    decode_seconds = time.time() - t0

    # Write the profile back out, or the browser paints these numbers as sRGB
    # and the reconstruction looks colour-shifted against its own original.
    icc = rec.info.get("icc_profile")
    img.save(slot / "original.png", icc_profile=icc)
    rec.save(slot / "decoded.png", icc_profile=icc)

    stats = codec.stats(doc, json_path)
    quality = psnr(img, rec)
    structural = ms_ssim(img, rec)

    payload = {
        "id": slot.name,
        "note": note,
        "original_name": Path(file.filename or "image").name,
        "json_name": json_name,
        "width": img.width,
        "height": img.height,
        "source_bytes": len(raw),
        "raw_bytes": stats["raw_bytes"],
        "bitstream_bytes": stats["bitstream_bytes"],
        "json_bytes": stats["json_bytes"],
        "bpp": stats["bpp"],
        "text_overhead_pct": stats["text_overhead_pct"],
        "ratio_vs_raw": stats["ratio_vs_raw_json"],
        "psnr": quality,
        "ms_ssim": structural,
        "ms_ssim_db": ms_ssim_db(structural),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "latent": doc["latent"],
        "symbols": doc["codec"]["count"],
        "clipped": doc["codec"]["clipped_symbols"],
        "model": {"id": model_id, "fingerprint": doc["model"]["fingerprint"]},
        "header": {k: v for k, v in doc.items() if k != "payload"},
        "payload_preview": doc["payload"]["data"][:220],
        "payload_chars": len(doc["payload"]["data"]),
        "jpeg": None,
    }

    if str(compare).lower() in ("1", "true", "yes", "on"):
        jq, jn, jbytes = jpeg_at_size(img, stats["json_bytes"])
        jrec = Image.open(io.BytesIO(jbytes)).convert("RGB")
        jrec.save(slot / "jpeg.png", icc_profile=icc)
        (slot / "jpeg.jpg").write_bytes(jbytes)
        jm = ms_ssim(img, jrec)
        payload["jpeg"] = {
            "quality": jq, "bytes": jn, "psnr": psnr(img, jrec),
            "ms_ssim": jm, "ms_ssim_db": ms_ssim_db(jm),
        }

    (slot / "meta.json").write_text(json.dumps({"json_name": json_name}))
    return payload


def _compress_lossless(file, img, raw, stem, note):
    """Nothing discarded. No model involved, and no quality number to report:
    the reconstruction is the original, so PSNR is infinite by construction."""
    t0 = time.time()
    doc = lossless.encode_image(img, name=Path(file.filename or "image").name)
    encode_seconds = time.time() - t0

    slot = new_slot()
    json_path = slot / "payload.json"
    codec.write_json(doc, json_path)

    t0 = time.time()
    rec = lossless.decode_dict(doc)
    decode_seconds = time.time() - t0

    # Verify rather than assert. A lossless codec that is quietly lossy is worse
    # than no lossless codec, so the claim is checked on every single request.
    import numpy as np

    exact = np.array_equal(np.asarray(img), np.asarray(rec))

    icc = rec.info.get("icc_profile")
    img.save(slot / "original.png", icc_profile=icc)
    rec.save(slot / "decoded.png", icc_profile=icc)

    st = lossless.stats(doc, json_path)
    png_buf, webp_buf = io.BytesIO(), io.BytesIO()
    img.save(png_buf, "PNG", optimize=True)
    img.save(webp_buf, "WEBP", lossless=True, quality=100)

    (slot / "meta.json").write_text(json.dumps({"json_name": f"{stem}.json"}))
    return {
        "id": slot.name, "note": note, "lossless": True, "bit_exact": exact,
        "original_name": Path(file.filename or "image").name,
        "json_name": f"{stem}.json",
        "width": img.width, "height": img.height,
        "source_bytes": len(raw), "raw_bytes": st["raw_bytes"],
        "bitstream_bytes": st["bitstream_bytes"], "json_bytes": st["json_bytes"],
        "bpp": st["bpp"], "ratio_vs_raw": st["ratio_vs_raw_json"],
        "png_bytes": png_buf.tell(), "webp_bytes": webp_buf.tell(),
        "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
        "header": {k: v for k, v in doc.items() if k not in ("payload", "codec")},
        "payload_preview": doc["payload"]["data"][:220],
        "payload_chars": len(doc["payload"]["data"]),
        "jpeg": None, "psnr": None, "ms_ssim": None,
    }


@app.post("/api/decompress")
async def api_decompress(file: UploadFile = File(...), model_id: str = Form(None)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "that file was empty")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file is larger than {human(MAX_UPLOAD)}")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(415, "that is not a JSON file we can read")
    if not isinstance(doc, dict):
        raise HTTPException(415, "that JSON is not a json-camera file")
    if doc.get("format") == lossless.FORMAT:
        return _decompress_lossless(doc, raw, file)
    if doc.get("format") != "json-camera/1":
        raise HTTPException(415, "that JSON is not a json-camera file")

    wanted = doc.get("model", {}).get("fingerprint")
    chosen = model_id
    if not chosen:
        # Pick the checkpoint the file was actually made with, when we hold it.
        for meta in discover_models():
            if codec.model_fingerprint(load_model(meta["id"])) == wanted:
                chosen = meta["id"]
                break
    chosen = chosen or default_model_id()
    model = load_model(chosen)

    t0 = time.time()
    try:
        img = codec.decode_dict(model, doc, device="cpu")
    except ValueError as error:
        raise HTTPException(409, str(error))
    decode_seconds = time.time() - t0

    slot = new_slot()
    img.save(slot / "decoded.png", icc_profile=img.info.get("icc_profile"))
    # The name the picture went in with, not the name of the .json it arrived as.
    stored = (doc.get("image") or {}).get("name")
    stem = safe_stem(stored or file.filename)
    png_name = f"{stem}.png"
    (slot / "meta.json").write_text(json.dumps({"png_name": png_name}))

    return {
        "id": slot.name,
        "original_name": stored,
        "png_name": png_name,
        "width": img.width,
        "height": img.height,
        "created": doc.get("created"),
        "bitstream_bytes": doc["codec"]["bitstream_bytes"],
        "json_bytes": len(raw),
        "fingerprint": wanted,
        "model": chosen,
        "decode_seconds": decode_seconds,
        "latent": doc.get("latent"),
        "header": {k: v for k, v in doc.items() if k != "payload"},
    }


def _decompress_lossless(doc, raw, file):
    t0 = time.time()
    try:
        img = lossless.decode_dict(doc)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, f"that file is damaged: {error}")
    decode_seconds = time.time() - t0

    slot = new_slot()
    img.save(slot / "decoded.png", icc_profile=img.info.get("icc_profile"))
    stored = (doc.get("image") or {}).get("name")
    png_name = f"{safe_stem(stored or file.filename)}.png"
    (slot / "meta.json").write_text(json.dumps({"png_name": png_name}))

    return {
        "id": slot.name, "lossless": True,
        "original_name": stored, "png_name": png_name,
        "width": img.width, "height": img.height,
        "created": doc.get("created"),
        "bitstream_bytes": doc["codec"]["bitstream_bytes"],
        "json_bytes": len(raw),
        "fingerprint": None, "model": "none needed, this format carries no weights",
        "decode_seconds": decode_seconds, "latent": None,
        "header": {k: v for k, v in doc.items() if k not in ("payload", "codec")},
    }


@app.get("/api/preview/{job}/{which}")
def api_preview(job: str, which: str):
    if which not in ("original", "decoded", "jpeg"):
        raise HTTPException(404, "no such preview")
    return FileResponse(slot_path(job, f"{which}.png"), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=600"})


@app.get("/api/download/{job}")
def api_download(job: str):
    meta = json.loads(slot_path(job, "meta.json").read_text())
    if "json_name" in meta:
        return FileResponse(slot_path(job, "payload.json"), media_type="application/json",
                            filename=meta["json_name"])
    return FileResponse(slot_path(job, "decoded.png"), media_type="image/png",
                        filename=meta["png_name"])


# --------------------------------------------------------------------------
# chat
#
# A private line between a phone and whoever is working on this repo at a
# terminal.  Messages are appended to one file on a mounted volume, because the
# machine stops when idle and its rootfs is replaced on every deploy: memory and
# the rootfs would both lose the conversation.
#
# One shared key guards it, compared in constant time and required on every call.
# Without a key set the endpoints refuse rather than defaulting to open, since a
# chat that silently accepts strangers is worse than one that does not work.

CHAT_DIR = Path(os.environ.get("JSONCAM_CHAT_DIR", tempfile.gettempdir()))
CHAT_FILE = CHAT_DIR / "chat.jsonl"
CHAT_KEY = os.environ.get("JSONCAM_CHAT_KEY", "")


def check_key(key):
    import hmac

    if not CHAT_KEY:
        raise HTTPException(503, "chat is not configured on this server")
    if not key or not hmac.compare_digest(key, CHAT_KEY):
        raise HTTPException(401, "wrong key")


def chat_read(after=0):
    if not CHAT_FILE.exists():
        return []
    out = []
    with CHAT_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except ValueError:
                continue                       # a torn final line, skip it
            if m.get("id", 0) > after:
                out.append(m)
    return out


def chat_append(who, text):
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    existing = chat_read()
    msg = {"id": (existing[-1]["id"] + 1) if existing else 1,
           "who": who, "text": text, "at": time.time()}
    with CHAT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(msg) + "\n")
    return msg


@app.get("/api/chat")
def api_chat_get(after: int = 0, key: str = "", wait: int = 0):
    """Messages after `after`. With `wait`, holds the request open until
    something arrives, so the phone does not have to poll in a tight loop."""
    check_key(key)
    deadline = time.time() + min(wait, 50)
    while True:
        msgs = chat_read(after)
        if msgs or time.time() >= deadline:
            return {"messages": msgs, "now": time.time()}
        time.sleep(1.0)


@app.post("/api/chat")
async def api_chat_post(text: str = Form(...), who: str = Form("mubaraq"), key: str = Form("")):
    check_key(key)
    text = text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    if len(text) > 8000:
        raise HTTPException(413, "message too long")
    return chat_append("claude" if who == "claude" else "mubaraq", text)


# --------------------------------------------------------------------------
# pages


def page(name):
    return HTMLResponse((HERE / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def index():
    return page("index.html")


@app.get("/llms.txt", response_class=Response)
def llms_txt():
    """The convention for making a site legible to a language model. Served from
    the root because that is where tools look for it."""
    return Response((HERE / "static" / "llms.txt").read_text(encoding="utf-8"),
                    media_type="text/plain; charset=utf-8")


@app.get("/AGENTS.md", response_class=Response)
def agents_md():
    return Response((ROOT / "AGENTS.md").read_text(encoding="utf-8"),
                    media_type="text/markdown; charset=utf-8")


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return page("chat.html")


@app.get("/developers", response_class=HTMLResponse)
def developers_page():
    return page("developers.html")


@app.get("/compress", response_class=HTMLResponse)
def compress_page():
    return page("compress.html")


@app.get("/decompress", response_class=HTMLResponse)
def decompress_page():
    return page("decompress.html")


@app.exception_handler(404)
def not_found(request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(exc, "detail", "not found")}, status_code=404)
    return HTMLResponse((HERE / "404.html").read_text(encoding="utf-8"), status_code=404)


app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"json-camera  ->  http://localhost:{port}")
    print(f"  models {MODEL_DIR}")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")
