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
import logging
import os
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from jsoncam import formats
from jsoncam import meta as jsoncam_meta


class _Lazy:
    """A module that imports itself on first use.

    torch takes about half a minute to import on this machine, and until it
    finished, uvicorn had not bound the port. Fly's proxy wakes a sleeping
    machine, waits roughly eight seconds for it to answer, and gives up: so the
    first upload after an idle spell was refused before the app existed, with no
    error anybody could see. A Shortcut sending two photographs lost both.

    Deferring the import lets the port open in about a second. The connection
    then succeeds and the first request merely waits, which is a completely
    different outcome from being refused. `warm()` below loads it in the
    background so usually nothing waits at all.

    A proxy rather than rewriting thirty call sites: `codec.encode_image(...)`
    reads exactly as it did.
    """

    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, attribute):
        if self._module is None:
            import importlib

            self._module = importlib.import_module(self._name)
        return getattr(self._module, attribute)


codec = _Lazy("jsoncam.codec")
lossless = _Lazy("jsoncam.lossless")
torch = _Lazy("torch")


def ms_ssim(*args, **kwargs):
    from jsoncam.metrics import from_images

    return from_images(*args, **kwargs)


def ms_ssim_db(*args, **kwargs):
    from jsoncam.metrics import ms_ssim_db as real

    return real(*args, **kwargs)

import faces
import library
import vision

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

def warm():
    """Load torch now, in the background, so the first request does not wait.

    Called at startup. The port is already open by then, which is the point:
    the machine is reachable while this is still running.
    """
    try:
        # Use the whole machine. Requests are already serialised by the
        # concurrency limit in front of this, so holding cores back only makes
        # each one slower.
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        # Touch the codec itself, not just torch: that is what the request
        # handlers reach for, and warming only torch left the first upload
        # paying for the rest of the import anyway.
        codec.TILE
        lossless.FORMAT
        discover_models()
        log.info("codec ready")
    except Exception:
        log.exception("warm-up failed; the first request will do this itself")

# Register before any request arrives: a Shortcut uploading straight from the
# camera roll sends HEIC, and without this every one of them is a 415.
HEIF_OK = formats.enable_heif()

# Without this, vision.log's warnings go nowhere and a library with no captions
# gives no clue why. The codec itself stays quiet.
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("jsoncam.web")

app = FastAPI(title="json-camera", docs_url=None, redoc_url=None)
_models = {}

# The gallery is served from mubby.space and the codec runs here, so every
# library call is cross-origin. Named origins rather than "*", because these
# requests carry a key: a wildcard would let any page a visitor happens to have
# open read their library out of the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(
        "JSONCAM_ALLOW_ORIGINS",
        "https://mubby.space,https://www.mubby.space,http://localhost:3000").split(",") if o],
    # Vercel gives every deployment its own hostname, so previews of the gallery
    # would otherwise be blocked while production worked.
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["X-Library-Key", "X-Admin-Key", "Content-Type"],
    max_age=86400,
)


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
    """Which checkpoint new uploads are encoded with.

    Honours the admin override when it names a checkpoint that is actually
    present, and falls back to the first otherwise: a config value left behind
    by a checkpoint that has since been removed must not take encoding down.
    """
    models = discover_models()
    if not models:
        raise HTTPException(503, "no trained checkpoint is available")
    wanted = library.get_config("codec_model")
    if wanted and any(m["id"] == wanted for m in models):
        return wanted
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
        # Bake the orientation tag in here, before anything measures this image.
        # Both encoders call exif_transpose internally, so a phone photo held
        # sideways would otherwise be encoded rotated while this function kept
        # the unrotated original: psnr() then compares a 1600x900 array against
        # a 900x1600 one and the request dies with a broadcast error, and the
        # lossless path quietly reports bit_exact false on a bit-exact codec.
        # Doing it first also means MAX_SIDE caps the edge the viewer sees.
        img = ImageOps.exif_transpose(img)
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
# the captioning worker
#
# Runs once per photograph, in the background, and never on a page load. The row
# is written pending at upload; this thread fills it in. That means a restart
# resumes where it stopped rather than redoing work, the gallery is always just
# a query, and a slow or missing vision API never holds up an upload.
#
# One at a time on purpose. The machine has a single core that spends most of it
# encoding, and the queue draining slowly is invisible to a visitor while a
# thread pool competing with the codec is not.

WORKER_IDLE = float(os.environ.get("JSONCAM_WORKER_IDLE", "5"))
_worker_started = False


def backfill_one_place():
    """Re-resolve one photograph's place label. True if it did any work.

    Needs no API key, so this runs even on a machine with captioning turned
    off entirely: it is only OpenStreetMap.
    """
    waiting = library.next_place_backfill(limit=1)
    if not waiting:
        return False
    row = waiting[0]
    library.set_place(row["id"], library.place_for(row["lat"], row["lon"]))
    return True


def adjudicate_one():
    """Ask about one borderline pair of people. True if it did any work.

    This is the only place a vision model is allowed near identity, and only
    for pairs the face model already declined to call. A clear "same" merges
    them; "different" and "unsure" both leave them apart, and either way the
    answer is stored so the pair is never paid for twice.
    """
    waiting = library.next_verdict(limit=1)
    if not waiting:
        return False
    pair = waiting[0]
    left = library.face_crop(pair["library"], pair["left_cover"] or "")
    right = library.face_crop(pair["library"], pair["right_cover"] or "")
    if not left or not right:
        library.save_verdict(pair["pair"], "unsure", "no crop")
        return True

    answer = vision.compare_faces(left, right)
    if not answer:
        # Could not ask. Leave it unjudged so it is retried when the provider
        # is working, rather than recording a verdict nobody gave.
        return False

    library.save_verdict(pair["pair"], answer["verdict"], answer["model"])
    if answer["verdict"] == "same":
        library.merge_people(pair["library"], pair["left"], pair["right"])
        log.info("merged two piles at similarity %.3f: %s",
                 pair["similarity"] or 0, answer["why"])
    return True


def analyse_one():
    """Describe the next waiting photograph. True if it did any work."""
    pending = library.next_pending(limit=1)
    if not pending:
        return False
    row = pending[0]
    # Count the attempt before making it, so a request that crashes the worker
    # cannot be retried forever.
    library.note_attempt(row["item"])
    described = vision.describe(row["preview"], f"image/{row['preview_type'] or 'webp'}")
    if not described:
        return True                      # attempted; it will retry or give up
    library.save_analysis(row["library"], row["item"], described,
                          vision.searchable(described, row))
    return True


def worker_loop():
    while True:
        did_work = False
        for job in (backfill_one_place, adjudicate_one, analyse_one):
            try:
                did_work = job() or did_work
            except Exception:
                # A worker that dies takes the whole feature with it, silently,
                # and one failing job must not stop the other.
                log.exception("background job %s failed", job.__name__)
        time.sleep(0 if did_work else WORKER_IDLE)


def start_worker():
    """Start the background worker once.

    Started regardless of whether captioning is configured, because place labels
    are the other job here and those need only OpenStreetMap.
    """
    global _worker_started
    if _worker_started:
        return
    import threading

    _worker_started = True
    threading.Thread(target=worker_loop, daemon=True, name="jsoncam-worker").start()


@app.on_event("startup")
def _on_startup():
    import threading

    # Both in the background. The port is bound before either runs, which is
    # the whole reason a cold machine now accepts the request that woke it.
    threading.Thread(target=warm, daemon=True, name="jsoncam-warm").start()
    start_worker()


@app.get("/api/health")
def api_health():
    """Answers without touching the codec, so it is true the moment the port opens."""
    # `warm` is what flips these; until then the port is open and a request
    # would simply wait for the import rather than be refused.
    return {"ok": True,
            "codec_loaded": codec._module is not None,
            "torch_loaded": torch._module is not None,
            "heic": HEIF_OK, "faces": faces.available()}


# --------------------------------------------------------------------------
# library
#
# The endpoints a phone talks to. A Shortcut POSTs photographs here one at a
# time and a gallery reads them back; see web/library.py for why the store is
# shaped the way it is.
#
# The key arrives either as an `X-Library-Key` header or a `key` field, because
# the two clients cannot both use the same one: fetch() from the gallery sets a
# header, and a Shortcut sending a multipart form finds a form field far easier.
# Neither is ever logged, and only the hash is stored.

LIBRARY_MAX_SIDE = int(os.environ.get("JSONCAM_LIBRARY_MAX_SIDE", str(MAX_SIDE)))


def require_key(header_key, form_key=None, query_key=None):
    key = (header_key or form_key or query_key or "").strip()
    verdict = library.inspect_key(key)
    if verdict == "short":
        raise HTTPException(401, "this needs a library key")
    # A checkable key that fails its own checksum is a typo, and saying so is
    # the whole point of the checksum: the alternative is silently opening a
    # different, empty library and letting somebody conclude their photographs
    # are gone. This leaks nothing, because it says only that the key is
    # malformed, never whether any library exists or holds anything.
    if verdict == "typo":
        raise HTTPException(400, (
            "That key has a typo in it. One of the characters is wrong, so it does "
            "not match its own checksum. Nothing has been lost: check it against "
            "the copy you saved."))
    return library.library_id(key)


@app.post("/api/library/new")
def api_library_new():
    """Mint a key. Generated here so a browser cannot pick a weak one."""
    key = library.new_key()
    return {"key": key, "library": library.library_id(key)}


@app.get("/api/library/check")
def api_library_check(key: str = None, x_library_key: str = Header(None)):
    """Is this key well formed?

    Answers only that, and never whether a library exists or has photographs in
    it, which would make this an oracle for guessing keys. It exists so somebody
    typing a key off a piece of paper is told about a typo while they are still
    looking at the field, rather than after they have concluded their library is
    empty.
    """
    return {"verdict": library.inspect_key(x_library_key or key or "")}


# How many photographs one request may carry. Encoding is seconds each on a
# single core, so a request holding fifty would sit silent long enough for
# something in the path to give up. Past this the Shortcut's loop is the right
# shape, and the error says so.
BATCH_MAX = int(os.environ.get("JSONCAM_BATCH_MAX", "8"))


@app.post("/api/library/upload")
async def api_library_upload(
    # A list, not one file. This used to be a bare `UploadFile`, and when a
    # Shortcut sent several photographs under the same field name FastAPI kept
    # the first and silently dropped the rest: select twelve, get one, no error.
    # Accepting a list means the endpoint works whether the Shortcut loops per
    # photograph or hands over the whole selection at once.
    file: list[UploadFile] = File(...),
    key: str = Form(None),
    model_id: str = Form(None),
    mode: str = Form("lossy"),
    gps: str = Form("true"),
    x_library_key: str = Header(None),
):
    """Encode the photographs in this request and file them.

    Returns one result per photograph. A single-photograph request still gets
    the flat shape it always did, so an existing Shortcut keeps working
    unchanged.
    """
    lib = require_key(x_library_key, key)
    incoming = [f for f in (file or []) if f is not None]
    if not incoming:
        raise HTTPException(400, "no photo in that request")
    if len(incoming) > BATCH_MAX:
        raise HTTPException(413, (
            f"That request carried {len(incoming)} photos and this endpoint takes "
            f"{BATCH_MAX} at a time, because each one is seconds of encoding. Send them "
            f"in smaller batches, or use a Repeat with Each loop in the Shortcut so each "
            f"photo is its own request, which has no limit."))

    results, failures = [], []
    for one in incoming:
        try:
            results.append(await _store_one(lib, one, model_id, mode, gps))
        except HTTPException as error:
            # One unreadable photograph must not throw away the others in the
            # same request.
            failures.append({"name": (one.filename or "photo"), "error": error.detail})

    if not results and failures:
        raise HTTPException(415, failures[0]["error"])

    if len(incoming) == 1 and results:
        return results[0]                 # the shape older Shortcuts expect
    return {"uploaded": len(results), "failed": len(failures),
            "photos": results, "errors": failures,
            "library_items": library.usage(lib)["items"]}


async def _store_one(lib, file, model_id, mode, gps):
    """Encode and file exactly one photograph."""
    used = library.usage(lib)
    if used["items"] >= library.MAX_ITEMS:
        raise HTTPException(413, f"this library is full at {library.MAX_ITEMS} photos")
    if used["bytes"] >= library.MAX_BYTES:
        raise HTTPException(413, f"this library is full at {human(library.MAX_BYTES)}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "that file was empty")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file is larger than {human(MAX_UPLOAD)}")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "that does not look like an image we can read")

    keep_gps = str(gps).lower() not in ("false", "0", "no", "off")
    lossless_mode = str(mode).lower() == "lossless"
    note = None

    # Read EXIF off the original, before any resize or transpose touches it.
    info = jsoncam_meta.extract(img, gps=keep_gps)
    img = ImageOps.exif_transpose(img)

    if not lossless_mode and max(img.size) > LIBRARY_MAX_SIDE:
        scale = LIBRARY_MAX_SIDE / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
        note = f"resized to {img.width}x{img.height}"

    t0 = time.time()
    if lossless_mode:
        mp = img.width * img.height / 1e6
        if mp > MAX_LOSSLESS_MP:
            raise HTTPException(413, f"lossless is capped at {MAX_LOSSLESS_MP:.0f} megapixels here")
        doc = lossless.encode_image(img.convert("RGB"), name=Path(file.filename or "photo").name,
                                    exif=False, gps=keep_gps)
    else:
        model = load_model(model_id or default_model_id())
        doc = codec.encode_image(model, img.convert("RGB"), device="cpu",
                                 name=Path(file.filename or "photo").name,
                                 exif=False, gps=keep_gps)
    # The image handed to the encoder has already been transposed and resized, so
    # it no longer carries EXIF. Put back the block read off the original.
    if info:
        doc["meta"] = info
    encode_seconds = time.time() - t0

    body = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    item_id = library.add(lib, doc, body, source_bytes=len(raw))
    wants = library.settings(lib)
    # Queued, not run: the Shortcut is waiting on this response and the caption
    # is worth nothing to it.
    if wants["ai"] and vision.available():
        library.enqueue_analysis(lib, item_id)

    # Faces, on the other hand, run now. They need the actual pixels at a usable
    # size, and this is the one moment the full image is already in memory:
    # doing it later would mean decoding the photograph again, which is seconds
    # of neural network to recover something we are holding for free. Measured
    # at 35 ms for a frame with twenty-four faces in it.
    people_found = 0
    if wants["faces"] and faces.available():
        try:
            found = faces.find(img)
            if found:
                library.add_faces(lib, item_id, found)
            people_found = len(found)
        except Exception:
            pass          # a photograph must upload even if this misbehaves

    return {
        "id": item_id,
        "name": doc["image"]["name"],
        "width": doc["image"]["width"],
        "height": doc["image"]["height"],
        "source_bytes": len(raw),
        "json_bytes": len(body),
        "ratio": round(len(raw) / max(len(body), 1), 2),
        "captured_at": (info or {}).get("captured_at"),
        "encode_seconds": round(encode_seconds, 2),
        "note": note,
        "faces": people_found,
        "library_items": used["items"] + 1,
    }


@app.get("/api/library/items")
def api_library_items(key: str = None, limit: int = 500, offset: int = 0,
                      view: str = "all", x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    if view not in library.VIEWS:
        raise HTTPException(400, f"no such view: {view}")
    # Sweep anything past its thirty days before answering, so the trash count
    # the gallery shows is the trash that actually still exists.
    library.purge(lib)
    items = library.listing(lib, limit=min(max(limit, 1), 1000), offset=max(offset, 0),
                            view=view)
    return {"items": _with_captions(lib, items), "usage": library.usage(lib), "view": view,
            "trash_days": library.TRASH_DAYS, "max_side": LIBRARY_MAX_SIDE,
            "settings": library.settings(lib),
            "ai_available": vision.available(),
            "pending": library.queue_depth(lib),
            "recent": library.arrived_recently(lib),
            "checking": library.verdicts_pending()}


def _with_captions(lib, items):
    """Attach what the vision model saw, in one query for the whole page."""
    ids = [i["id"] for i in items]
    described = library.analysis_for(lib, ids)
    who = library.faces_on(lib, ids) if library.settings(lib)["faces"] else {}
    for item in items:
        item["analysis"] = described.get(item["id"])
        item["faces"] = who.get(item["id"], [])
    return items


@app.get("/api/library/search")
def api_library_search(q: str, key: str = None, x_library_key: str = Header(None)):
    """Find photographs by what is in them, not just when they were taken."""
    lib = require_key(x_library_key, None, key)
    items = library.search(lib, q)
    return {"items": _with_captions(lib, items), "query": q,
            "usage": library.usage(lib), "settings": library.settings(lib),
            "pending": library.queue_depth(lib)}


@app.get("/api/library/people")
def api_library_people(key: str = None, x_library_key: str = Header(None)):
    """Everybody found in this library, most photographed first, named ones on top."""
    lib = require_key(x_library_key, None, key)
    return {"people": library.people_in(lib),
            "hidden": library.hidden_people(lib),
            "min_appearances": library.MIN_APPEARANCES,
            "checking": library.verdicts_pending(),
            "faces_available": faces.available()}


@app.get("/api/library/people/{person_id}/photos")
def api_library_person_photos(person_id: str, key: str = None,
                              x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    items = library.photos_of(lib, person_id)
    return {"items": _with_captions(lib, items), "person": person_id}


@app.post("/api/library/people/{person_id}/name")
def api_library_name_person(person_id: str, name: str = "", key: str = None,
                            x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    if not library.name_person(lib, person_id, name):
        raise HTTPException(404, "no such person")
    return {"id": person_id, "name": (name or "").strip() or None}


@app.post("/api/library/people/merge")
def api_library_merge_people(keep: str, absorb: str, key: str = None,
                             x_library_key: str = Header(None)):
    """Fold one person into another, for when clustering split somebody in two.

    A first-class operation rather than an afterthought: greedy clustering
    depends on arrival order and will sometimes make two piles of one person.
    Merging by hand takes five seconds; a wrong automatic merge loses
    information, which is why the threshold errs towards splitting.
    """
    lib = require_key(x_library_key, None, key)
    if not library.merge_people(lib, keep, absorb):
        raise HTTPException(404, "those two are not both in this library")
    return {"kept": keep, "absorbed": absorb, "people": library.people_in(lib)}


@app.get("/api/library/face/{face_id}/crop")
def api_library_face_crop(face_id: str, key: str = None, x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    crop = library.face_crop(lib, face_id)
    if not crop:
        raise HTTPException(404, "no such face")
    return Response(crop, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/library/settings")
def api_library_get_settings(key: str = None, x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    return {"settings": library.settings(lib), "ai_available": vision.available(),
            "provider": vision.active(), "model": vision.model_name(),
            "pending": library.queue_depth(lib)}


@app.post("/api/library/settings")
def api_library_set_settings(ai: bool = None, faces_on: bool = Query(None, alias="faces"),
                             key: str = None, x_library_key: str = Header(None)):
    """Turn the derived features on or off for this library.

    Both are off until somebody asks. Turning `ai` on only affects photographs
    uploaded afterwards plus anything already queued; it does not reach back and
    describe a library retroactively, because that would spend a surprising
    amount of somebody's money without asking.
    """
    lib = require_key(x_library_key, None, key)
    was = library.settings(lib)
    updated = library.set_settings(lib, ai=ai, faces=faces_on)
    # "Off" for biometric data means erased, not hidden. Somebody withdrawing
    # consent should not have their face vectors sitting in a table waiting to
    # be switched back on.
    forgotten = None
    if was["faces"] and not updated["faces"]:
        forgotten = library.forget_faces(lib)
    return {"settings": updated, "pending": library.queue_depth(lib),
            "forgotten": forgotten}


@app.post("/api/library/item/{item_id}/favourite")
def api_library_favourite(item_id: str, on: bool = True, key: str = None,
                          x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    if not library.set_favourite(lib, item_id, on):
        raise HTTPException(404, "no such item")
    return {"id": item_id, "favourite": on, "usage": library.usage(lib)}


@app.post("/api/library/item/{item_id}/restore")
def api_library_restore(item_id: str, key: str = None, x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    if not library.restore(lib, item_id):
        raise HTTPException(404, "no such item")
    return {"id": item_id, "usage": library.usage(lib)}


@app.post("/api/library/trash/empty")
def api_library_empty_trash(key: str = None, x_library_key: str = Header(None)):
    """Purge everything in the trash now, rather than waiting out the month."""
    lib = require_key(x_library_key, None, key)
    gone = library.purge(lib, older_than_days=0)
    return {"purged": len(gone), "usage": library.usage(lib)}


@app.get("/api/library/thumb/{item_id}")
def api_library_thumb(item_id: str, key: str = None, x_library_key: str = Header(None)):
    lib = require_key(x_library_key, None, key)
    row = library.get(lib, item_id)
    if not row or not row["preview"]:
        raise HTTPException(404, "no preview for that item")
    return Response(row["preview"], media_type=f"image/{row['preview_type'] or 'webp'}",
                    headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/library/file/{item_id}")
def api_library_file(item_id: str, key: str = None, x_library_key: str = Header(None)):
    """The stored .json itself, for anyone who wants the file rather than the picture."""
    lib = require_key(x_library_key, None, key)
    row = library.get(lib, item_id)
    body = library.payload(lib, item_id)
    if not row or body is None:
        raise HTTPException(404, "no such item")
    stem = safe_stem(row["name"] or item_id)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})


@app.get("/api/library/photo/{item_id}")
def api_library_photo(item_id: str, key: str = None, download: str = None,
                      x_library_key: str = Header(None)):
    """Decode one photograph back to a real picture.

    The expensive call, and the only one that touches the model, which is why
    the gallery grid never makes it: thumbnails come from the header instead.
    """
    lib = require_key(x_library_key, None, key)
    row = library.get(lib, item_id)
    body = library.payload(lib, item_id)
    if not row or body is None:
        raise HTTPException(404, "no such item")

    doc = json.loads(body)
    if doc.get("format") == lossless.FORMAT:
        img = lossless.decode_dict(doc)
    else:
        wanted = (doc.get("model") or {}).get("fingerprint")
        chosen = None
        for meta_row in discover_models():
            if codec.model_fingerprint(load_model(meta_row["id"])) == wanted:
                chosen = meta_row["id"]
                break
        try:
            img = codec.decode_dict(load_model(chosen or default_model_id()), doc, device="cpu")
        except ValueError as error:
            raise HTTPException(409, str(error))

    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=94,
                            icc_profile=img.info.get("icc_profile"))
    stem = safe_stem(row["name"] or item_id)
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{stem}.jpg"'
    return Response(buf.getvalue(), media_type="image/jpeg", headers=headers)


class _Drain(io.RawIOBase):
    """A file object that hands whatever is written to it straight to a generator.

    zipfile insists on writing to something; a streaming response insists on
    yielding. This is the joint between them: ZipFile writes here, the chunks
    pile up in a list, and the generator below drains that list after each
    photograph. Nothing larger than one decoded JPEG is ever held in memory, and
    bytes reach the browser while the rest of the archive is still being made.
    """

    def __init__(self):
        self.chunks = []

    def writable(self):
        return True

    def write(self, data):
        self.chunks.append(bytes(data))
        return len(data)

    def drain(self):
        out, self.chunks = b"".join(self.chunks), []
        return out


BULK_MAX = int(os.environ.get("JSONCAM_BULK_MAX", "40"))


@app.get("/api/library/zip")
def api_library_zip(ids: str = None, key: str = None, view: str = None,
                    kind: str = "photos", x_library_key: str = Header(None)):
    """Decode a selection and stream it back as one archive.

    Decoding is seconds per photograph, so a selection of thirty is minutes of
    work. Doing it as one blocking request would sit silent long enough for
    every proxy in the path to give up, so the archive is streamed: each picture
    is decoded, appended, and flushed before the next one starts.
    """
    lib = require_key(x_library_key, None, key)
    if view:
        # The whole library. Only allowed for `kind=files`, which does no
        # decoding: asking a single core to decode two thousand photographs
        # inside one request is not a download, it is an outage.
        if kind != "files":
            raise HTTPException(400, (
                "Downloading a whole library as photos would take hours. Use kind=files for the "
                "complete backup, which needs no decoding, or select a batch."))
        wanted = library.all_ids(lib, view=view)
    else:
        wanted = [i for i in (ids or "").split(",") if i][:BULK_MAX]
    if not wanted:
        raise HTTPException(400, "nothing selected")

    rows = [(i, library.get(lib, i)) for i in wanted]
    rows = [(i, r) for i, r in rows if r]
    if not rows:
        raise HTTPException(404, "none of those are in this library")

    def stream():
        import zipfile

        sink = _Drain()
        seen = {}
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as archive:
            for item_id, row in rows:
                body = library.payload(lib, item_id)
                if body is None:
                    continue

                stem_only = safe_stem(row["name"] or item_id)
                if kind == "files":
                    # The .json exactly as stored: the real backup of a library,
                    # and instant, because nothing is decoded.
                    seen[stem_only] = seen.get(stem_only, 0) + 1
                    suffix = "" if seen[stem_only] == 1 else f" ({seen[stem_only]})"
                    archive.writestr(f"{stem_only}{suffix}.json", body)
                    yield sink.drain()
                    continue

                try:
                    doc = json.loads(body)
                    if doc.get("format") == lossless.FORMAT:
                        img = lossless.decode_dict(doc)
                    else:
                        img = codec.decode_dict(load_model(default_model_id()), doc,
                                                device="cpu")
                except Exception:
                    # One bad file must not truncate an archive of forty.
                    continue
                buf = io.BytesIO()
                img.convert("RGB").save(buf, "JPEG", quality=94)

                # Two photos off a phone are very often both IMG_0042.jpg.
                stem = safe_stem(row["name"] or item_id)
                seen[stem] = seen.get(stem, 0) + 1
                name = f"{stem}.jpg" if seen[stem] == 1 else f"{stem} ({seen[stem]}).jpg"
                archive.writestr(name, buf.getvalue())
                yield sink.drain()
        yield sink.drain()

    stamp = time.strftime("%Y-%m-%d")
    label = "library" if kind == "files" else "photos"
    return StreamingResponse(
        stream(), media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="json-camera {label} {stamp}.zip"'})


@app.delete("/api/library/item/{item_id}")
def api_library_delete(item_id: str, key: str = None, forever: bool = False,
                       x_library_key: str = Header(None)):
    """Move a photograph to the trash, or with `forever`, actually delete it.

    Soft by default. Hard deleting on the first tap is the one mistake in a
    photo library that cannot be walked back, and a month of grace costs
    nothing but disk that the quota already accounts for.
    """
    lib = require_key(x_library_key, None, key)
    if forever:
        if not library.purge(lib, item_id=item_id):
            raise HTTPException(404, "that is not in the trash")
        return {"deleted": item_id, "forever": True, "usage": library.usage(lib)}
    if not library.trash(lib, item_id):
        raise HTTPException(404, "no such item")
    return {"deleted": item_id, "forever": False, "recoverable_days": library.TRASH_DAYS,
            "usage": library.usage(lib)}


# --------------------------------------------------------------------------
# admin
#
# One key for the whole server, separate from every library key, and it grants
# something different in kind: how the service behaves, not what is in one
# person's photographs. Nothing here can read or list a library, and the
# overview deliberately counts libraries rather than naming them, because the
# handles are hashes and there is nothing in this design that identifies a
# person. An admin page that quietly became a way to browse strangers'
# photographs would be a worse thing than no admin page.
#
# Refuses outright when no key is configured, rather than defaulting to open.

ADMIN_KEY = os.environ.get("JSONCAM_ADMIN_KEY", "")


def require_admin(header_key, query_key=None):
    import hmac

    if not ADMIN_KEY:
        raise HTTPException(503, (
            "No admin key is configured on this server. Set JSONCAM_ADMIN_KEY and redeploy; "
            "until then there is deliberately no way in."))
    supplied = header_key or query_key or ""
    if not supplied or not hmac.compare_digest(supplied, ADMIN_KEY):
        raise HTTPException(401, "wrong admin key")
    return True


@app.get("/api/admin/overview")
def api_admin_overview(key: str = None, x_admin_key: str = Header(None)):
    """Everything the admin page needs in one request."""
    require_admin(x_admin_key, key)
    provider = vision.active()
    return {
        "totals": library.totals(),
        "capabilities": {
            "heic": HEIF_OK,
            "faces": faces.available(),
            "captions": vision.available(),
            "claude_key": vision.claude_ready(),
            "deepseek_key": vision.deepseek_ready(),
        },
        "vision": {
            "provider": provider,
            "provider_setting": vision.provider_setting(),
            "model": vision.model_name(),
            "cost_per_photo": vision.cost_per_photo(),
            "claude_choices": list(vision.CLAUDE_CHOICES),
            "deepseek_choices": list(vision.DEEPSEEK_CHOICES),
            "costs": vision.COST,
            "env_provider": vision.ENV_PROVIDER,
            "env_claude_model": vision.ENV_CLAUDE_MODEL,
            "env_deepseek_model": vision.ENV_DEEPSEEK_MODEL,
        },
        "faces": {
            "available": faces.available(),
            "match": faces.MATCH,
            "adjudicate": faces.ADJUDICATE,
            "same_look": faces.SAME_LOOK,
            "margin": faces.MARGIN,
            "min_edge": faces.MIN_EDGE,
            "max_yaw": faces.MAX_YAW,
            "min_sharpness": faces.MIN_SHARPNESS,
            "confidence": faces.CONFIDENCE,
            "min_appearances": library.MIN_APPEARANCES,
        },
        "codec": {
            "models": discover_models(),
            "default": default_model_id(),
            "max_side": LIBRARY_MAX_SIDE,
            "max_upload": MAX_UPLOAD,
            "batch_max": BATCH_MAX,
            "max_lossless_mp": MAX_LOSSLESS_MP,
        },
        "limits": {
            "max_items": library.MAX_ITEMS,
            "max_bytes": library.MAX_BYTES,
            "trash_days": library.TRASH_DAYS,
        },
        "overrides": library.all_config(),
    }


# Only these can be set from the web, and each is validated. A free-text
# config endpoint is a way to break the service from a browser.
ADMIN_SETTABLE = {
    "vision_provider": ("auto", "claude", "deepseek"),
    "vision_model_claude": vision.CLAUDE_CHOICES,
    "vision_model_deepseek": vision.DEEPSEEK_CHOICES,
    "codec_model": None,        # validated against the checkpoints actually present
}


@app.post("/api/admin/config")
def api_admin_config(name: str, value: str = "", key: str = None,
                     x_admin_key: str = Header(None)):
    """Set one override, or clear it with an empty value.

    Clearing is how you go back to the deployed default rather than having to
    remember what it was.
    """
    require_admin(x_admin_key, key)
    if name not in ADMIN_SETTABLE:
        raise HTTPException(400, f"{name} is not a setting this endpoint can change")

    if value:
        allowed = ADMIN_SETTABLE[name]
        if name == "codec_model":
            allowed = tuple(m["id"] for m in discover_models())
        if allowed and value not in allowed:
            raise HTTPException(400, (
                f"{value!r} is not one of {list(allowed)}. Refused rather than accepted, "
                f"because a model name that does not exist looks exactly like a feature "
                f"that stopped working."))

    stored = library.set_config(name, value)
    return {"name": name, "value": stored,
            "cleared": stored is None,
            "vision": {"provider": vision.active(), "model": vision.model_name(),
                       "cost_per_photo": vision.cost_per_photo()}}


@app.post("/api/admin/requeue")
def api_admin_requeue(what: str = "failed", key: str = None,
                      x_admin_key: str = Header(None)):
    """Put photographs back in the captioning queue.

    `failed` retries the ones that used up their attempts, which is what you
    want after fixing a missing key. `all` queues photographs that were never
    described at all, which is the back catalogue and costs real money, so it
    reports how many it queued before any of it runs.
    """
    require_admin(x_admin_key, key)
    if what == "failed":
        count = library.requeue_failed()
    elif what == "all":
        count = library.requeue_all()
    else:
        raise HTTPException(400, "what must be 'failed' or 'all'")
    estimate = vision.cost_per_photo()
    return {"queued": count, "what": what,
            "estimated_cost": round(count * estimate, 2) if estimate else None,
            "model": vision.model_name()}


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
    # Lives at the repo root, outside the directories the image otherwise copies,
    # so it went missing in the container once and surfaced as a bare 500.
    path = ROOT / "AGENTS.md"
    if not path.exists():
        raise HTTPException(404, "AGENTS.md is not present in this deployment")
    return Response(path.read_text(encoding="utf-8"),
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
