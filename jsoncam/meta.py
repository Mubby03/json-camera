"""Sidecar data that rides inside the container: EXIF, and a preview thumbnail.

Both exist for the same reason, which is that a `.json` photograph is otherwise
unusable as a photograph.  Nothing on any operating system can draw one, and
nothing can sort one, so an archive of them is a pile of files a person can
neither look through nor put in order.  The codec is not what is missing.  Two
small blocks in the header are.

**EXIF.**  A phone records when the shutter opened, where it was pointing, and
which lens took it.  That is not decoration, it is the entire basis on which any
photo library organises itself: by day, by trip, by camera.  Convert ten
thousand photographs without carrying it and you get ten thousand files ordered
by nothing at all.  `extract` pulls the fields worth keeping into plain
JSON-safe types, since raw EXIF is full of rationals, byte strings and tuples
that `json.dump` refuses.

**Preview.**  A small WebP of the picture, base64'd into the header.  It costs a
few percent of the file and buys the ability to *see* the photograph without the
decoder, the checkpoint, or torch: a gallery, a Finder thumbnail and a phone all
read it with code they already have.  A format that can only be viewed by the
machine that wrote it is a trap, and this is the cheapest way out of it.

Both blocks are optional and both are additive.  A reader that does not know
about them ignores two extra keys, so files written by this module still open in
any decoder that could read the format before it existed.
"""

import base64
import io

from PIL import Image, ImageOps

# Measured, not guessed. On six DIV2K photographs at 1600px, where the learned
# codec writes a bitstream of roughly 75 KB, WebP previews cost:
#
#     side  quality   mean     share of the bitstream
#      256       60   7.6 KB        10%
#      320       60  11.2 KB        15%
#      384       60  15.5 KB        21%
#      512       70  29.2 KB        40%
#
# 320 is the last row where the preview is still clearly a rounding error next
# to the picture it belongs to, and it is enough to fill a gallery tile on a 3x
# phone screen, stand in full-bleed while the real decode arrives, and give
# Finder something to draw. 512 was the first thing tried and 40% is not a
# preview, it is a second copy of the photograph. Raise it if you want, but know
# what it costs: this comes straight off the compression ratio.
PROXY_SIDE = 320
PROXY_QUALITY = 60

# Tags, by number, so this does not depend on Pillow's name table.
_ORIENTATION = 274
_MAKE = 271
_MODEL = 272
_SOFTWARE = 305
_DATETIME = 306

_DATETIME_ORIGINAL = 36867
_DATETIME_DIGITIZED = 36868
_EXPOSURE_TIME = 33434
_F_NUMBER = 33437
_ISO = 34855
_FOCAL_LENGTH = 37386
_FOCAL_35MM = 41989
_LENS_MODEL = 42036


def _plain(value):
    """Coerce one EXIF value into something json.dump will accept."""
    if isinstance(value, bytes):
        # Byte strings in EXIF are usually latin-1 text with a trailing NUL.
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    if isinstance(value, str):
        value = value.replace("\x00", "").strip()
        return value or None
    if isinstance(value, (int, float)):
        return value
    # IFDRational and friends: numerator/denominator, and a zero denominator is
    # a real thing in files written by real cameras.
    num = getattr(value, "numerator", None)
    den = getattr(value, "denominator", None)
    if num is not None and den is not None:
        return None if not den else round(num / den, 6)
    if isinstance(value, (tuple, list)):
        out = [_plain(v) for v in value]
        return [v for v in out if v is not None] or None
    return None


def _timestamp(raw):
    """EXIF writes `2026:09:09 14:03:11`. Turn it into something sortable."""
    text = _plain(raw)
    if not text or len(text) < 19:
        return None
    date, _, clock = text.partition(" ")
    date = date.replace(":", "-")
    stamp = f"{date}T{clock}"
    # Refuse anything that is not actually a date rather than poison the index
    # with a string that sorts between real ones.
    try:
        from datetime import datetime

        datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return stamp[:19]


def _coordinate(values, ref):
    """GPS stores degrees, minutes and seconds separately. Fold them to one float."""
    if not values or len(values) < 3:
        return None
    parts = [_plain(v) for v in values[:3]]
    if any(p is None for p in parts):
        return None
    deg, minutes, seconds = parts
    value = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref or "").upper() in ("S", "W"):
        value = -value
    return round(value, 6)


def extract(img, gps=True):
    """Pull the organising fields out of a PIL image.

    Call this *before* `exif_transpose`, which consumes the orientation tag.
    Returns None when the image carries nothing worth recording, so the header
    does not grow an empty block for every screenshot and rendered chart.

    `gps` is a switch rather than an assumption. Location is the one field here
    that is genuinely sensitive: it says where someone lives, and it survives
    every copy of the file made afterwards. Callers converting a library for
    themselves want it, and callers about to hand the files to somebody else
    very much do not, so neither answer can be baked in.
    """
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None

    out = {}

    make = _plain(exif.get(_MAKE))
    model = _plain(exif.get(_MODEL))
    if make and model:
        # "Apple" + "iPhone 15 Pro" reads as one thing, not two, and phones
        # already prefix the model with the make often enough to check.
        out["camera"] = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    elif make or model:
        out["camera"] = make or model

    if (software := _plain(exif.get(_SOFTWARE))):
        out["software"] = software
    if (orientation := _plain(exif.get(_ORIENTATION))):
        out["orientation"] = orientation

    try:
        ifd = exif.get_ifd(0x8769)          # the Exif sub-IFD
    except Exception:
        ifd = {}

    captured = (_timestamp(ifd.get(_DATETIME_ORIGINAL))
                or _timestamp(ifd.get(_DATETIME_DIGITIZED))
                or _timestamp(exif.get(_DATETIME)))
    if captured:
        out["captured_at"] = captured

    if (lens := _plain(ifd.get(_LENS_MODEL))):
        out["lens"] = lens

    shot = {}
    if (v := _plain(ifd.get(_EXPOSURE_TIME))) is not None:
        shot["exposure_seconds"] = v
    if (v := _plain(ifd.get(_F_NUMBER))) is not None:
        shot["aperture"] = v
    if (v := _plain(ifd.get(_ISO))) is not None:
        shot["iso"] = v if isinstance(v, (int, float)) else None
    if (v := _plain(ifd.get(_FOCAL_LENGTH))) is not None:
        shot["focal_mm"] = v
    if (v := _plain(ifd.get(_FOCAL_35MM))) is not None:
        shot["focal_35mm"] = v
    shot = {k: v for k, v in shot.items() if v is not None}
    if shot:
        out["shot"] = shot

    if gps:
        try:
            gps_ifd = exif.get_ifd(0x8825)
        except Exception:
            gps_ifd = {}
        if gps_ifd:
            lat = _coordinate(gps_ifd.get(2), _plain(gps_ifd.get(1)))
            lon = _coordinate(gps_ifd.get(4), _plain(gps_ifd.get(3)))
            if lat is not None and lon is not None:
                place = {"lat": lat, "lon": lon}
                altitude = _plain(gps_ifd.get(6))
                if altitude is not None:
                    # Reference 1 means below sea level, which is rare and real.
                    place["altitude_m"] = -altitude if _plain(gps_ifd.get(5)) == 1 else altitude
                out["gps"] = place

    return out or None


def proxy(img, side=PROXY_SIDE, quality=PROXY_QUALITY):
    """Build the embedded preview from an already-upright RGB image.

    Returns None rather than raising if the encoder is unavailable, because a
    missing thumbnail must never be the reason a photograph fails to compress.
    """
    try:
        thumb = img.convert("RGB")
        thumb = ImageOps.contain(thumb, (side, side), Image.LANCZOS)
        buf = io.BytesIO()
        # WebP is roughly 25% under JPEG at this size and quality. Pillow is
        # built with it almost everywhere, but "almost" is not "always" and a
        # server that cannot write WebP should still produce a usable file.
        fmt = "webp"
        try:
            thumb.save(buf, "WEBP", quality=quality, method=6)
        except Exception:
            buf = io.BytesIO()
            fmt = "jpeg"
            thumb.save(buf, "JPEG", quality=quality, optimize=True)
        blob = buf.getvalue()
    except Exception:
        return None
    return {
        "format": fmt,
        "width": thumb.width,
        "height": thumb.height,
        "bytes": len(blob),
        "data": base64.b64encode(blob).decode("ascii"),
    }


def proxy_bytes(doc):
    """The embedded preview of a container, as raw bytes and a media type.

    Returns (None, None) when the file carries no preview, which every file
    written before this block existed does.
    """
    block = (doc or {}).get("preview")
    if not block or not block.get("data"):
        return None, None
    try:
        blob = base64.b64decode(block["data"])
    except Exception:
        return None, None
    return blob, f"image/{block.get('format', 'webp')}"


def data_uri(doc):
    """The embedded preview as a `data:` URI, ready to drop into an <img src>."""
    blob, media = proxy_bytes(doc)
    if not blob:
        return None
    return f"data:{media};base64,{base64.b64encode(blob).decode('ascii')}"
