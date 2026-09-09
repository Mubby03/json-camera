"""A keyed photo library: what the Shortcut writes into and the gallery reads out of.

The two halves of the phone story need somewhere to meet.  A Shortcut on an
iPhone can loop over four hundred photographs and POST them one at a time, but
it has nowhere to keep the results and no way to draw them.  A web gallery can
draw anything but cannot reach into the camera roll.  So the meeting point is
here: an append-only store on the mounted volume, and an index over it.

**Keys, not accounts.**  A library is addressed by a secret the owner generates
and keeps.  There is no sign-up, no password reset and no email, which is the
same trade the chat endpoint in `server.py` already makes and for the same
reason: an account system is a large thing to maintain and the only question it
answers here is "are these your photographs".  A 32-character secret answers
that too.  The key is never stored, only its SHA-256, so a copy of the database
does not hand somebody every library in it.

**SQLite, not a folder of files.**  Thousands of small files is slow to list, has
nowhere to put a capture date, and makes "show me last August" a directory walk.
One index with the metadata pulled out of the header at upload time makes the
gallery a query.  The payloads stay on disk beside it, because a few hundred
kilobytes of base85 per row is not what SQLite is good at.

**Thumbnails are served from the header.**  The preview block that `jsoncam.meta`
embeds is what the grid draws, so filling a screen with sixty photographs costs
sixty small WebP requests and no model inference at all.  Decoding only happens
when somebody asks for a full picture, which is the expensive thing and is
therefore the thing that happens once, on purpose, per photograph.
"""

import base64
import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

LIBRARY_DIR = Path(os.environ.get("JSONCAM_LIBRARY_DIR", "/data/library"))
DB_PATH = LIBRARY_DIR / "index.db"

# A library is capped so one enthusiastic phone cannot fill the volume and take
# the chat and every other library down with it. Both numbers are per key.
MAX_ITEMS = int(os.environ.get("JSONCAM_LIBRARY_MAX_ITEMS", "2000"))
MAX_BYTES = int(os.environ.get("JSONCAM_LIBRARY_MAX_BYTES", str(512 * 1024 * 1024)))

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id           TEXT PRIMARY KEY,
    library      TEXT NOT NULL,
    name         TEXT,
    stored_at    REAL NOT NULL,
    captured_at  TEXT,
    width        INTEGER,
    height       INTEGER,
    json_bytes   INTEGER,
    source_bytes INTEGER,
    lossless     INTEGER DEFAULT 0,
    camera       TEXT,
    lens         TEXT,
    lat          REAL,
    lon          REAL,
    fingerprint  TEXT,
    preview      BLOB,
    preview_type TEXT,
    meta         TEXT
);
-- The gallery's only ordering is newest-first within one library, and its only
-- filter is the library, so this one index answers every query the app makes.
CREATE INDEX IF NOT EXISTS items_by_library
    ON items (library, captured_at DESC, stored_at DESC);
"""


def _connect():
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL so a long upload does not block the gallery reading in another request.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def library_id(key):
    """The public handle for a secret key.

    Hashed, so the database never holds the thing that grants access to it, and
    truncated to 32 hex characters, which is still far past the point where
    guessing one is worth anybody's time.
    """
    return hashlib.sha256(f"jsoncam-library-v1:{key}".encode()).hexdigest()[:32]


def new_key():
    """A fresh library key. 32 url-safe characters, generated on the server."""
    return secrets.token_urlsafe(24)


def _payload_path(lib, item_id):
    return LIBRARY_DIR / lib[:2] / lib / f"{item_id}.json"


def usage(lib):
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(json_bytes), 0) AS b "
            "FROM items WHERE library = ?", (lib,)).fetchone()
    return {"items": row["n"], "bytes": row["b"],
            "max_items": MAX_ITEMS, "max_bytes": MAX_BYTES}


def add(lib, doc, raw_json, source_bytes=None):
    """Store one encoded photograph and index what the gallery needs to sort it.

    `raw_json` is the serialised container exactly as it will be served back, so
    the bytes a caller downloads are the bytes that were measured here.
    """
    info = doc.get("meta") or {}
    image = doc.get("image") or {}
    preview = doc.get("preview") or {}
    blob = None
    if preview.get("data"):
        try:
            blob = base64.b64decode(preview["data"])
        except Exception:
            blob = None

    item_id = secrets.token_urlsafe(12).replace("-", "_")
    path = _payload_path(lib, item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_json)

    place = info.get("gps") or {}
    row = {
        "id": item_id,
        "library": lib,
        "name": image.get("name"),
        "stored_at": time.time(),
        # Sorting falls back to arrival time for anything with no capture date,
        # such as a screenshot, so those land at the top rather than vanishing
        # to the bottom of a library ordered by a column full of nulls.
        "captured_at": info.get("captured_at"),
        "width": image.get("width"),
        "height": image.get("height"),
        "json_bytes": len(raw_json),
        "source_bytes": source_bytes,
        "lossless": 1 if doc.get("format", "").startswith("json-camera/lossless") else 0,
        "camera": info.get("camera"),
        "lens": info.get("lens"),
        "lat": place.get("lat"),
        "lon": place.get("lon"),
        "fingerprint": (doc.get("model") or {}).get("fingerprint"),
        "preview": blob,
        "preview_type": preview.get("format"),
        "meta": json.dumps(info) if info else None,
    }
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO items ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
            tuple(row.values()))
    return item_id


def listing(lib, limit=500, offset=0):
    """Everything the grid needs, without the payloads or the thumbnails.

    Thumbnails are a separate request each so the browser caches them
    individually and a listing stays small enough to parse on a phone.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, stored_at, captured_at, width, height, json_bytes, "
            "       source_bytes, lossless, camera, lens, lat, lon, "
            "       preview IS NOT NULL AS has_preview "
            "FROM items WHERE library = ? "
            "ORDER BY COALESCE(captured_at, datetime(stored_at, 'unixepoch')) DESC, "
            "         stored_at DESC LIMIT ? OFFSET ?",
            (lib, limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get(lib, item_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM items WHERE library = ? AND id = ?",
                           (lib, item_id)).fetchone()
    return dict(row) if row else None


def payload(lib, item_id):
    """The stored container, as bytes."""
    path = _payload_path(lib, item_id)
    return path.read_bytes() if path.exists() else None


def remove(lib, item_id):
    with _connect() as conn:
        cur = conn.execute("DELETE FROM items WHERE library = ? AND id = ?", (lib, item_id))
        gone = cur.rowcount > 0
    if gone:
        path = _payload_path(lib, item_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return gone
