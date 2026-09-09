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
    meta         TEXT,
    favourite    INTEGER DEFAULT 0,
    -- Soft delete, the way every photo app does it: a deleted photo is out of
    -- the way but recoverable for a month. Hard deleting on the first tap is
    -- the one mistake in a photo library that cannot be walked back.
    deleted_at   REAL,
    place        TEXT
);
-- The gallery's only ordering is newest-first within one library, and its only
-- filter is the library, so this one index answers every query the app makes.
CREATE INDEX IF NOT EXISTS items_by_library
    ON items (library, deleted_at, captured_at DESC, stored_at DESC);

-- Reverse geocoding is rate limited and the same coordinates recur constantly,
-- so answers are cached on a rounded grid rather than asked for twice.
CREATE TABLE IF NOT EXISTS places (
    cell    TEXT PRIMARY KEY,
    name    TEXT,
    asked_at REAL
);

-- What the vision model saw. One row per photograph, written pending at upload
-- and filled in by the worker, so this is computed exactly once per photo and
-- never on a page load. `done_at IS NULL` is the queue.
CREATE TABLE IF NOT EXISTS analysis (
    item     TEXT PRIMARY KEY,
    library  TEXT NOT NULL,
    caption  TEXT,
    tags     TEXT,          -- JSON array
    text     TEXT,          -- text legible in the image
    people   INTEGER,
    kind     TEXT,
    search   TEXT,          -- everything above, lowercased, for matching
    model    TEXT,
    attempts INTEGER DEFAULT 0,
    done_at  REAL
);
CREATE INDEX IF NOT EXISTS analysis_pending ON analysis (done_at, attempts);
CREATE INDEX IF NOT EXISTS analysis_by_library ON analysis (library);

-- Faces found in photographs, and the people they were grouped into.
--
-- The embedding is 128 float32s as a blob. Matching a new face compares it
-- against one centroid per person rather than against every face ever stored,
-- which is what keeps this cheap enough to run inline at upload.
CREATE TABLE IF NOT EXISTS faces (
    id        TEXT PRIMARY KEY,
    item      TEXT NOT NULL,
    library   TEXT NOT NULL,
    person    TEXT,
    bbox      TEXT,          -- JSON [x, y, w, h], normalised 0..1
    embedding BLOB,
    crop      BLOB,          -- a 96px JPEG of the face, for the picker
    score     REAL,
    edge      INTEGER,
    found_at  REAL
);
CREATE INDEX IF NOT EXISTS faces_by_item ON faces (item);
CREATE INDEX IF NOT EXISTS faces_by_person ON faces (library, person);

CREATE TABLE IF NOT EXISTS people (
    id         TEXT PRIMARY KEY,
    library    TEXT NOT NULL,
    name       TEXT,
    centroid   BLOB,
    face_count INTEGER DEFAULT 0,
    cover      TEXT,          -- the face id whose crop represents them
    cover_edge INTEGER DEFAULT 0,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS people_by_library ON people (library, face_count DESC);

-- Per-library switches. Both default off: the vision pass costs the operator
-- money per photograph and faces are biometric data about people who never
-- agreed to anything, so neither can be an assumption.
CREATE TABLE IF NOT EXISTS settings (
    library    TEXT PRIMARY KEY,
    ai         INTEGER DEFAULT 0,
    faces      INTEGER DEFAULT 0,
    updated_at REAL
);
"""

# Columns added after the first release. SQLite has no ADD COLUMN IF NOT EXISTS,
# so this is the idiom: try each, ignore the one error that means "already there".
MIGRATIONS = (
    "ALTER TABLE items ADD COLUMN favourite INTEGER DEFAULT 0",
    "ALTER TABLE items ADD COLUMN deleted_at REAL",
    "ALTER TABLE items ADD COLUMN place TEXT",
    # 0 means "resolved under an older wording", which is also what every row
    # written before this column existed gets. Those are re-resolved rather than
    # left reading like a postal address forever.
    "ALTER TABLE items ADD COLUMN place_version INTEGER DEFAULT 0",
)


def _connect():
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL so a long upload does not block the gallery reading in another request.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    for statement in MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass                      # the column is already there
    return conn


def library_id(key):
    """The public handle for a secret key.

    Hashed, so the database never holds the thing that grants access to it, and
    truncated to 32 hex characters, which is still far past the point where
    guessing one is worth anybody's time.
    """
    return hashlib.sha256(f"jsoncam-library-v1:{key}".encode()).hexdigest()[:32]


# Keys carry a version prefix and a checksum. Both exist to answer one question
# the original design could not: is this key wrong, or is this library empty?
#
# Without a checksum those two are indistinguishable. A key is not checked
# against a list of real ones, because there is no such list, so one mistyped
# character silently addresses a different, empty library. Somebody who has
# uploaded four hundred photographs then opens the gallery, sees nothing, and
# reasonably concludes they are gone. That is the worst screen this app can
# show, and it is caused by a typo.
#
# The prefix is what makes the verdict safe to give. Keys minted before this
# existed are 32 random characters with no checksum, and a checksum test on one
# would fail and accuse a perfectly good key of being a typo. `jc1_` says "this
# key was built to be checkable", so anything without it gets no verdict rather
# than a wrong one.
KEY_PREFIX = "jc1_"
KEY_BODY_BYTES = 21          # -> 28 url-safe characters, 168 bits
KEY_CHECK_CHARS = 4


def _check_chars(body):
    digest = hashlib.sha256(f"jsoncam-key-check-v1:{body}".encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()[:KEY_CHECK_CHARS]


def new_key():
    """A fresh library key, self-describing and self-checking."""
    body = secrets.token_urlsafe(KEY_BODY_BYTES)
    return f"{KEY_PREFIX}{body}{_check_chars(body)}"


def inspect_key(key):
    """Say what is wrong with a key, when anything can be said.

    Returns one of:
      "ok"      the checksum agrees, so this is the key it was meant to be
      "typo"    built to be checkable, and the check fails: a character is wrong
      "short"   not long enough to be a key at all
      "legacy"  no verdict available, which is the honest answer for old keys

    Deliberately does not say whether the library exists or has anything in it.
    That would turn the endpoint into an oracle for guessing keys, and it is not
    the question anybody is actually asking.
    """
    key = (key or "").strip()
    if len(key) < 16:
        return "short"
    if not key.startswith(KEY_PREFIX):
        return "legacy"
    rest = key[len(KEY_PREFIX):]
    if len(rest) <= KEY_CHECK_CHARS:
        return "typo"
    body, check = rest[:-KEY_CHECK_CHARS], rest[-KEY_CHECK_CHARS:]
    return "ok" if check == _check_chars(body) else "typo"


# --------------------------------------------------------------------------
# place names
#
# Coordinates are not an answer to "where was this". Nobody recognises
# 6.4540, 3.4092; everybody recognises Lekki. So the numbers get turned into a
# name once, and the name is what the gallery shows.
#
# OpenStreetMap's Nominatim does this for free and asks three things in return:
# one request a second, a real User-Agent, and no repeat lookups. The cache
# below is how the third is honoured, and it is what makes this cheap: a day out
# is forty photographs within a few hundred metres of each other, so rounding
# the coordinates to a grid turns forty lookups into one.

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
# Three decimal places is about 110 metres, which is the right grain: finer
# splits one beach into six cells, coarser merges neighbouring streets.
PLACE_GRID = 3

# Zoom 17 is street level. Lower and the answer is a district or a city, which
# is too coarse to be the "near X" people actually say.
PLACE_ZOOM = 17

# Bumped when the wording below changes, so cached answers and stored labels
# from an older shape are re-resolved instead of sitting there in the old
# format forever.
PLACE_FORMAT = 2

_last_lookup = [0.0]


def _cell(lat, lon):
    return f"v{PLACE_FORMAT}:{round(lat, PLACE_GRID)},{round(lon, PLACE_GRID)}"


# The most recognisable local name, most specific first.
#
# Not a postal address. "Dolphin Estate, Eti Osa, Nigeria" is technically
# correct and nobody has ever said it; "near Stratford" is what a person
# actually calls that place. Roads sit below neighbourhoods because a district
# is the more memorable unit for finding a photograph again, but above towns,
# because in plenty of places the road is the only thing mapped and "near Akala
# Way" beats "near Ibadan North".
PLACE_KEYS = ("neighbourhood", "suburb", "quarter", "village", "hamlet",
              "road", "pedestrian", "town", "city_district", "city")

# Categories where Nominatim's top-level `name` is just the road again rather
# than a landmark worth naming.
NOT_A_LANDMARK = ("highway", "place", "boundary", "landuse")


def _label(payload):
    """A short "near somewhere" from a Nominatim answer, or None."""
    payload = payload or {}
    address = payload.get("address") or {}

    # A named landmark is the best anchor there is: "near Victoria Park" tells
    # you more than any street will.
    name = payload.get("name")
    if name and payload.get("category") not in NOT_A_LANDMARK:
        return f"near {name}"

    for key in PLACE_KEYS:
        value = address.get(key)
        if value:
            return f"near {value}"
    return None


def place_for(lat, lon):
    """A human place name for a coordinate, cached, or None.

    Never raises and never blocks for long: a photograph must still upload if
    OpenStreetMap is slow or down, so a failure here just means the gallery
    shows coordinates for that one instead of a name.
    """
    if lat is None or lon is None:
        return None
    cell = _cell(lat, lon)
    with _connect() as conn:
        row = conn.execute("SELECT name FROM places WHERE cell = ?", (cell,)).fetchone()
    if row:
        return row["name"]

    try:
        import json as _json
        import urllib.parse
        import urllib.request

        # Their usage policy is one request per second, and this process is the
        # only caller, so a sleep here is enough to honour it.
        wait = 1.05 - (time.time() - _last_lookup[0])
        if wait > 0:
            time.sleep(wait)
        _last_lookup[0] = time.time()

        query = urllib.parse.urlencode({
            "lat": f"{lat:.5f}", "lon": f"{lon:.5f}",
            "format": "jsonv2", "zoom": str(PLACE_ZOOM), "addressdetails": "1",
        })
        request = urllib.request.Request(
            f"{NOMINATIM}?{query}",
            headers={"User-Agent": "json-camera/1.0 (https://mubby.space/json-camera)"})
        with urllib.request.urlopen(request, timeout=6) as response:
            name = _label(_json.loads(response.read().decode("utf-8")))
    except Exception:
        return None

    if name:
        with _connect() as conn:
            conn.execute("INSERT OR REPLACE INTO places (cell, name, asked_at) VALUES (?, ?, ?)",
                         (cell, name, time.time()))
    return name


# --------------------------------------------------------------------------
# settings
#
# Off by default, both of them, and that is the whole point of them existing.
# The vision pass spends the operator's money on every photograph, and face
# recognition is biometric data about people who never agreed to it, so turning
# either on has to be somebody's decision rather than a default nobody saw.

def settings(lib):
    with _connect() as conn:
        row = conn.execute("SELECT ai, faces FROM settings WHERE library = ?", (lib,)).fetchone()
    return {"ai": bool(row["ai"]) if row else False,
            "faces": bool(row["faces"]) if row else False}


def set_settings(lib, ai=None, faces=None):
    current = settings(lib)
    merged = {"ai": current["ai"] if ai is None else bool(ai),
              "faces": current["faces"] if faces is None else bool(faces)}
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (library, ai, faces, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(library) DO UPDATE SET ai = ?, faces = ?, updated_at = ?",
            (lib, int(merged["ai"]), int(merged["faces"]), time.time(),
             int(merged["ai"]), int(merged["faces"]), time.time()))
    return merged


# --------------------------------------------------------------------------
# the analysis queue

# A photograph the model cannot describe should not be retried forever. Three
# attempts covers a transient outage; past that it is something about the file.
MAX_ATTEMPTS = 3


def enqueue_analysis(lib, item_id):
    """Mark a photograph as wanting a caption."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO analysis (item, library, attempts) VALUES (?, ?, 0)",
            (item_id, lib))


def next_pending(limit=1):
    """Photographs waiting to be described, oldest first, across all libraries."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT a.item, a.library, i.preview, i.preview_type, i.name, i.camera, "
            "       i.place, i.captured_at "
            "FROM analysis a JOIN items i ON i.id = a.item "
            "WHERE a.done_at IS NULL AND a.attempts < ? AND i.deleted_at IS NULL "
            "LIMIT ?", (MAX_ATTEMPTS, limit)).fetchall()
    return [dict(r) for r in rows]


def note_attempt(item_id):
    with _connect() as conn:
        conn.execute("UPDATE analysis SET attempts = attempts + 1 WHERE item = ?", (item_id,))


def save_analysis(lib, item_id, data, search):
    with _connect() as conn:
        conn.execute(
            "UPDATE analysis SET caption = ?, tags = ?, text = ?, people = ?, kind = ?, "
            "       search = ?, model = ?, done_at = ? WHERE item = ? AND library = ?",
            (data.get("caption"), json.dumps(data.get("tags") or []), data.get("text"),
             data.get("people"), data.get("kind"), search, data.get("model"),
             time.time(), item_id, lib))


def analysis_for(lib, item_ids):
    """Captions for a page of the gallery, in one query rather than one each."""
    if not item_ids:
        return {}
    marks = ",".join("?" * len(item_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT item, caption, tags, text, people, kind FROM analysis "
            f"WHERE library = ? AND item IN ({marks})", (lib, *item_ids)).fetchall()
    out = {}
    for r in rows:
        out[r["item"]] = {
            "caption": r["caption"],
            "tags": json.loads(r["tags"]) if r["tags"] else [],
            "text": r["text"],
            "people": r["people"],
            "kind": r["kind"],
        }
    return out


def next_place_backfill(limit=1):
    """Photographs whose place label predates the current wording.

    Rate limiting is the reason this is a queue rather than a migration: OSM
    allows one lookup a second, so a library of four hundred photographs taken
    across twenty places is twenty lookups spread over twenty seconds, not one
    request that hangs.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, library, lat, lon FROM items "
            "WHERE lat IS NOT NULL AND lon IS NOT NULL "
            "AND COALESCE(place_version, 0) != ? AND deleted_at IS NULL "
            "LIMIT ?", (PLACE_FORMAT, limit)).fetchall()
    return [dict(r) for r in rows]


def set_place(item_id, name):
    """Record a resolved label, and mark it current even when nothing was found.

    Marking a failure as current is deliberate: without it, a coordinate OSM has
    no name for would be looked up again on every sweep, forever.
    """
    with _connect() as conn:
        conn.execute("UPDATE items SET place = ?, place_version = ? WHERE id = ?",
                     (name, PLACE_FORMAT, item_id))


def places_pending():
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE lat IS NOT NULL "
            "AND COALESCE(place_version, 0) != ? AND deleted_at IS NULL",
            (PLACE_FORMAT,)).fetchone()
    return row["n"]


def queue_depth(lib):
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM analysis a JOIN items i ON i.id = a.item "
            "WHERE a.library = ? AND a.done_at IS NULL AND a.attempts < ? "
            "AND i.deleted_at IS NULL", (lib, MAX_ATTEMPTS)).fetchone()
    return row["n"]


def search(lib, query, limit=200):
    """Find photographs by what is in them.

    Every term has to match somewhere, which is what makes "beach sunset" narrow
    rather than widen. LIKE over a prepared lowercase column rather than FTS,
    because a library is thousands of rows and one indexed scan is already fast,
    while FTS would need a second table kept in step with this one.
    """
    terms = [t for t in (query or "").lower().split() if t][:8]
    if not terms:
        return []
    where = " AND ".join(["a.search LIKE ?"] * len(terms))
    args = [f"%{t}%" for t in terms]
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_columns('i.')} "
            f"FROM items i JOIN analysis a ON a.item = i.id "
            f"WHERE i.library = ? AND i.deleted_at IS NULL AND {where} "
            f"ORDER BY COALESCE(i.captured_at, datetime(i.stored_at, 'unixepoch')) DESC "
            f"LIMIT ?", (lib, *args, limit)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# people
#
# Clustering is incremental and greedy, on purpose. Each new face is compared
# against one centroid per person and joins the best match above the threshold,
# or starts a new person. That is O(people) per face rather than O(faces), so
# filing the four hundredth photograph costs the same as the fourth.
#
# The tradeoff is that it depends on arrival order and can split one person into
# two clusters when early photographs are unflattering. That is why `merge` is a
# first-class operation rather than an afterthought: a person merging two piles
# by hand is a five second fix, whereas a wrong merge loses information.


def add_faces(lib, item_id, found):
    """File the faces from one photograph and group them. Returns person ids."""
    import faces as face_model

    touched = []
    for face in found:
        vector = face["embedding"]
        with _connect() as conn:
            people = conn.execute(
                "SELECT id, centroid, face_count FROM people WHERE library = ?",
                (lib,)).fetchall()

            best, best_score, runner_up = None, -1.0, -1.0
            for person in people:
                score = face_model.similarity(
                    face_model.from_blob(person["centroid"]), vector)
                if score > best_score:
                    best, best_score, runner_up = person, score, best_score
                elif score > runner_up:
                    runner_up = score

            face_id = secrets.token_urlsafe(12).replace("-", "_")
            # Two conditions, not one. The face has to look like this person,
            # and it has to look like this person *more clearly than like anyone
            # else*. A face that sits between two piles is the face that welds
            # them together, and a new pile is the recoverable answer.
            decisive = runner_up < 0 or (best_score - runner_up) >= face_model.MARGIN
            if best is not None and best_score >= face_model.MATCH and decisive:
                person_id = best["id"]
                centroid = face_model.merged_centroid(
                    face_model.from_blob(best["centroid"]), best["face_count"], vector)
                conn.execute(
                    "UPDATE people SET centroid = ?, face_count = face_count + 1, "
                    "updated_at = ? WHERE id = ?",
                    (face_model.to_blob(centroid), time.time(), person_id))
            else:
                person_id = secrets.token_urlsafe(9).replace("-", "_")
                conn.execute(
                    "INSERT INTO people (id, library, name, centroid, face_count, "
                    "cover, cover_edge, updated_at) VALUES (?, ?, NULL, ?, 1, ?, ?, ?)",
                    (person_id, lib, face_model.to_blob(vector), face_id,
                     face["edge"], time.time()))

            conn.execute(
                "INSERT INTO faces (id, item, library, person, bbox, embedding, crop, "
                "score, edge, found_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (face_id, item_id, lib, person_id, json.dumps(face["bbox"]),
                 face_model.to_blob(vector), face["crop"], face["score"],
                 face["edge"], time.time()))

            # The biggest, clearest face becomes the one shown in the picker.
            conn.execute(
                "UPDATE people SET cover = ?, cover_edge = ? "
                "WHERE id = ? AND ? > cover_edge",
                (face_id, face["edge"], person_id, face["edge"]))
        touched.append(person_id)
    return touched


def people_in(lib, named_first=True):
    """Everybody found in this library, most photographed first.

    Only counts faces in photographs that are not in the trash, so deleting a
    photo takes its people with it rather than leaving a person who appears in
    nothing.
    """
    order = ("p.name IS NULL, live DESC" if named_first else "live DESC")
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT p.id, p.name, p.cover, "
            # DISTINCT on the photograph, not the face: after two piles are
            # merged, one photograph can hold two of the same person's faces,
            # and the gallery labels this number "photos".
            f"       COUNT(DISTINCT f.item) AS live "
            f"FROM people p "
            f"JOIN faces f ON f.person = p.id "
            f"JOIN items i ON i.id = f.item AND i.deleted_at IS NULL "
            f"WHERE p.library = ? "
            f"GROUP BY p.id HAVING live > 0 "
            f"ORDER BY {order}", (lib,)).fetchall()
    return [dict(r) for r in rows]


def face_crop(lib, face_id):
    with _connect() as conn:
        row = conn.execute("SELECT crop FROM faces WHERE library = ? AND id = ?",
                           (lib, face_id)).fetchone()
    return row["crop"] if row else None


def name_person(lib, person_id, name):
    cleaned = (name or "").strip()[:60] or None
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE people SET name = ?, updated_at = ? WHERE library = ? AND id = ?",
            (cleaned, time.time(), lib, person_id))
    return cur.rowcount > 0


def photos_of(lib, person_id, limit=500):
    """Every photograph this person appears in."""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT {_columns('i.')} FROM items i "
            f"JOIN faces f ON f.item = i.id "
            f"WHERE i.library = ? AND f.person = ? AND i.deleted_at IS NULL "
            f"ORDER BY COALESCE(i.captured_at, datetime(i.stored_at, 'unixepoch')) DESC "
            f"LIMIT ?", (lib, person_id, limit)).fetchall()
    return [dict(r) for r in rows]


def merge_people(lib, keep_id, absorb_id):
    """Fold one person into another, for when clustering split somebody in two."""
    import faces as face_model

    if keep_id == absorb_id:
        return False
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, centroid, face_count, name, cover, cover_edge FROM people "
            "WHERE library = ? AND id IN (?, ?)", (lib, keep_id, absorb_id)).fetchall()
        found = {r["id"]: r for r in rows}
        if len(found) != 2:
            return False
        keep, absorb = found[keep_id], found[absorb_id]

        # Weighted mean of the two centroids, so the result reflects how many
        # faces each side actually contributed.
        import numpy as np

        total = keep["face_count"] + absorb["face_count"]
        blended = (face_model.from_blob(keep["centroid"]) * keep["face_count"]
                   + face_model.from_blob(absorb["centroid"]) * absorb["face_count"]) / total
        norm = float(np.linalg.norm(blended))
        if norm:
            blended = blended / norm

        conn.execute("UPDATE faces SET person = ? WHERE library = ? AND person = ?",
                     (keep_id, lib, absorb_id))
        cover, cover_edge = keep["cover"], keep["cover_edge"]
        if absorb["cover_edge"] > (cover_edge or 0):
            cover, cover_edge = absorb["cover"], absorb["cover_edge"]
        conn.execute(
            "UPDATE people SET centroid = ?, face_count = ?, name = COALESCE(name, ?), "
            "cover = ?, cover_edge = ?, updated_at = ? WHERE id = ?",
            (face_model.to_blob(blended), total, absorb["name"], cover, cover_edge,
             time.time(), keep_id))
        conn.execute("DELETE FROM people WHERE library = ? AND id = ?", (lib, absorb_id))
    return True


def forget_faces(lib):
    """Delete every face and person in a library.

    Called when the switch is turned off, because "off" for biometric data has to
    mean erased rather than hidden. Somebody withdrawing consent should not have
    their vectors sitting in a table waiting to be switched back on.
    """
    with _connect() as conn:
        faces_gone = conn.execute("DELETE FROM faces WHERE library = ?", (lib,)).rowcount
        people_gone = conn.execute("DELETE FROM people WHERE library = ?", (lib,)).rowcount
    return {"faces": faces_gone, "people": people_gone}


def faces_on(lib, item_ids):
    """Which people appear in each photograph, for a page of the gallery."""
    if not item_ids:
        return {}
    marks = ",".join("?" * len(item_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT f.item, f.person, f.bbox, p.name FROM faces f "
            f"LEFT JOIN people p ON p.id = f.person "
            f"WHERE f.library = ? AND f.item IN ({marks})", (lib, *item_ids)).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["item"], []).append(
            {"person": r["person"], "name": r["name"],
             "bbox": json.loads(r["bbox"]) if r["bbox"] else None})
    return out


def _payload_path(lib, item_id):
    return LIBRARY_DIR / lib[:2] / lib / f"{item_id}.json"


def usage(lib):
    with _connect() as conn:
        live = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(json_bytes), 0) AS b "
            "FROM items WHERE library = ? AND deleted_at IS NULL", (lib,)).fetchone()
        extra = conn.execute(
            "SELECT COUNT(*) AS trashed, "
            "       COALESCE(SUM(CASE WHEN favourite = 1 THEN 1 ELSE 0 END), 0) AS favourites "
            "FROM items WHERE library = ? AND (deleted_at IS NULL OR favourite = 0)",
            (lib,)).fetchone()
        trashed = conn.execute(
            "SELECT COUNT(*) AS n FROM items WHERE library = ? AND deleted_at IS NOT NULL",
            (lib,)).fetchone()["n"]
    # Deleted photos still occupy the volume, so they count against the quota.
    # Hiding that would let somebody fill the disk with things they believe are
    # already gone.
    return {"items": live["n"], "bytes": live["b"],
            "trashed": trashed, "favourites": extra["favourites"],
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
        # Resolved inline when it is cheap, which after the first photo of a
        # trip it always is: the cache answers from the same 110 metre cell.
        "place": place_for(place.get("lat"), place.get("lon")),
        "place_version": PLACE_FORMAT if place.get("lat") is not None else 0,
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


# The columns the gallery reads. Kept as names so both the plain and the
# table-qualified form below are generated rather than hand-maintained: search
# joins two tables and needs "i.id", and string-replacing into finished SQL is
# how you get a query that breaks when a column name contains another one.
COLUMN_NAMES = ("id", "name", "stored_at", "captured_at", "width", "height", "json_bytes",
                "source_bytes", "lossless", "camera", "lens", "lat", "lon", "place",
                "favourite", "deleted_at")


def _columns(prefix=""):
    listed = ", ".join(f"{prefix}{name}" for name in COLUMN_NAMES)
    return f"{listed}, {prefix}preview IS NOT NULL AS has_preview"


COLUMNS = _columns()

# The three views the gallery offers. Kept here rather than assembled from a
# caller's string so no request can invent its own WHERE clause.
VIEWS = {
    "all": "deleted_at IS NULL",
    "favourites": "deleted_at IS NULL AND favourite = 1",
    "trash": "deleted_at IS NOT NULL",
}


def listing(lib, limit=500, offset=0, view="all"):
    """Everything the grid needs, without the payloads or the thumbnails.

    Thumbnails are a separate request each so the browser caches them
    individually and a listing stays small enough to parse on a phone.
    """
    where = VIEWS.get(view, VIEWS["all"])
    order = ("deleted_at DESC" if view == "trash"
             else "COALESCE(captured_at, datetime(stored_at, 'unixepoch')) DESC, stored_at DESC")
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {COLUMNS} FROM items WHERE library = ? AND {where} "
            f"ORDER BY {order} LIMIT ? OFFSET ?",
            (lib, limit, offset)).fetchall()
    return [dict(r) for r in rows]


def all_ids(lib, view="all"):
    """Every id in a view, for the download-everything path."""
    where = VIEWS.get(view, VIEWS["all"])
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id FROM items WHERE library = ? AND {where} "
            f"ORDER BY COALESCE(captured_at, datetime(stored_at, 'unixepoch')) DESC",
            (lib,)).fetchall()
    return [r["id"] for r in rows]


def set_favourite(lib, item_id, on):
    with _connect() as conn:
        cur = conn.execute("UPDATE items SET favourite = ? WHERE library = ? AND id = ?",
                           (1 if on else 0, lib, item_id))
    return cur.rowcount > 0


def trash(lib, item_id):
    """Soft delete. Recoverable until it is purged."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE items SET deleted_at = ? WHERE library = ? AND id = ? AND deleted_at IS NULL",
            (time.time(), lib, item_id))
    return cur.rowcount > 0


def restore(lib, item_id):
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE items SET deleted_at = NULL WHERE library = ? AND id = ?", (lib, item_id))
    return cur.rowcount > 0


# How long a deleted photograph stays recoverable. Thirty days is what people
# already expect from every photo app they have used.
TRASH_DAYS = int(os.environ.get("JSONCAM_TRASH_DAYS", "30"))


def purge(lib, item_id=None, older_than_days=None):
    """Delete for real: the row and the payload on disk.

    With no item id, sweeps everything in the trash past its expiry, which is
    what keeps deleted photographs from occupying the volume forever.
    """
    with _connect() as conn:
        if item_id:
            rows = conn.execute(
                "SELECT id FROM items WHERE library = ? AND id = ? AND deleted_at IS NOT NULL",
                (lib, item_id)).fetchall()
        else:
            # `is None`, not `or`: older_than_days=0 means "everything, now",
            # which is exactly what Empty Trash asks for, and `or` would read
            # that falsy zero as "unset" and quietly apply the 30 day window
            # instead, so the button would appear to do nothing.
            days = TRASH_DAYS if older_than_days is None else older_than_days
            cutoff = time.time() - days * 86400
            rows = conn.execute(
                "SELECT id FROM items WHERE library = ? AND deleted_at IS NOT NULL "
                "AND deleted_at < ?", (lib, cutoff)).fetchall()
        gone = [r["id"] for r in rows]
        for one in gone:
            conn.execute("DELETE FROM items WHERE library = ? AND id = ?", (lib, one))
    for one in gone:
        try:
            _payload_path(lib, one).unlink(missing_ok=True)
        except OSError:
            pass
    return gone


def get(lib, item_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM items WHERE library = ? AND id = ?",
                           (lib, item_id)).fetchone()
    return dict(row) if row else None


def payload(lib, item_id):
    """The stored container, as bytes."""
    path = _payload_path(lib, item_id)
    return path.read_bytes() if path.exists() else None

