"""Tests for the two blocks that make an archive of these files usable.

The codec was already correct before any of this existed; what was missing was
everything around it. A `.json` photograph nothing can draw and nothing can sort
is not a photograph you have kept, it is one you have put somewhere you cannot
get it back from. So these tests are about the promises the container makes to
whoever has to live with a library of them:

  * the capture date, camera and location survive the round trip, because they
    are the only thing an archive can be ordered by;
  * the preview is a real image that opens with no model, no checkpoint and no
    torch, because that is the only way a gallery or a Finder window can show
    anything;
  * the location can be refused, because it is the one field here that says
    where somebody lives;
  * and none of it breaks a decoder that predates it.
"""

import base64
import datetime
import io
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from jsoncam import codec, lossless, meta
from jsoncam.model import JSONCamera


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = JSONCamera(32, 48)
    for p in m.prior.parameters():
        p.data += 0.05 * torch.randn_like(p)
    return m.eval()


def phone_photo(size=(160, 96), orientation=None, gps=True, when="2026:07:04 18:12:09"):
    """A JPEG shaped the way an iPhone shapes one, EXIF and all."""
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    a = np.stack([127 + 120 * np.sin(xx / 9.0),
                  127 + 120 * np.cos(yy / 6.0),
                  127 + 120 * np.sin((xx + yy) / 13.0)], -1)
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))

    exif = img.getexif()
    exif[271], exif[272] = "Apple", "iPhone 15 Pro"
    if orientation:
        exif[274] = orientation
    sub = exif.get_ifd(0x8769)
    sub[36867] = when
    sub[42036] = "iPhone 15 Pro back camera 6.765mm f/1.78"
    sub[33437] = 1.78
    if gps:
        block = exif.get_ifd(0x8825)
        block[1], block[2] = "N", (6.0, 27.0, 14.5)
        block[3], block[4] = "E", (3.0, 24.0, 33.2)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92, exif=exif.tobytes())
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out


# --------------------------------------------------------------------------
# metadata


def test_exif_becomes_json_safe_types():
    """Raw EXIF is rationals and byte strings, and json.dump refuses both."""
    info = meta.extract(phone_photo())
    # The real assertion: this survives serialisation at all.
    json.dumps(info)
    assert info["camera"] == "Apple iPhone 15 Pro"
    assert info["captured_at"] == "2026-07-04T18:12:09"
    assert info["shot"]["aperture"] == pytest.approx(1.78, abs=1e-4)
    assert isinstance(info["gps"]["lat"], float)


def test_capture_date_is_sortable():
    """A library orders itself by this string, so it has to sort lexically."""
    early = meta.extract(phone_photo(when="2026:03:09 08:00:00"))["captured_at"]
    late = meta.extract(phone_photo(when="2026:11:09 08:00:00"))["captured_at"]
    assert early < late
    # ISO order and real chronological order must not disagree, which is exactly
    # what EXIF's own "2026:03:09" format gets wrong: it sorts fine by accident
    # for dates but breaks the moment anything parses it as a date.
    assert early == "2026-03-09T08:00:00"


def test_nonsense_dates_are_dropped_not_stored():
    """A junk date is worse than none: it sorts in among the real ones."""
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif.get_ifd(0x8769)[36867] = "not a date at all"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    reopened = Image.open(io.BytesIO(buf.getvalue()))
    reopened.load()
    assert (meta.extract(reopened) or {}).get("captured_at") is None


def test_gps_can_be_refused_while_the_rest_is_kept():
    """The privacy switch has to be surgical, not all or nothing."""
    with_place = meta.extract(phone_photo(), gps=True)
    without = meta.extract(phone_photo(), gps=False)
    assert with_place["gps"]["lat"] == pytest.approx(6.4540, abs=1e-3)
    assert "gps" not in without
    # Everything that helps sort a library is still there.
    assert without["captured_at"] == with_place["captured_at"]
    assert without["camera"] == with_place["camera"]


def test_southern_and_western_coordinates_go_negative():
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    block = exif.get_ifd(0x8825)
    block[1], block[2] = "S", (33.0, 55.0, 30.0)
    block[3], block[4] = "W", (18.0, 25.0, 12.0)
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    reopened = Image.open(io.BytesIO(buf.getvalue()))
    reopened.load()

    place = meta.extract(reopened)["gps"]
    assert place["lat"] == pytest.approx(-33.925, abs=1e-3)
    assert place["lon"] == pytest.approx(-18.42, abs=1e-2)


def test_images_with_no_exif_get_no_block():
    """A screenshot should not grow an empty dictionary in its header."""
    assert meta.extract(Image.new("RGB", (16, 16))) is None


# --------------------------------------------------------------------------
# the embedded preview


def test_preview_opens_with_no_model_and_no_checkpoint(model):
    """The whole point: a picture out of the file without the decoder.

    If this fails, the format is only viewable by the machine that wrote it, and
    a gallery, a Finder thumbnail and a phone all have nothing to draw.
    """
    doc = codec.encode_image(model, phone_photo(), name="IMG_0001.jpg")
    blob, media = meta.proxy_bytes(doc)

    thumb = Image.open(io.BytesIO(blob))
    thumb.load()
    assert media in ("image/webp", "image/jpeg")
    assert max(thumb.size) == meta.PROXY_SIDE
    # Nothing from jsoncam was needed to get here beyond base64.


def test_preview_is_upright_even_when_the_source_is_not(model):
    """Orientation 6 is a phone held sideways, which is most phone photos.

    The preview is built after the transpose on purpose. Building it before
    would give every sideways photograph a gallery tile lying on its side while
    the decoded picture came out upright.
    """
    sideways = phone_photo(size=(160, 96), orientation=6)
    doc = codec.encode_image(model, sideways, name="rotated.jpg")

    blob, _ = meta.proxy_bytes(doc)
    thumb = Image.open(io.BytesIO(blob))
    upright = ImageOps.exif_transpose(sideways)

    # Portrait source, portrait thumbnail: the two must agree about which way up
    # the picture goes.
    assert (thumb.width < thumb.height) == (upright.width < upright.height)
    assert doc["image"]["width"] == upright.width


def test_preview_costs_are_reported_not_hidden(model):
    """`stats` must not let the thumbnail masquerade as container overhead."""
    doc = codec.encode_image(model, phone_photo(), name="x.jpg")
    stats = codec.stats(doc)
    assert stats["preview_bytes"] == doc["preview"]["bytes"]
    assert stats["preview_pct"] > 0


def test_switches_actually_switch_things_off(model):
    doc = codec.encode_image(model, phone_photo(), preview=False, exif=False)
    assert "preview" not in doc
    assert "meta" not in doc

    only_place_gone = codec.encode_image(model, phone_photo(), gps=False)
    assert "gps" not in only_place_gone["meta"]
    assert only_place_gone["meta"]["captured_at"]


# --------------------------------------------------------------------------
# the blocks must not break the codec


def test_lossless_is_still_bit_exact_with_both_blocks():
    """The new header must not cost the one guarantee this mode makes."""
    source = phone_photo(size=(120, 80))
    doc = lossless.encode_image(source, name="exact.jpg")
    back = lossless.decode_dict(doc)

    assert "meta" in doc and "preview" in doc
    assert np.array_equal(np.asarray(ImageOps.exif_transpose(source)), np.asarray(back))


def test_a_decoder_that_predates_the_blocks_still_reads_the_file(model):
    """Both additions are additive, and this is what that claim means.

    Stripping the two new keys must leave a file the old decoder path handles,
    because files are already in the world that were written without them and
    files written now have to keep opening in whatever read those.
    """
    doc = codec.encode_image(model, phone_photo(), name="new.jpg")
    old_shape = {k: v for k, v in doc.items() if k not in ("meta", "preview")}

    new = codec.decode_dict(model, doc)
    old = codec.decode_dict(model, old_shape)
    assert np.array_equal(np.asarray(new), np.asarray(old))


def test_preview_survives_a_json_round_trip(model, tmp_path):
    """base64 in, base64 out: the block has to be text-safe, not just bytes."""
    doc = codec.encode_image(model, phone_photo(), name="disk.jpg")
    path = tmp_path / "photo.json"
    codec.write_json(doc, path)

    reloaded = codec.read_json(path)
    assert base64.b64decode(reloaded["preview"]["data"]) == base64.b64decode(
        doc["preview"]["data"])
    assert reloaded["meta"] == doc["meta"]


# --------------------------------------------------------------------------
# convert and restore, the way somebody with a folder of photographs uses them


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "jsoncam.cli", *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def checkpoint(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "test.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, path)
    return path


def test_convert_then_restore_keeps_the_library_sortable(tmp_path, checkpoint):
    """The round trip a person actually performs, including the part that matters.

    A restore that hands back the pixels and drops the dates has still ruined
    the library, because a thousand photographs ordered by nothing is not a
    library. So this checks the dates come back, not just the images.
    """
    source = tmp_path / "photos" / "trip"
    source.mkdir(parents=True)
    for i in range(3):
        phone_photo(when=f"2026:07:0{i + 1} 09:0{i}:00").save(
            source / f"IMG_000{i}.jpg", "JPEG", quality=92,
            exif=phone_photo(when=f"2026:07:0{i + 1} 09:0{i}:00").getexif().tobytes())

    library = tmp_path / "library"
    done = run_cli("convert", tmp_path / "photos", "--out", library,
                   "-c", checkpoint, "--quiet")
    assert done.returncode == 0, done.stderr

    # The tree is mirrored, so the folders a person organised by still exist.
    made = sorted(p.name for p in (library / "trip").glob("*.json"))
    assert made == ["IMG_0000.json", "IMG_0001.json", "IMG_0002.json"]

    restored = tmp_path / "back"
    out = run_cli("restore", library, "--out", restored, "-c", checkpoint, "--quiet")
    assert out.returncode == 0, out.stderr

    files = sorted((restored / "trip").glob("*.jpg"))
    assert len(files) == 3
    for index, path in enumerate(files):
        exif = Image.open(path).getexif()
        shot = exif.get_ifd(0x8769)[36867]
        assert shot == f"2026:07:0{index + 1} 09:0{index}:00"
        assert exif.get(272) == "iPhone 15 Pro"
        # Finder and Explorer sort a folder by this, so it has to be the moment
        # the shutter opened and not the moment the file was written.
        expected = datetime.datetime(2026, 7, index + 1, 9, index, 0).timestamp()
        assert path.stat().st_mtime == pytest.approx(expected, abs=1)


def test_convert_is_resumable(tmp_path, checkpoint):
    """A batch of three thousand will be interrupted. It must not start over."""
    source = tmp_path / "in"
    source.mkdir()
    phone_photo().save(source / "one.jpg")

    library = tmp_path / "out"
    first = run_cli("convert", source, "--out", library, "-c", checkpoint, "--quiet")
    second = run_cli("convert", source, "--out", library, "-c", checkpoint, "--quiet")

    assert "1 converted" in first.stdout
    assert "0 converted, 1 already done" in second.stdout


def test_restore_can_rescue_previews_with_no_checkpoint(tmp_path, checkpoint):
    """The exit that works even when the weights are gone.

    The checkpoint is the codebook, so losing it normally means losing the
    pictures. The embedded thumbnails are a lossy way out, and being able to
    reach them without a model is the entire reason they are in the file.
    """
    source = tmp_path / "in"
    source.mkdir()
    phone_photo().save(source / "one.jpg")

    library = tmp_path / "out"
    run_cli("convert", source, "--out", library, "-c", checkpoint, "--quiet")

    sheet = tmp_path / "sheet"
    # Note: no -c at all. Nothing here loads a model.
    out = run_cli("restore", library, "--out", sheet, "--previews", "--quiet")
    assert out.returncode == 0, out.stderr

    rescued = list(sheet.glob("*.jpg"))
    assert len(rescued) == 1
    assert max(Image.open(rescued[0]).size) == meta.PROXY_SIDE


def test_one_bad_file_does_not_end_the_run(tmp_path, checkpoint):
    """Somewhere in a real folder there is a truncated download."""
    source = tmp_path / "in"
    source.mkdir()
    phone_photo().save(source / "good.jpg")
    (source / "broken.jpg").write_bytes(b"this is not a JPEG")

    library = tmp_path / "out"
    done = run_cli("convert", source, "--out", library, "-c", checkpoint)
    assert done.returncode == 0
    assert "1 converted" in done.stdout
    assert "1 failed" in done.stdout
    assert (library / "good.json").exists()


# --------------------------------------------------------------------------
# keys
#
# The server stores only a hash of a key, so it cannot hand a lost one back.
# What it can do is tell somebody their key is mistyped, which is the failure
# that actually happens, and the one whose old symptom was the most alarming
# screen in the app: an empty gallery that looks exactly like deletion.


def test_a_fresh_key_passes_its_own_check():
    import sys

    sys.path.insert(0, "web")
    import library

    key = library.new_key()
    assert key.startswith(library.KEY_PREFIX)
    assert library.inspect_key(key) == "ok"


def test_a_single_wrong_character_is_caught():
    import sys

    sys.path.insert(0, "web")
    import library

    key = library.new_key()
    for position in (5, 12, 20, len(key) - 6):
        broken = list(key)
        broken[position] = "a" if broken[position] != "a" else "b"
        assert library.inspect_key("".join(broken)) == "typo", position


def test_old_keys_are_never_accused_of_being_typos():
    """Keys minted before the checksum existed must keep working.

    A checksum test on one fails by construction, so without the version prefix
    this feature would tell every early user their good key was broken.
    """
    import sys

    sys.path.insert(0, "web")
    import library

    assert library.inspect_key("Kvh6Ye9QVffR0PTaeDq9Hx-ISpRUDE-j") == "legacy"
    # And it still addresses a library, rather than being refused.
    assert len(library.library_id("Kvh6Ye9QVffR0PTaeDq9Hx-ISpRUDE-j")) == 32


def test_the_key_itself_is_never_stored():
    """The reason recovery cannot exist, asserted so it stays true."""
    import sys

    sys.path.insert(0, "web")
    import library

    key = library.new_key()
    handle = library.library_id(key)
    assert key not in handle
    assert handle != key
    # Same key, same library, every time: the handle is a pure function of it.
    assert handle == library.library_id(key)


# --------------------------------------------------------------------------
# views, favourites and the trash


def _lib(tmp_path, monkeypatch):
    """A library module pointed at a temp directory."""
    import importlib
    import sys

    sys.path.insert(0, "web")
    monkeypatch.setenv("JSONCAM_LIBRARY_DIR", str(tmp_path / "lib"))
    import library

    importlib.reload(library)
    return library


def _file(library, lib, name, favourite=False):
    doc = {"format": "json-camera/1", "image": {"name": name, "width": 4, "height": 4},
           "codec": {"bitstream_bytes": 10}, "meta": {}}
    item = library.add(lib, doc, json.dumps(doc).encode())
    if favourite:
        library.set_favourite(lib, item, True)
    return item


def test_delete_is_recoverable_not_final(tmp_path, monkeypatch):
    """The one mistake a photo library must never make permanent on one tap."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "keep.jpg")

    assert library.trash(lib, item) is True
    assert [i["id"] for i in library.listing(lib, view="all")] == []
    assert [i["id"] for i in library.listing(lib, view="trash")] == [item]
    # The payload is still on disk, which is what makes the restore real.
    assert library.payload(lib, item) is not None

    assert library.restore(lib, item) is True
    assert [i["id"] for i in library.listing(lib, view="all")] == [item]


def test_trashed_photos_still_count_against_the_quota(tmp_path, monkeypatch):
    """They occupy the volume, so hiding them would let somebody fill the disk."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "a.jpg")
    library.trash(lib, item)

    usage = library.usage(lib)
    assert usage["items"] == 0
    assert usage["trashed"] == 1


def test_purge_removes_the_row_and_the_bytes(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "gone.jpg")
    library.trash(lib, item)

    assert library.purge(lib, item_id=item) == [item]
    assert library.payload(lib, item) is None
    assert library.listing(lib, view="trash") == []


def test_purge_spares_anything_inside_its_grace_period(tmp_path, monkeypatch):
    """A sweep on every listing must not delete what was trashed a moment ago."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "recent.jpg")
    library.trash(lib, item)

    assert library.purge(lib) == []                      # default 30 day window
    assert library.purge(lib, older_than_days=0) == [item]


def test_a_live_photo_cannot_be_purged(tmp_path, monkeypatch):
    """Purge only ever touches the trash, so it can never skip the grace period."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "live.jpg")

    assert library.purge(lib, item_id=item) == []
    assert library.payload(lib, item) is not None


def test_views_do_not_leak_between_libraries(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    mine = library.library_id(library.new_key())
    yours = library.library_id(library.new_key())
    _file(library, mine, "mine.jpg", favourite=True)
    _file(library, yours, "yours.jpg", favourite=True)

    assert len(library.listing(mine, view="favourites")) == 1
    assert len(library.listing(yours, view="favourites")) == 1
    assert len(library.all_ids(mine)) == 1


def test_place_names_are_cached_on_a_grid(tmp_path, monkeypatch):
    """Two photos from the same spot must not be two network calls."""
    library = _lib(tmp_path, monkeypatch)

    calls = []

    def fake(lat, lon):
        calls.append((lat, lon))
        return {"address": {"suburb": "Lekki", "city": "Lagos", "country": "Nigeria"}}

    # Seed the cache the way a real lookup would, then confirm a nearby
    # coordinate in the same cell never asks again.
    assert library._shorten(fake(6.4540, 3.4092)) == "Lekki, Lagos, Nigeria"
    with library._connect() as conn:
        conn.execute("INSERT OR REPLACE INTO places (cell, name, asked_at) VALUES (?, ?, ?)",
                     (library._cell(6.4540, 3.4092), "Lekki, Lagos, Nigeria", 0))

    before = len(calls)
    assert library.place_for(6.45402, 3.40921) == "Lekki, Lagos, Nigeria"
    assert len(calls) == before          # answered from the cache, no lookup


def test_place_names_drop_repeated_components(tmp_path, monkeypatch):
    """"Lagos, Lagos, Nigeria" reads like a bug, because it is one."""
    library = _lib(tmp_path, monkeypatch)
    assert library._shorten(
        {"address": {"suburb": "Lagos", "city": "Lagos", "country": "Nigeria"}}
    ) == "Lagos, Nigeria"


# --------------------------------------------------------------------------
# the derived layer: captions, search, and the switches that gate them


def test_derived_features_are_off_until_asked(tmp_path, monkeypatch):
    """Both cost something somebody has to agree to.

    The vision pass spends the operator's money per photograph, and faces are
    biometric data about people who never agreed to anything. Neither can be a
    default that nobody saw.
    """
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    assert library.settings(lib) == {"ai": False, "faces": False}

    assert library.set_settings(lib, ai=True) == {"ai": True, "faces": False}
    # Setting one must not silently clear the other.
    assert library.set_settings(lib, faces=True) == {"ai": True, "faces": True}
    assert library.set_settings(lib, ai=False) == {"ai": False, "faces": True}


def test_a_photo_is_described_once_and_never_again(tmp_path, monkeypatch):
    """The whole point of storing it: no work on a refresh, ever."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "a.jpg")
    library.enqueue_analysis(lib, item)

    assert library.queue_depth(lib) == 1
    assert [r["item"] for r in library.next_pending()] == [item]

    library.save_analysis(lib, item, {"caption": "a cat", "tags": ["cat"], "text": None,
                                      "people": 0, "kind": "photo", "model": "test"},
                          "a cat cat")
    assert library.queue_depth(lib) == 0
    assert library.next_pending() == []
    # Queuing it again must not resurrect finished work.
    library.enqueue_analysis(lib, item)
    assert library.next_pending() == []


def test_a_photo_that_keeps_failing_is_eventually_left_alone(tmp_path, monkeypatch):
    """Otherwise one bad file is an infinite loop of paid API calls."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "bad.jpg")
    library.enqueue_analysis(lib, item)

    for _ in range(library.MAX_ATTEMPTS):
        assert library.next_pending(), "gave up too early"
        library.note_attempt(item)
    assert library.next_pending() == []


def test_search_requires_every_term(tmp_path, monkeypatch):
    """"beach sunset" must narrow, not widen."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    beach = _file(library, lib, "beach.jpg")
    lake = _file(library, lib, "lake.jpg")
    for item, blob in ((beach, "a sunset over a sandy beach"), (lake, "a lake at sunset")):
        library.enqueue_analysis(lib, item)
        library.save_analysis(lib, item, {"caption": blob, "tags": [], "text": None,
                                          "people": 0, "kind": "photo", "model": "t"}, blob)

    assert len(library.search(lib, "sunset")) == 2
    assert [i["id"] for i in library.search(lib, "beach sunset")] == [beach]
    assert library.search(lib, "beach lake") == []


def test_search_never_returns_another_library(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    mine = library.library_id(library.new_key())
    yours = library.library_id(library.new_key())
    for lib in (mine, yours):
        item = _file(library, lib, "same.jpg")
        library.enqueue_analysis(lib, item)
        library.save_analysis(lib, item, {"caption": "a dog", "tags": [], "text": None,
                                          "people": 0, "kind": "photo", "model": "t"}, "a dog")

    assert len(library.search(mine, "dog")) == 1
    assert len(library.search(yours, "dog")) == 1


def test_deleted_photos_drop_out_of_search_and_the_queue(tmp_path, monkeypatch):
    """A photo in the trash should not be findable, or paid to describe."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "gone.jpg")
    library.enqueue_analysis(lib, item)
    library.save_analysis(lib, item, {"caption": "a boat", "tags": [], "text": None,
                                      "people": 0, "kind": "photo", "model": "t"}, "a boat")
    assert len(library.search(lib, "boat")) == 1

    library.trash(lib, item)
    assert library.search(lib, "boat") == []

    other = _file(library, lib, "pending.jpg")
    library.enqueue_analysis(lib, other)
    library.trash(lib, other)
    assert library.next_pending() == []
    assert library.queue_depth(lib) == 0


def test_searchable_blob_includes_what_people_actually_type():
    import sys

    sys.path.insert(0, "web")
    import vision

    blob = vision.searchable(
        {"caption": "A Palm Tree", "tags": ["Beach"], "text": "GATE 22", "kind": "photo"},
        {"name": "IMG_1.HEIC", "camera": "Apple iPhone 15 Pro", "place": "Lekki, Lagos",
         "captured_at": "2026-07-04T18:12:09"},
    )
    for term in ("palm", "beach", "gate 22", "iphone", "lekki", "2026-07"):
        assert term in blob, term


def test_captioning_is_skipped_rather_than_crashing_without_credentials(monkeypatch):
    """An upload must survive a missing API key, an outage, and an unpaid bill."""
    import sys

    sys.path.insert(0, "web")
    import vision

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/nonexistent-for-this-test")
    assert vision.available() is False
    assert vision.describe(b"not really an image") is None


# --------------------------------------------------------------------------
# faces
#
# The failure that matters is a wrong merge. Splitting one person into two piles
# is a five second fix with the merge button; welding two people together
# quietly destroys the distinction, and nothing in the UI can tell you it
# happened. So these lean on the merge direction being the safe one.


def _vec(*values):
    """A unit vector in the first few dimensions, for clustering arithmetic."""
    import numpy as np

    v = np.zeros(128, dtype="float32")
    for i, value in enumerate(values):
        v[i] = value
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def _face(vector, edge=80, crop=b"jpegbytes"):
    return {"embedding": vector, "bbox": [0.1, 0.1, 0.2, 0.2], "crop": crop,
            "score": 0.99, "edge": edge}


def test_the_same_face_twice_is_one_person(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    alice = _vec(1, 0, 0)

    for name in ("one.jpg", "two.jpg", "three.jpg"):
        item = _file(library, lib, name)
        library.add_faces(lib, item, [_face(alice)])

    people = library.people_in(lib)
    assert len(people) == 1
    assert people[0]["live"] == 3


def test_two_different_faces_stay_two_people(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "both.jpg")
    library.add_faces(lib, item, [_face(_vec(1, 0, 0)), _face(_vec(0, 1, 0))])

    assert len(library.people_in(lib)) == 2


def test_a_face_between_two_people_starts_its_own_pile(tmp_path, monkeypatch):
    """The chaining defence, and the reason MARGIN exists.

    A face that looks nearly equally like two known people is exactly the face
    that welds them together. Refusing to guess costs one extra pile that
    somebody can merge on purpose.
    """
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    library.add_faces(lib, _file(library, lib, "a.jpg"), [_face(_vec(1, 0, 0))])
    library.add_faces(lib, _file(library, lib, "b.jpg"), [_face(_vec(0, 1, 0))])
    assert len(library.people_in(lib)) == 2

    # Exactly between the two: similarity ~0.707 to each, so it clears the
    # threshold for both and is decisive for neither.
    library.add_faces(lib, _file(library, lib, "c.jpg"), [_face(_vec(1, 1, 0))])
    assert len(library.people_in(lib)) == 3, "an ambiguous face merged two people"


def test_tiny_faces_are_never_stored(tmp_path, monkeypatch):
    """Where the false merges came from, measured. A crowd of 12px faces is noise."""
    import sys

    sys.path.insert(0, "web")
    import faces as face_model
    from PIL import Image

    # find() filters by edge, so a synthetic image of nothing must yield nothing
    # and, more importantly, the constant must stay above the noise floor.
    assert face_model.MIN_EDGE >= 40
    assert face_model.find(Image.new("RGB", (64, 64))) == []


def test_the_threshold_is_stricter_than_opencv_verification_default():
    """Clustering chains errors, so it needs a stricter cut than verification.

    Measured: different people in one photograph reached 0.477, and 0.363 merged
    three of those pairs into one person. If this ever drifts back down, that
    regression returns silently.
    """
    import sys

    sys.path.insert(0, "web")
    import faces as face_model

    assert face_model.MATCH > 0.477, "below the measured different-person ceiling"
    assert face_model.MARGIN > 0


def test_merging_two_piles_keeps_every_photo(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    library.add_faces(lib, _file(library, lib, "a.jpg"), [_face(_vec(1, 0, 0))])
    library.add_faces(lib, _file(library, lib, "b.jpg"), [_face(_vec(0, 1, 0))])
    first, second = (p["id"] for p in library.people_in(lib))

    assert library.merge_people(lib, first, second) is True
    people = library.people_in(lib)
    assert len(people) == 1
    assert people[0]["live"] == 2
    assert len(library.photos_of(lib, first)) == 2
    # And the absorbed person is gone rather than left empty.
    assert library.photos_of(lib, second) == []


def test_merging_keeps_a_name_that_was_already_given(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    library.add_faces(lib, _file(library, lib, "a.jpg"), [_face(_vec(1, 0, 0))])
    library.add_faces(lib, _file(library, lib, "b.jpg"), [_face(_vec(0, 1, 0))])
    unnamed, named = (p["id"] for p in library.people_in(lib))
    library.name_person(lib, named, "Grandpa")

    library.merge_people(lib, unnamed, named)
    assert library.people_in(lib)[0]["name"] == "Grandpa"


def test_merging_refuses_a_person_from_another_library(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    mine = library.library_id(library.new_key())
    yours = library.library_id(library.new_key())
    library.add_faces(lib := mine, _file(library, mine, "a.jpg"), [_face(_vec(1, 0, 0))])
    library.add_faces(yours, _file(library, yours, "b.jpg"), [_face(_vec(0, 1, 0))])
    ours = library.people_in(mine)[0]["id"]
    theirs = library.people_in(yours)[0]["id"]

    assert library.merge_people(mine, ours, theirs) is False
    assert len(library.people_in(yours)) == 1


def test_a_trashed_photo_takes_its_people_with_it(tmp_path, monkeypatch):
    """A person who appears in nothing should not be listed."""
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "only.jpg")
    library.add_faces(lib, item, [_face(_vec(1, 0, 0))])
    assert len(library.people_in(lib)) == 1

    library.trash(lib, item)
    assert library.people_in(lib) == []


def test_turning_faces_off_erases_rather_than_hides(tmp_path, monkeypatch):
    """Consent withdrawn has to mean the vectors are gone.

    Biometric data left sitting in a table waiting to be switched back on is not
    off, and for most of these faces the person in them never agreed to anything.
    """
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    library.set_settings(lib, faces=True)
    item = _file(library, lib, "a.jpg")
    library.add_faces(lib, item, [_face(_vec(1, 0, 0)), _face(_vec(0, 1, 0))])

    erased = library.forget_faces(lib)
    assert erased == {"faces": 2, "people": 2}
    assert library.people_in(lib) == []
    assert library.faces_on(lib, [item]) == {}


def test_names_are_trimmed_and_capped(tmp_path, monkeypatch):
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    library.add_faces(lib, _file(library, lib, "a.jpg"), [_face(_vec(1, 0, 0))])
    person = library.people_in(lib)[0]["id"]

    library.name_person(lib, person, "   Grandpa   ")
    assert library.people_in(lib)[0]["name"] == "Grandpa"
    library.name_person(lib, person, "x" * 200)
    assert len(library.people_in(lib)[0]["name"]) == 60
    # Blanking a name puts the person back to unnamed rather than storing "".
    library.name_person(lib, person, "  ")
    assert library.people_in(lib)[0]["name"] is None


def test_a_person_is_counted_in_photos_not_faces(tmp_path, monkeypatch):
    """After a merge one photograph can hold two of the same person's faces.

    The gallery labels this number "photos", so counting faces would tell
    somebody they have six photographs of their grandfather when they have
    three.
    """
    library = _lib(tmp_path, monkeypatch)
    lib = library.library_id(library.new_key())
    item = _file(library, lib, "both.jpg")
    library.add_faces(lib, item, [_face(_vec(1, 0, 0)), _face(_vec(0, 1, 0))])
    first, second = (p["id"] for p in library.people_in(lib))

    library.merge_people(lib, first, second)
    assert library.people_in(lib)[0]["live"] == 1, "counted faces instead of photos"


def test_effort_is_dropped_for_a_model_that_rejects_it(monkeypatch):
    """Haiku 4.5 rejects output_config.effort, and it is the cheap model.

    Sending it anyway failed every caption silently on exactly the model
    somebody picks when they are watching their bill.
    """
    import sys
    import types

    sys.path.insert(0, "web")
    import vision

    calls = []

    class FakeBadRequest(Exception):
        pass

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs["output_config"])
            if "effort" in kwargs["output_config"]:
                raise FakeBadRequest("output_config.effort is not supported for this model")
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(
                    type="text",
                    text='{"caption":"a cat","tags":["cat"],"text":"",'
                         '"people":0,"kind":"photo"}')],
            )

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=FakeMessages())
    fake.BadRequestError = FakeBadRequest
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(vision, "available", lambda: True)
    monkeypatch.setattr(vision, "_effort_ok", {})

    out = vision.describe(b"pretend-webp")
    assert out is not None, "gave up instead of retrying without effort"
    assert out["caption"] == "a cat"
    assert len(calls) == 2 and "effort" in calls[0] and "effort" not in calls[1]

    # And it remembers, so the next photo costs one request rather than two.
    calls.clear()
    vision.describe(b"pretend-webp")
    assert len(calls) == 1 and "effort" not in calls[0]
