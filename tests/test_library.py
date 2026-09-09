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
