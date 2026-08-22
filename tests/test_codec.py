"""Regression tests for the parts that must be exactly right.

Compression is unusually unforgiving: a one-symbol desync between encoder and
decoder does not degrade the image, it destroys everything after that point.
So these tests check for *exactness*, not closeness.
"""

import io
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from jsoncam import codec, rans
from jsoncam.entropy import _to_integer_freqs
from jsoncam.metrics import _blur, _gaussian_window, ms_ssim
from jsoncam.model import JSONCamera, rate_distortion_loss


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = JSONCamera(32, 48)
    for p in m.prior.parameters():
        p.data += 0.05 * torch.randn_like(p)
    return m.eval()


@pytest.fixture(scope="module")
def image():
    yy, xx = np.mgrid[0:157, 0:203]
    a = np.stack([127 + 120 * np.sin(xx / 11.0) * np.cos(yy / 7.0),
                  127 + 120 * np.sin((xx + yy) / 5.0),
                  127 + 120 * np.cos(xx * yy / 700.0)], -1)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def test_integer_freqs_sum_exactly():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(2, 300)
        p = rng.random(n) ** 4
        p /= p.sum()
        f = _to_integer_freqs(p, 1 << 12)
        assert f.sum() == 1 << 12
        assert f.min() >= 1  # every symbol must stay codable


@pytest.mark.parametrize("n", [1, 63, 64, 65, 4097, 100_000])
def test_rans_roundtrip(n):
    rng = np.random.default_rng(n)
    C, S, prec = 6, 25, 12
    freqs = np.stack([
        _to_integer_freqs(
            np.exp(-np.abs(np.arange(S) - S // 2) / (0.5 + 2 * rng.random())) /
            np.exp(-np.abs(np.arange(S) - S // 2) / (0.5 + 2 * rng.random())).sum(),
            1 << prec)
        for _ in range(C)])
    starts, lut = rans.build_luts(freqs)
    chans = rng.integers(0, C, n)
    syms = rng.integers(0, S, n)
    blob, lanes = rans.encode(syms, chans, freqs, starts, prec)
    back = rans.decode(blob, chans, freqs, starts, lut, prec, lanes, n)
    assert np.array_equal(back, syms)


def test_freq_tables_are_valid(model):
    t = codec._tables(model, 12)
    assert (t["freqs"].sum(axis=1) == 1 << 12).all()
    assert (t["range_lo"] <= 0).all() and (t["range_hi"] >= 0).all()


@pytest.mark.parametrize("encoding", ["b85", "b64"])
def test_image_roundtrip_shape_and_json_safety(model, image, encoding, tmp_path):
    doc = codec.encode_image(model, image, encoding=encoding)
    p = tmp_path / "x.json"
    codec.write_json(doc, p)
    out = codec.decode_dict(model, codec.read_json(p))
    assert out.size == image.size  # survives non-multiple-of-16 dimensions
    assert '"' not in doc["payload"]["data"] and "\\" not in doc["payload"]["data"]


def test_latent_symbols_survive_the_coder(model, image):
    """The real invariant: what the decoder reads == what the encoder wrote."""
    import torch.nn.functional as F

    doc = codec.encode_image(model, image)
    x = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    x = x.float().div(255).unsqueeze(0)
    W, H = image.size
    x = F.pad(x, (0, (-W) % 16, 0, (-H) % 16), mode="replicate")
    with torch.no_grad():
        y = codec._tiled_encode(model, x)

    t = codec._tables(model, 12)
    lo, hi = t["range_lo"][:, None, None], t["range_hi"][:, None, None]
    enc = np.clip(model.prior.symbols(y)[0].numpy().astype(np.int64), lo, hi)

    C, lh, lw = enc.shape
    chans = np.repeat(np.arange(C, dtype=np.int64), lh * lw)
    blob = codec._unpack(doc["payload"]["data"], "b85")
    dec = rans.decode(blob, chans, t["freqs"], t["starts"], t["lut"], 12,
                      doc["codec"]["lanes"], doc["codec"]["count"])
    assert np.array_equal(enc, dec.reshape(C, lh, lw) + lo)


def test_tiling_matches_untiled(model, image):
    """Tile seams must not change the latents."""
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    x = x.float().div(255).unsqueeze(0)
    W, H = image.size
    x = F.pad(x, (0, (-W) % 16, 0, (-H) % 16), mode="replicate")
    with torch.no_grad():
        whole = model.encoder(x)
        tiled = codec._tiled_encode(model, x, tile=64, margin=96)
    assert torch.allclose(whole, tiled, atol=1e-4)


def test_wrong_checkpoint_is_refused(model, image):
    doc = codec.encode_image(model, image)
    other = JSONCamera(32, 48).eval()
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        codec.decode_dict(other, doc)


def test_rate_term_is_differentiable():
    m = JSONCamera(16, 24)
    x = torch.rand(2, 3, 64, 64)
    r = rate_distortion_loss(m(x), x, 0.01)
    r["loss"].backward()
    # The entropy model must actually receive gradient -- if it does not, the
    # rate term is decorative and the model is not optimising file size.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.prior.parameters())


def test_separable_blur_matches_the_2d_window():
    """MS-SSIM filters separably for speed; it must still be the same filter.

    A 2D Gaussian is separable, so two 1D passes are mathematically identical to
    one 11x11 convolution.  This pins that down, because a drift here would move
    every quality number the project reports without anything failing.
    """
    import torch.nn.functional as F

    def two_dimensional(x, size=11, sigma=1.5):
        c = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-(c**2) / (2 * sigma**2))
        g = g / g.sum()
        win = (g[:, None] @ g[None, :]).expand(x.shape[1], 1, size, size).contiguous()
        return F.conv2d(x, win, padding=size // 2, groups=x.shape[1])

    torch.manual_seed(0)
    for shape in [(1, 3, 64, 64), (1, 3, 97, 131)]:
        x = torch.rand(*shape)
        separable = _blur(x, _gaussian_window(channels=shape[1]))
        assert torch.allclose(separable, two_dimensional(x), atol=1e-5)


def test_ms_ssim_bounds_and_ordering():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 160)
    slightly_off = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    very_off = torch.rand_like(x)

    assert ms_ssim(x, x) == pytest.approx(1.0, abs=1e-6)
    assert ms_ssim(x, slightly_off) > ms_ssim(x, very_off)
    assert 0.0 <= ms_ssim(x, very_off) <= 1.0
    # Small images must fall back to fewer scales rather than crashing.
    assert 0.0 <= ms_ssim(x[:, :, :48, :48], x[:, :, :48, :48]) <= 1.0


# --------------------------------------------------------------------------
# lossless mode


@pytest.mark.parametrize("size", [(1, 1), (3, 7), (16, 16), (37, 53)])
def test_lossless_is_bit_exact(size):
    """The whole promise of this mode. Any drift here is a total failure.

    Awkward sizes on purpose: the wavefront reconstruction walks anti-diagonals,
    so a single pixel, a thin strip and a non-square image all exercise the
    boundary handling differently.
    """
    from jsoncam import lossless

    rng = np.random.default_rng(0)
    w, h = size
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(a)
    back = lossless.decode_dict(lossless.encode_image(img))
    assert np.array_equal(np.asarray(back), a)


def test_lossless_survives_hard_content():
    """Flat, saturated and noisy content all in one, plus a hard edge.

    Pure noise is the worst case for a predictor and should still round-trip;
    it just will not compress.
    """
    from jsoncam import lossless

    rng = np.random.default_rng(1)
    a = np.zeros((64, 64, 3), np.uint8)
    a[:32, :32] = 255                      # saturated block
    a[:32, 32:] = 0                        # hard edge against black
    a[32:, :32] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)   # noise
    a[32:, 32:] = 127                      # flat mid grey
    back = lossless.decode_dict(lossless.encode_image(Image.fromarray(a)))
    assert np.array_equal(np.asarray(back), a)


def test_ycocg_r_is_reversible():
    from jsoncam.lossless import rgb_to_ycocg, ycocg_to_rgb

    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (23, 29, 3), dtype=np.uint8)
    assert np.array_equal(ycocg_to_rgb(*rgb_to_ycocg(a)), a)


def test_lossless_keeps_name_and_profile():
    from jsoncam import lossless

    a = np.full((8, 8, 3), 90, np.uint8)
    img = Image.fromarray(a)
    img.info["icc_profile"] = b"not-a-real-profile-but-bytes"
    doc = lossless.encode_image(img, name="Holiday Photo.png")
    assert doc["image"]["name"] == "Holiday Photo.png"
    assert lossless.decode_dict(doc).info["icc_profile"] == b"not-a-real-profile-but-bytes"


def test_lossless_beats_png_on_a_real_photograph():
    """Beats PNG on photographs, which is the only claim being made.

    This needs a real photograph and skips without one, deliberately.  Synthetic
    content does not settle the question and would make the test lie in either
    direction: PNG filters rows and runs them through zlib, so it finds *exact
    repeats* and wins by about half on a pure sine pattern, and it still wins by
    a few percent on 1/f noise.  This coder has no match model at all, only local
    prediction and entropy coding.  That is the right trade for a photograph,
    where large areas are smooth and neighbours genuinely predict each other, and
    the wrong one for wallpaper.  Measured on DIV2K: 9 to 20 percent smaller.
    """
    from jsoncam import lossless

    photos = sorted(pathlib.Path("data/val_images").glob("*.png"))[:1]
    if not photos:
        pytest.skip("no real photograph available; run scripts/get_data.py")

    img = Image.open(photos[0]).convert("RGB")
    doc = lossless.encode_image(img)

    assert np.array_equal(np.asarray(lossless.decode_dict(doc)), np.asarray(img))
    assert doc["codec"]["bitstream_bytes"] < photos[0].stat().st_size


# --------------------------------------------------------------------------
# library surface and compressed-domain datasets


def test_public_api_round_trips_both_modes(tmp_path):
    import jsoncam

    a = np.random.default_rng(0).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    src = tmp_path / "x.png"
    Image.fromarray(a).save(src)

    doc = jsoncam.encode_lossless(src)
    assert np.array_equal(np.asarray(jsoncam.decode(doc)), a)   # decode auto-detects

    out = tmp_path / "y.png"
    jsoncam.decode_lossless(doc, out=out)
    assert np.array_equal(np.asarray(Image.open(out).convert("RGB")), a)


def test_latent_shard_round_trip(model, tmp_path):
    """A shard must survive being written, reopened, and read by workers."""
    from jsoncam.dataset import LatentDataset, ShardWriter

    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "s.jcl", model, size=64)
    for i in range(5):
        w.add(Image.fromarray(rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)), label=i % 2)
    assert w.close() == 5

    ck = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, ck)
    ds = LatentDataset(tmp_path / "s.jcl", checkpoint=ck)
    assert len(ds) == 5
    assert ds.classes == [0, 1]
    x, y = ds[3]
    assert x.shape == torch.Size(ds.latent_shape) and x.dtype == torch.float32
    assert y == 1


def test_latent_shard_is_picklable_for_dataloader_workers(model, tmp_path):
    """DataLoader pickles the dataset per worker, and an open file handle is not
    picklable. Reading first, then pickling, is the exact sequence that broke."""
    import pickle

    from jsoncam.dataset import LatentDataset, ShardWriter

    w = ShardWriter(tmp_path / "s.jcl", model, size=64)
    w.add(Image.fromarray(np.full((64, 64, 3), 120, np.uint8)))
    w.close()
    ck = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, ck)

    ds = LatentDataset(tmp_path / "s.jcl", checkpoint=ck)
    _ = ds[0]                                   # opens the handle
    revived = pickle.loads(pickle.dumps(ds))    # must not raise
    assert revived[0][0].shape == ds[0][0].shape


def test_latent_shard_refuses_a_different_model(model, tmp_path):
    """A latent is meaningless to any other checkpoint, so opening must fail
    loudly rather than hand back plausible nonsense."""
    from jsoncam.dataset import LatentDataset, ShardWriter

    w = ShardWriter(tmp_path / "s.jcl", model, size=64)
    w.add(Image.fromarray(np.full((64, 64, 3), 90, np.uint8)))
    w.close()

    other = JSONCamera(32, 48).eval()
    ck = tmp_path / "other.pt"
    torch.save({"model": other.state_dict(), "config": other.config}, ck)
    with pytest.raises(ValueError, match="shard was built with model"):
        LatentDataset(tmp_path / "s.jcl", checkpoint=ck)


def test_latents_are_smaller_than_the_pixels_they_stand_for(model, tmp_path):
    """The whole premise of compressed-domain training."""
    from jsoncam.dataset import LatentDataset, ShardWriter

    w = ShardWriter(tmp_path / "s.jcl", model, size=128)
    w.add(Image.fromarray(np.random.default_rng(1).integers(0, 256, (128, 128, 3), dtype=np.uint8)))
    w.close()
    ck = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, ck)

    ds = LatentDataset(tmp_path / "s.jcl", checkpoint=ck)
    c, h, wd = ds.latent_shape
    assert c * h * wd < 3 * 128 * 128


def test_lossless_keeps_transparency():
    """Dropping a channel is not lossless. A logo whose alpha is flattened comes
    back with an opaque black background, which is worse than refusing."""
    from jsoncam import lossless

    a = np.zeros((40, 48, 4), np.uint8)
    a[8:32, 8:40, :3] = [220, 40, 40]
    a[8:32, 8:40, 3] = 255                       # opaque square on transparency
    img = Image.fromarray(a, "RGBA")

    back = lossless.decode_dict(lossless.encode_image(img))
    assert back.mode == "RGBA"
    assert np.array_equal(np.asarray(back), a)


def test_lossy_admits_it_discarded_alpha(model):
    """The learned codec has three input channels and cannot keep alpha. It must
    say so in the header rather than hand back a silently flattened image."""
    a = np.dstack([np.full((32, 32, 3), 100, np.uint8), np.zeros((32, 32), np.uint8)])
    doc = codec.encode_image(model, Image.fromarray(a, "RGBA"))
    assert doc["image"]["alpha_discarded"] is True

    plain = codec.encode_image(model, Image.fromarray(a[:, :, :3]))
    assert plain["image"]["alpha_discarded"] is False
