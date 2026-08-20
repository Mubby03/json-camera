"""Regression tests for the parts that must be exactly right.

Compression is unusually unforgiving: a one-symbol desync between encoder and
decoder does not degrade the image, it destroys everything after that point.
So these tests check for *exactness*, not closeness.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from jsoncam import codec, rans
from jsoncam.entropy import _to_integer_freqs
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
