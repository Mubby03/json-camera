"""Learned entropy model (factorised prior, Balle et al. 2018).

Compression is two jobs.  The convnet does the first: throw away detail the
eye will not miss.  This file does the second: *count the bits honestly*.

The network learns, per latent channel, the cumulative distribution of the
values that channel tends to produce.  Once you know a value's probability you
know its cost in bits (-log2 p), so during training we can put real bits in the
loss and let the optimiser trade quality against file size.  The same learned
CDF is then handed to the range coder at encode time, which is what turns
"common value" into "almost no bytes on disk".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedPrior(nn.Module):
    """Monotonic CDF per channel, built from a tiny per-channel MLP.

    Monotonicity is guaranteed structurally: every weight matrix is passed
    through softplus so it is non-negative, and the output goes through a
    sigmoid.  A non-negative-weight network is non-decreasing in its input, so
    the result is always a valid CDF -- no way for training to produce a
    negative probability.
    """

    def __init__(self, channels, filters=(3, 3, 3), init_scale=10.0, tail_mass=1e-9):
        super().__init__()
        self.channels = int(channels)
        self.tail_mass = float(tail_mass)
        dims = (1,) + tuple(filters) + (1,)
        scale = init_scale ** (1 / (len(dims) - 1))

        self._H, self._b, self._a = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(len(dims) - 1):
            din, dout = dims[i], dims[i + 1]
            # softplus^-1(scale/din) so the initial network is roughly linear
            init = np.log(np.expm1(1.0 / scale / dout))
            self._H.append(nn.Parameter(torch.full((self.channels, dout, din), float(init))))
            self._b.append(nn.Parameter(torch.empty(self.channels, dout, 1).uniform_(-0.5, 0.5)))
            if i < len(dims) - 2:
                self._a.append(nn.Parameter(torch.zeros(self.channels, dout, 1)))

        # Per-channel quantisation offset, learned. Lets a channel centre its
        # rounding grid where its mass actually is instead of always on 0.
        self.register_parameter("offset", nn.Parameter(torch.zeros(self.channels)))
        # Symbol range discovered at build_tables() time.
        self.register_buffer("_range_lo", torch.zeros(self.channels, dtype=torch.long))
        self.register_buffer("_range_hi", torch.zeros(self.channels, dtype=torch.long))

    def _logits(self, x):
        """x: (C, 1, N) -> logits of the CDF, same shape."""
        for i in range(len(self._H)):
            H = F.softplus(self._H[i])
            x = torch.matmul(H, x) + self._b[i]
            if i < len(self._a):
                x = x + torch.tanh(self._a[i]) * torch.tanh(x)
        return x

    def _cdf(self, x):
        return torch.sigmoid(self._logits(x))

    def likelihood(self, y_hat):
        """P(y_hat) for each element, via CDF(v + 0.5) - CDF(v - 0.5).

        y_hat is (B, C, H, W).  Returns the same shape.
        """
        B, C, H, W = y_hat.shape
        v = y_hat.permute(1, 0, 2, 3).reshape(C, 1, -1)
        # Evaluate logits at both edges of the quantisation bin.  Using the
        # logit difference through sigmoid directly loses precision in the
        # tails, so use the numerically stable sign trick from the reference
        # implementation.
        lo = self._logits(v - 0.5)
        hi = self._logits(v + 0.5)
        sign = -torch.sign(lo + hi).detach()
        p = torch.abs(torch.sigmoid(sign * hi) - torch.sigmoid(sign * lo))
        p = p.reshape(C, B, H, W).permute(1, 0, 2, 3)
        return torch.clamp(p, min=1e-9)

    def quantize(self, y, training):
        """Round to integers, but keep gradients flowing.

        During training we add uniform noise instead of rounding -- that is a
        smooth stand-in for quantisation and is what makes the rate term
        differentiable.  At eval we round for real.
        """
        off = self.offset.view(1, -1, 1, 1)
        if training:
            noise = torch.empty_like(y).uniform_(-0.5, 0.5)
            return y + noise
        return torch.round(y - off) + off

    def symbols(self, y):
        """Integer symbols for the range coder (offset removed)."""
        off = self.offset.view(1, -1, 1, 1)
        return torch.round(y - off).to(torch.long)

    def dequantize(self, s):
        off = self.offset.view(1, -1, 1, 1)
        return s.to(self.offset.dtype) + off

    # ---- discrete tables for the actual range coder -------------------------

    @torch.no_grad()
    def build_tables(self, precision=12, max_symbol=255):
        """Freeze the continuous CDF into integer frequency tables.

        The range coder needs exact integer probabilities that sum to 2**precision
        and are identical on both sides, otherwise encode and decode desync.
        """
        dev = self.offset.device
        # Find, per channel, the symbol range holding all but tail_mass of the mass.
        probe = torch.arange(-max_symbol, max_symbol + 1, device=dev, dtype=torch.float32)
        v = probe.view(1, 1, -1).expand(self.channels, 1, -1)
        lo_l = self._logits(v - 0.5)
        hi_l = self._logits(v + 0.5)
        sign = -torch.sign(lo_l + hi_l)
        pmf = torch.abs(torch.sigmoid(sign * hi_l) - torch.sigmoid(sign * lo_l)).squeeze(1)

        keep = pmf > self.tail_mass
        keep[:, max_symbol] = True  # always keep 0
        idx = torch.arange(pmf.shape[1], device=dev)
        lo_i = torch.where(keep, idx, torch.full_like(idx, 10**9)).min(dim=1).values
        hi_i = torch.where(keep, idx, torch.full_like(idx, -1)).max(dim=1).values

        self._range_lo = (lo_i - max_symbol).to(torch.long)
        self._range_hi = (hi_i - max_symbol).to(torch.long)

        width = int((self._range_hi - self._range_lo).max().item()) + 1
        total = 1 << precision
        freqs = np.zeros((self.channels, width), dtype=np.int64)
        pmf_np = pmf.cpu().numpy()
        lo_np = self._range_lo.cpu().numpy()
        hi_np = self._range_hi.cpu().numpy()

        for c in range(self.channels):
            a, b = lo_np[c] + max_symbol, hi_np[c] + max_symbol
            p = pmf_np[c, a : b + 1].astype(np.float64)
            p = np.maximum(p, 1e-12)
            p /= p.sum()
            f = _to_integer_freqs(p, total)
            freqs[c, : f.shape[0]] = f

        return {
            "freqs": freqs,
            "range_lo": lo_np.copy(),
            "range_hi": hi_np.copy(),
            "precision": precision,
        }


def _to_integer_freqs(p, total):
    """Turn float probabilities into positive integers summing exactly to total.

    Every symbol gets at least 1 so nothing becomes uncodable; the rounding
    error is then repaid by taking from whichever symbols can most afford it.
    """
    n = p.shape[0]
    assert total >= n, "precision too small for this many symbols"
    f = np.maximum(1, np.floor(p * total).astype(np.int64))
    delta = total - int(f.sum())
    if delta != 0:
        # Rank by how much each adjustment hurts, cheapest first.
        if delta > 0:
            order = np.argsort(-(p * total - f))
            for i in range(delta):
                f[order[i % n]] += 1
        else:
            order = np.argsort(f - p * total)[::-1]
            i = 0
            while delta < 0:
                j = order[i % n]
                if f[j] > 1:
                    f[j] -= 1
                    delta += 1
                i += 1
    assert int(f.sum()) == total and f.min() >= 1
    return f
