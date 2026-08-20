"""Interleaved rANS entropy coder (vectorised with numpy).

rANS packs a stream of symbols into a single big number, spending roughly
-log2(p) bits on each one.  Because the entropy model says most latents are
near zero, most symbols cost a fraction of a bit -- that is where the actual
compression ratio comes from, not from the convnet alone.

Plain rANS is strictly sequential, which is painfully slow in Python.  So we
run `lanes` independent coder states side by side (symbol i goes to lane
i % lanes) and step all of them at once with numpy.  Same output size, ~100x
the speed.

State is 32-bit, renormalisation is in 16-bit words, probabilities use 12 bits.
Those are chosen so at most ONE word moves per symbol per lane, which is what
makes the vectorised step a fixed, branch-free shape.
"""

import numpy as np

STATE_BITS = 32
LOWER = 1 << 16          # renormalisation floor; state always lives in [LOWER, 2**32)
WORD_BITS = 16
WORD_MASK = 0xFFFF


def build_luts(freqs):
    """freqs: (C, S) int64 rows each summing to 2**precision.

    Returns cumulative starts and a slot->symbol lookup table, which is how the
    decoder inverts "where did this number land" in O(1).
    """
    C, S = freqs.shape
    total = int(freqs[0].sum())
    starts = np.zeros((C, S), dtype=np.int64)
    np.cumsum(freqs[:, :-1], axis=1, out=starts[:, 1:])
    lut = np.zeros((C, total), dtype=np.int32)
    for c in range(C):
        lut[c] = np.repeat(np.arange(S, dtype=np.int32), freqs[c])
    return starts, lut


def _lane_count(n):
    if n < 1 << 15:
        return 64
    return 512


def encode(symbols, chans, freqs, starts, precision, lanes=None):
    """symbols/chans: flat int64 arrays of equal length. Returns (bytes, lanes)."""
    total = 1 << precision
    n = int(symbols.size)
    lanes = int(lanes or _lane_count(n))
    T = (n + lanes - 1) // lanes
    pad = T * lanes - n

    syms = np.concatenate([symbols, np.zeros(pad, np.int64)]).reshape(T, lanes)
    chs = np.concatenate([chans, np.zeros(pad, np.int64)]).reshape(T, lanes)

    f_all = freqs[chs, syms]
    s_all = starts[chs, syms]
    xmax_all = ((LOWER >> precision) << WORD_BITS) * f_all

    x = np.full(lanes, LOWER, dtype=np.int64)
    chunks = []
    # Encode backwards: rANS is a stack, so the decoder pops in forward order.
    for t in range(T - 1, -1, -1):
        f, s, xmax = f_all[t], s_all[t], xmax_all[t]
        m = x >= xmax
        if m.any():
            chunks.append((x[m] & WORD_MASK).astype(np.uint16))
            x = np.where(m, x >> WORD_BITS, x)
        x = (x // f) * total + (x % f) + s

    words = np.concatenate(chunks) if chunks else np.zeros(0, np.uint16)
    blob = words.astype("<u2").tobytes() + x.astype("<u4").tobytes()
    return blob, lanes


def decode(blob, chans, freqs, starts, lut, precision, lanes, count):
    """Inverse of encode(). `chans` must be the same array the encoder saw."""
    total = 1 << precision
    mask = total - 1
    T = (count + lanes - 1) // lanes
    pad = T * lanes - count

    state_bytes = lanes * 4
    states = np.frombuffer(blob[-state_bytes:], dtype="<u4").astype(np.int64).copy()
    words = np.frombuffer(blob[:-state_bytes], dtype="<u2").astype(np.int64)

    chs = np.concatenate([chans, np.zeros(pad, np.int64)]).reshape(T, lanes)
    out = np.zeros((T, lanes), dtype=np.int64)

    x = states
    wp = words.size  # exclusive read pointer, walks down
    for t in range(T):
        ch = chs[t]
        slot = x & mask
        sym = lut[ch, slot].astype(np.int64)
        out[t] = sym
        f = freqs[ch, sym]
        s = starts[ch, sym]
        x = f * (x >> precision) + slot - s
        m = x < LOWER
        k = int(m.sum())
        if k:
            lanes_desc = np.nonzero(m)[0][::-1]
            # Encoder wrote a step's words in ascending lane order; reading the
            # stream backwards therefore hands them back highest-lane-first.
            w = words[wp - k : wp][::-1]
            x[lanes_desc] = (x[lanes_desc] << WORD_BITS) | w
            wp -= k

    return out.reshape(-1)[:count]
