"""Quality metrics.

PSNR is what the loss optimises, but it is a poor stand-in for what a picture
looks like -- it punishes a small global shift far more than it punishes the
smeared texture that low-bitrate codecs actually produce.  MS-SSIM compares
local structure across several scales and tracks human judgement much better,
especially at the low bitrates this codec targets.  We report both.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Scale weights from Wang et al. 2003, "Multiscale structural similarity".
_MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    """The 1D kernel, shaped for a depthwise pass along one axis.

    A 2D Gaussian is separable, so filtering rows then columns gives exactly the
    same answer as one 11x11 convolution for a fraction of the work: 2n
    multiply-adds per pixel instead of n squared.  At the default size that is
    22 against 121, and it is the difference between MS-SSIM costing more than
    the entire codec and costing a fraction of it.
    """
    c = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(c**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.expand(channels, 1, 1, size).contiguous()


def _blur(x, win):
    """Depthwise Gaussian blur, applied separably."""
    C = x.shape[1]
    pad = win.shape[-1] // 2
    x = F.conv2d(x, win, padding=(0, pad), groups=C)
    return F.conv2d(x, win.transpose(-1, -2), padding=(pad, 0), groups=C)


def _ssim_maps(x, y, win, data_range):
    """Return (per-pixel SSIM, per-pixel contrast-structure) for one scale."""
    mu_x = _blur(x, win)
    mu_y = _blur(y, win)
    mx2, my2, mxy = mu_x**2, mu_y**2, mu_x * mu_y
    sx = _blur(x * x, win) - mx2
    sy = _blur(y * y, win) - my2
    sxy = _blur(x * y, win) - mxy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs = (2 * sxy + c2) / (sx + sy + c2)
    ssim = ((2 * mxy + c1) / (mx2 + my2 + c1)) * cs
    return ssim, cs


def ms_ssim(x, y, data_range=1.0, win_size=11, sigma=1.5):
    """Multi-scale SSIM for (B, C, H, W) tensors in [0, data_range].

    Drops to fewer scales on small images: each level halves the side, and a
    level is only meaningful while the side still exceeds the filter support.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    smallest = min(x.shape[-2], x.shape[-1])
    levels = min(len(_MS_WEIGHTS), max(1, int(np.floor(np.log2(smallest / (win_size + 1)))) + 1))
    weights = torch.tensor(_MS_WEIGHTS[:levels], dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()

    win = _gaussian_window(win_size, sigma, x.shape[1], x.device, x.dtype)
    vals = []
    for i in range(levels):
        ssim, cs = _ssim_maps(x, y, win, data_range)
        if i < levels - 1:
            vals.append(torch.relu(cs.mean()))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        else:
            vals.append(torch.relu(ssim.mean()))
    return float(torch.prod(torch.stack(vals) ** weights))


def ms_ssim_db(value):
    """MS-SSIM on a dB scale, the way compression papers usually plot it."""
    v = min(float(value), 1.0 - 1e-9)
    return -10.0 * np.log10(1.0 - v)


def from_images(a, b):
    """MS-SSIM between two PIL images (or HWC uint8 arrays)."""
    xa = torch.from_numpy(np.asarray(a, dtype=np.float32)).permute(2, 0, 1)[None] / 255.0
    xb = torch.from_numpy(np.asarray(b, dtype=np.float32)).permute(2, 0, 1)[None] / 255.0
    return ms_ssim(xa, xb, data_range=1.0)
