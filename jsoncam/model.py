"""Analysis/synthesis networks.

The encoder is a stack of stride-2 convolutions.  Each one halves the height
and width, so four of them turn a HxW image into an (H/16)x(W/16) grid of
feature vectors.  That grid *is* the compressed representation -- the numbers
that end up in the JSON file.  The decoder mirrors it with transposed
convolutions to climb back up to full resolution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _NonNegative(nn.Module):
    """Reparametrise a tensor so it stays >= a floor during training.

    We store sqrt(x - min) and square it back, which keeps gradients sane near
    the boundary instead of the hard clamp fighting the optimiser.
    """

    def __init__(self, minimum=0.0, eps=2**-18):
        super().__init__()
        self.minimum = float(minimum)
        self.eps = float(eps)
        self.bound = (self.minimum + self.eps**2) ** 0.5

    def init(self, x):
        return torch.sqrt(torch.clamp(x + self.eps**2, min=self.eps**2))

    def forward(self, x):
        return torch.clamp(x, min=self.bound) ** 2 - self.eps**2


class GDN(nn.Module):
    """Generalised Divisive Normalisation (Balle et al.).

    Divides each channel by a learned function of the local energy across
    channels.  It is the standard activation for compression nets because it
    decorrelates and gaussianises the features far better than ReLU, which
    directly means fewer bits out of the entropy coder.
    """

    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.beta_rp = _NonNegative(minimum=1e-6)
        self.gamma_rp = _NonNegative(minimum=0.0)

        beta = torch.ones(channels)
        gamma = 0.1 * torch.eye(channels)
        self.beta = nn.Parameter(self.beta_rp.init(beta))
        self.gamma = nn.Parameter(self.gamma_rp.init(gamma))

    def forward(self, x):
        beta = self.beta_rp(self.beta)
        gamma = self.gamma_rp(self.gamma)
        norm = F.conv2d(x**2, gamma.unsqueeze(-1).unsqueeze(-1), beta)
        norm = torch.sqrt(torch.clamp(norm, min=1e-10))
        return x * norm if self.inverse else x / norm


class ResidualBlock(nn.Module):
    """Cheap capacity boost that keeps the spatial size fixed."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    """Image -> latent grid.  Four stride-2 convs = 16x downsample."""

    def __init__(self, hidden=128, latent=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 5, stride=2, padding=2),
            GDN(hidden),
            nn.Conv2d(hidden, hidden, 5, stride=2, padding=2),
            GDN(hidden),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, hidden, 5, stride=2, padding=2),
            GDN(hidden),
            nn.Conv2d(hidden, latent, 5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Latent grid -> image.  Mirrors the encoder exactly."""

    def __init__(self, hidden=128, latent=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, hidden, 5, stride=2, padding=2, output_padding=1),
            GDN(hidden, inverse=True),
            nn.ConvTranspose2d(hidden, hidden, 5, stride=2, padding=2, output_padding=1),
            GDN(hidden, inverse=True),
            ResidualBlock(hidden),
            nn.ConvTranspose2d(hidden, hidden, 5, stride=2, padding=2, output_padding=1),
            GDN(hidden, inverse=True),
            nn.ConvTranspose2d(hidden, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, y):
        return self.net(y)


class JSONCamera(nn.Module):
    """Encoder + learned entropy model + decoder, trained end to end.

    The three parts are trained against a single loss that literally adds up
    "how wrong is the picture" and "how many bits did that cost", so the
    network is optimising the file size directly rather than hoping a generic
    compressor does well on its output afterwards.
    """

    def __init__(self, hidden=128, latent=192):
        super().__init__()
        from .entropy import FactorizedPrior

        self.hidden, self.latent = int(hidden), int(latent)
        self.encoder = Encoder(hidden, latent)
        self.decoder = Decoder(hidden, latent)
        self.prior = FactorizedPrior(latent)

    def forward(self, x):
        y = self.encoder(x)
        y_hat = self.prior.quantize(y, self.training)
        p = self.prior.likelihood(y_hat)
        x_hat = self.decoder(y_hat)
        return {"x_hat": x_hat, "likelihoods": p, "y": y}

    @property
    def config(self):
        return {"hidden": self.hidden, "latent": self.latent}


def rate_distortion_loss(out, target, lmbda):
    """Loss = lambda * distortion + rate.

    Rate is in bits per pixel, straight out of the entropy model.  Distortion
    is MSE in 0-255 units so lambda has a consistent meaning across models.
    Raising lambda buys quality with bytes; lowering it does the reverse.
    """
    n_pixels = target.shape[0] * target.shape[2] * target.shape[3]
    bpp = torch.log(out["likelihoods"]).sum() / (-np.log(2) * n_pixels)
    mse = F.mse_loss(out["x_hat"], target)
    return {"loss": lmbda * 255.0**2 * mse + bpp, "bpp": bpp, "mse": mse}
