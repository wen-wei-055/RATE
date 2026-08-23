"""Loss for a mixture-of-Gaussians prediction of log10 PGA."""

from __future__ import annotations

import numpy as np
import torch

from .config import WeightedLoss


def mixture_density_loss(y_true, y_pred, eps=1e-6):
    """Negative log likelihood of ``y_true`` under the predicted mixture.

    ``y_pred`` is ``(samples, components, 1 + 2d)``: mixture weight, then the
    mean and standard deviation of each of the ``d`` dimensions.  ``y_true`` is
    ``(samples, d, 1)``.
    """
    alpha = y_pred[:, :, 0]
    d = (y_pred.shape[-1] - 1) // 2
    density = torch.ones(y_pred.shape[0], y_pred.shape[1], device=y_pred.device)
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = torch.clamp_min(y_pred[:, :, j + 1 + d], eps)
        density = density * 1 / (np.sqrt(2 * np.pi).astype("float32") * sigma)
        density = density * torch.exp(-((y_true[:, j] - mu) ** 2) / (2 * sigma**2))
    return -torch.log(torch.sum(density * alpha, axis=-1) + eps)


def band_weights(y_true, weighted_loss: WeightedLoss, device) -> torch.Tensor:
    """One weight per sample, from the intensity band its target falls in.

    Strong shaking is rare and is what an early warning system exists for, so
    those samples can be made to count for more.
    """
    thresholds = torch.tensor(weighted_loss.thresholds, device=device)
    weights = torch.tensor(weighted_loss.weights, device=device, dtype=torch.float32)
    band = torch.sum((y_true.reshape(-1, 1) >= thresholds).int(), dim=1)
    return weights[band]


def pga_loss(y_true, y_pred, weighted_loss: WeightedLoss | None = None):
    """Mean loss over every station of every event in a batch."""
    d = (y_pred.shape[-1] - 1) // 2
    y_true = y_true.contiguous().view(-1, d, 1)
    y_pred = y_pred.contiguous().view(-1, y_pred.shape[-2], y_pred.shape[-1])
    loss = mixture_density_loss(y_true, y_pred)

    if weighted_loss is not None and weighted_loss.enabled:
        weights = band_weights(y_true, weighted_loss, y_pred.device)
        return torch.sum(loss * weights) / torch.sum(weights)
    return torch.mean(loss)
