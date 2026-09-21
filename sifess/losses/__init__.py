"""Losses: DINO + coherence-weighted soft equivariance (+ plane / orth)."""

from sifess.losses.dino import DINOLoss
from sifess.losses.equivariance import SoftEquivarianceLoss, PlaneEnergyLoss
from sifess.losses.combined import SIFESSLoss

__all__ = ["DINOLoss", "SoftEquivarianceLoss", "PlaneEnergyLoss", "SIFESSLoss"]
