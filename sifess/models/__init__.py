"""Models: DINO student-teacher backbone + soft equivariance head."""

from sifess.models.ssl_model import DINOSifessModel, build_backbone
from sifess.models.equivariance import SoftEquivarianceHead

__all__ = ["DINOSifessModel", "build_backbone", "SoftEquivarianceHead"]
