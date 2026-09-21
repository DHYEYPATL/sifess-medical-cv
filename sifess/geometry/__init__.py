"""Geometry: structure tensor eigenframes and isophote-frame transforms."""

from sifess.geometry.structure_tensor import StructureTensorResult, compute_structure_tensor
from sifess.geometry.frame_transforms import (
    apply_frame_transform,
    mean_abs_image_diff,
    rotate_features_so2,
    sample_frame_group_element,
)

__all__ = [
    "StructureTensorResult",
    "compute_structure_tensor",
    "apply_frame_transform",
    "mean_abs_image_diff",
    "rotate_features_so2",
    "sample_frame_group_element",
]
