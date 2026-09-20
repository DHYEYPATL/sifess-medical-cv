"""Tests for isophote-frame geometric transforms."""

from __future__ import annotations

import torch

from sifess.geometry.frame_transforms import (
    FrameGroupElement,
    apply_frame_transform,
    sample_frame_group_element,
)


def test_identity_near_noop():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    theta = torch.zeros(2)
    g = FrameGroupElement(angle=torch.zeros(2), reflect=torch.zeros(2, dtype=torch.bool))
    y = apply_frame_transform(x, theta, g)
    # Border padding + bilinear may introduce tiny error
    assert (y - x).abs().mean() < 0.05


def test_rotation_changes_image():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    # Strong spatial structure
    x[:, :, :, :16] = 1.0
    theta = torch.zeros(2)
    g = FrameGroupElement(
        angle=torch.tensor([torch.pi / 2, torch.pi / 2]),
        reflect=torch.zeros(2, dtype=torch.bool),
    )
    y = apply_frame_transform(x, theta, g)
    assert (y - x).abs().mean() > 0.1


def test_sample_shapes():
    g = sample_frame_group_element(4, device=torch.device("cpu"))
    assert g.angle.shape == (4,)
    assert g.reflect.shape == (4,)
