"""Unit tests for structure tensor — synthetic step-edge."""

from __future__ import annotations

import torch

from sifess.geometry.structure_tensor import compute_structure_tensor, pooled_frame_angle


def _vertical_step_edge(h=64, w=64, batch=2) -> torch.Tensor:
    """Bright left / dark right → strong horizontal gradient → e1 ~ (1,0)."""
    img = torch.zeros(batch, 1, h, w)
    img[:, :, :, : w // 2] = 1.0
    return img


def _horizontal_step_edge(h=64, w=64, batch=2) -> torch.Tensor:
    img = torch.zeros(batch, 1, h, w)
    img[:, :, : h // 2, :] = 1.0
    return img


def test_step_edge_coherence_high():
    x = _vertical_step_edge()
    st = compute_structure_tensor(x, sigma_rho=1.5)
    # Near the edge, C should be high; mean over image still clearly > isotropic noise
    assert st.C.shape == (2, 1, 64, 64)
    assert float(st.Cbar.mean()) > 0.2
    assert torch.all(st.Cbar == st.Cbar.detach())  # stopgrad buffer values


def test_vertical_edge_orientation_near_horizontal():
    x = _vertical_step_edge()
    st = compute_structure_tensor(x, sigma_rho=1.5)
    # e1 x-component should dominate near the vertical step (cols ~30-33)
    rows = slice(20, 44)
    cols = slice(30, 34)
    e1x = st.e1[:, 0, rows, cols].abs().mean()
    e1y = st.e1[:, 1, rows, cols].abs().mean()
    assert e1x > e1y


def test_horizontal_edge_orientation_near_vertical():
    x = _horizontal_step_edge()
    st = compute_structure_tensor(x, sigma_rho=1.5)
    rows = slice(30, 34)
    cols = slice(20, 44)
    e1x = st.e1[:, 0, rows, cols].abs().mean()
    e1y = st.e1[:, 1, rows, cols].abs().mean()
    assert e1y > e1x


def test_eigenvectors_orthonormal():
    x = _vertical_step_edge()
    st = compute_structure_tensor(x, sigma_rho=1.5)
    # e1 · e2 ≈ 0, ||e1||≈1, ||e2||≈1 (all pixels after degenerate fix)
    dot = (st.e1 * st.e2).sum(dim=1)
    assert float(dot.abs().max()) < 1e-4
    n1 = st.e1.norm(dim=1)
    n2 = st.e2.norm(dim=1)
    assert torch.allclose(n1, torch.ones_like(n1), atol=1e-4)
    assert torch.allclose(n2, torch.ones_like(n2), atol=1e-4)


def test_isotropic_low_coherence():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    st = compute_structure_tensor(x, sigma_rho=1.5)
    edge = compute_structure_tensor(_vertical_step_edge(), sigma_rho=1.5)
    assert float(st.Cbar.mean()) < float(edge.Cbar.mean())


def test_pooled_angle_finite():
    x = _vertical_step_edge()
    st = compute_structure_tensor(x, sigma_rho=1.5)
    ang = pooled_frame_angle(st)
    assert ang.shape == (2,)
    assert torch.isfinite(ang).all()


def test_rgb_input_accepted():
    x = _vertical_step_edge().repeat(1, 3, 1, 1)
    st = compute_structure_tensor(x, sigma_rho=1.5)
    assert st.e1.shape[0] == 2
