"""Tests for soft equivariance head + residual shapes."""

from __future__ import annotations

import torch

from sifess.losses.combined import SIFESSLoss
from sifess.models.equivariance import SoftEquivarianceHead, coherence_weighted_residual
from sifess.models.ssl_model import DINOSifessModel


def test_rho_identity_init():
    head = SoftEquivarianceHead(dim=32)
    z = torch.randn(4, 32)
    angles = torch.zeros(4)
    out = head(z, angles)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-5)


def test_residual_shapes_and_gate():
    z_t = torch.randn(4, 16)
    z_r = torch.randn(4, 16)
    cbar = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = coherence_weighted_residual(z_t, z_r, cbar)
    assert loss.ndim == 0
    # Zero gate → smaller or equal than full gate
    loss_full = coherence_weighted_residual(z_t, z_r, torch.ones(4))
    assert float(loss) <= float(loss_full) + 1e-5


def test_dino_sifess_forward_shapes():
    model = DINOSifessModel(backbone="resnet18", out_dim=128, use_equivariance_head=True)
    x = torch.randn(2, 3, 64, 64)
    logits, z = model.embed_student(x)
    assert logits.shape == (2, 128)
    assert z.shape == (2, 256)
    z2 = model.apply_rho(z, torch.zeros(2))
    assert z2.shape == z.shape


def test_combined_loss_keys():
    crit = SIFESSLoss(out_dim=32, lambda_eq=0.5, lambda_orth=0.01)
    s = [torch.randn(2, 32) for _ in range(3)]
    t = [torch.randn(2, 32) for _ in range(2)]
    z = torch.randn(2, 16)
    out = crit(s, t, z_src=z, z_tgt=z, z_rho=z, cbar=torch.ones(2))
    assert "loss" in out and "l_dino" in out and "l_eq" in out
