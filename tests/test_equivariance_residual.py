"""Tests for soft equivariance head + residual shapes."""

from __future__ import annotations

import torch

from sifess.geometry.frame_transforms import apply_frame_transform, sample_frame_group_element
from sifess.geometry.structure_tensor import compute_structure_tensor, pooled_frame_angle
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
    # Vector-norm residual must be O(1), not diluted by feature dim
    assert float(loss_full) > 0.1


def test_residual_uses_vector_norm_not_mean():
    """Regression: .mean(dim=-1) diluted L_eq by D and logged as 0.0000."""
    torch.manual_seed(0)
    z_t = torch.randn(2, 256)
    z_r = torch.randn(2, 256)
    cbar = torch.ones(2)
    loss = float(coherence_weighted_residual(z_t, z_r, cbar))
    mean_diluted = float(((z_t - z_r) ** 2).mean())
    sum_norm = float(((z_t - z_r) ** 2).sum(dim=-1).mean())
    assert abs(loss - sum_norm) < 1e-5
    assert loss > mean_diluted * 100  # ~D=256 larger than the buggy mean form


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


def test_sifess_full_one_step_leq_nonzero():
    """One sifess_full train step must yield meaningfully nonzero L_eq."""
    torch.manual_seed(42)
    model = DINOSifessModel(backbone="resnet18", out_dim=128, use_equivariance_head=True)
    model.train()
    crit = SIFESSLoss(out_dim=128, lambda_eq=0.5, lambda_orth=0.01)

    g0 = torch.randn(4, 3, 64, 64)
    # Fake multi-crop student/teacher logits path
    crops = [g0, g0 + 0.01 * torch.randn_like(g0), *[torch.randn(4, 3, 48, 48) for _ in range(4)]]
    out = model.forward_multicrop(crops, crops[:2])

    st = compute_structure_tensor(g0, sigma_rho=1.5)
    frame_theta = pooled_frame_angle(st)
    g_el = sample_frame_group_element(g0.shape[0], device=g0.device)
    g0_warp = apply_frame_transform(g0, frame_theta, g_el)

    _, z_src = model.embed_student(g0)
    _, z_tgt = model.embed_student(g0_warp)
    z_rho = model.apply_rho(z_src, g_el.angle, g_el.reflect)

    losses = crit(
        out["student_logits"],
        out["teacher_logits"],
        z_src=z_src,
        z_tgt=z_tgt,
        z_rho=z_rho,
        cbar=st.Cbar,
    )
    assert "l_eq" in losses
    l_eq = float(losses["l_eq"].detach())
    assert l_eq > 0.01, f"expected L_eq > 0.01 for sifess_full, got {l_eq}"

    # vanilla path: lambda_eq=0 → no l_eq key / zero contribution
    crit_v = SIFESSLoss(out_dim=128, lambda_eq=0.0, lambda_orth=0.0)
    losses_v = crit_v(out["student_logits"], out["teacher_logits"])
    assert "l_eq" not in losses_v
    assert float(losses_v["loss"]) == float(losses_v["l_dino"])
