"""Tests for equivariance maps + residual (incl. sustained L_eq signal)."""

from __future__ import annotations

import torch

from sifess.geometry.frame_transforms import (
    apply_frame_transform,
    mean_abs_image_diff,
    rotate_features_so2,
    sample_frame_group_element,
)
from sifess.geometry.structure_tensor import compute_structure_tensor, pooled_frame_angle
from sifess.losses.combined import SIFESSLoss
from sifess.models.equivariance import (
    FixedSO2Action,
    SoftEquivarianceHead,
    coherence_weighted_residual,
)
from sifess.models.ssl_model import DINOSifessModel


def test_rho_film_identity_init():
    head = SoftEquivarianceHead(dim=32)
    z = torch.randn(4, 32)
    angles = torch.zeros(4)
    out = head(z, angles)
    assert out.shape == z.shape
    assert torch.allclose(out, z, atol=1e-5)


def test_fixed_so2_rotates_first_two_dims():
    z = torch.tensor([[1.0, 0.0, 0.5, -0.2], [0.0, 1.0, 0.1, 0.0]])
    angles = torch.tensor([torch.pi / 2, torch.pi / 2])
    out = rotate_features_so2(z, angles)
    assert torch.allclose(out[0, :2], torch.tensor([0.0, 1.0]), atol=1e-5)
    assert torch.allclose(out[1, :2], torch.tensor([-1.0, 0.0]), atol=1e-5)
    assert torch.allclose(out[:, 2:], z[:, 2:], atol=1e-5)
    assert not torch.allclose(out, z)


def test_fixed_so2_action_module():
    rho = FixedSO2Action(dim=8)
    z = torch.randn(3, 8)
    angles = torch.ones(3) * 0.7
    out = rho(z, angles)
    assert out.shape == z.shape
    assert not torch.allclose(out[:, :2], z[:, :2])


def test_residual_shapes_and_gate():
    z_t = torch.randn(4, 16)
    z_r = torch.randn(4, 16)
    cbar = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = coherence_weighted_residual(z_t, z_r, cbar)
    assert loss.ndim == 0
    loss_full = coherence_weighted_residual(z_t, z_r, torch.ones(4))
    assert float(loss) <= float(loss_full) + 1e-5
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
    assert loss > mean_diluted * 100


def test_stopgrad_target_blocks_tgt_grad():
    z_src = torch.randn(2, 8, requires_grad=True)
    z_tgt = torch.randn(2, 8, requires_grad=True)
    z_rho = z_src
    loss = coherence_weighted_residual(z_tgt, z_rho, torch.ones(2), stopgrad_target=True)
    loss.backward()
    assert z_src.grad is not None
    assert z_tgt.grad is None or torch.all(z_tgt.grad == 0)


def test_dino_sifess_forward_shapes_so2():
    model = DINOSifessModel(
        backbone="resnet18", out_dim=128, use_equivariance_head=True, eq_action="so2_2d"
    )
    x = torch.randn(2, 3, 64, 64)
    logits, z = model.embed_student(x)
    assert logits.shape == (2, 128)
    assert z.shape == (2, 256)
    z2 = model.apply_rho(z, torch.ones(2) * 0.5)
    assert z2.shape == z.shape
    assert not torch.allclose(z2[:, :2], z[:, :2])


def test_combined_loss_keys():
    crit = SIFESSLoss(out_dim=32, lambda_eq=1.0, lambda_orth=0.0, lambda_plane=0.1)
    s = [torch.randn(2, 32) for _ in range(3)]
    t = [torch.randn(2, 32) for _ in range(2)]
    z = torch.randn(2, 16)
    z = torch.nn.functional.normalize(z, dim=-1)
    z = z.clone()
    z[:, 0] = 0.3
    z[:, 1] = 0.3
    z = torch.nn.functional.normalize(z, dim=-1)
    out = crit(s, t, z_src=z, z_tgt=z, z_rho=z, cbar=torch.ones(2))
    assert "loss" in out and "l_dino" in out and "l_eq" in out


def test_tg_image_diff_nonzero():
    """T_g must materially change the image (mean |x - T_g x| > 0)."""
    torch.manual_seed(0)
    g0 = torch.randn(4, 3, 64, 64)
    frame_theta = torch.zeros(4)
    g_el = sample_frame_group_element(4, g0.device, angles=torch.ones(4) * 1.2)
    g_el.reflect[:] = False
    g0_warp = apply_frame_transform(g0, frame_theta, g_el)
    diff = float(mean_abs_image_diff(g0, g0_warp))
    assert diff > 0.05, f"expected material image warp, got {diff}"


def test_sifess_full_one_step_leq_nonzero():
    """One sifess_full train step must yield meaningfully nonzero L_eq."""
    torch.manual_seed(42)
    model = DINOSifessModel(
        backbone="resnet18", out_dim=128, use_equivariance_head=True, eq_action="so2_2d"
    )
    model.train()
    crit = SIFESSLoss(out_dim=128, lambda_eq=1.0, lambda_orth=0.0, lambda_plane=0.1)

    g0 = torch.randn(4, 3, 64, 64)
    crops = [g0, g0 + 0.01 * torch.randn_like(g0), *[torch.randn(4, 3, 48, 48) for _ in range(4)]]
    out = model.forward_multicrop(crops, crops[:2])

    st = compute_structure_tensor(g0, sigma_rho=1.5)
    frame_theta = pooled_frame_angle(st)
    g_el = sample_frame_group_element(g0.shape[0], device=g0.device)
    g0_warp = apply_frame_transform(g0, frame_theta, g_el)
    assert float(mean_abs_image_diff(g0, g0_warp)) > 0.01

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

    crit_v = SIFESSLoss(out_dim=128, lambda_eq=0.0, lambda_orth=0.0, lambda_plane=0.0)
    losses_v = crit_v(out["student_logits"], out["teacher_logits"])
    assert "l_eq" not in losses_v
    assert float(losses_v["loss"]) == float(losses_v["l_dino"])


def test_sifess_full_sustained_leq_after_n_steps():
    """After N train steps, mean L_eq must stay > 0.01 (collapse regression)."""
    torch.manual_seed(0)
    model = DINOSifessModel(
        backbone="resnet18", out_dim=128, use_equivariance_head=True, eq_action="so2_2d"
    )
    model.train()
    crit = SIFESSLoss(out_dim=128, lambda_eq=2.0, lambda_orth=0.0, lambda_plane=0.1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    n_steps = 20
    leq_vals = []
    tg_vals = []
    for _ in range(n_steps):
        g0 = torch.randn(8, 3, 64, 64)
        crops = [g0, g0 + 0.01 * torch.randn_like(g0)] + [
            torch.randn(8, 3, 48, 48) for _ in range(4)
        ]
        out = model.forward_multicrop(crops, crops[:2])
        st = compute_structure_tensor(g0, sigma_rho=1.5)
        frame_theta = pooled_frame_angle(st)
        g_el = sample_frame_group_element(8, g0.device)
        g0_warp = apply_frame_transform(g0, frame_theta, g_el)
        tg_vals.append(float(mean_abs_image_diff(g0, g0_warp)))

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
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        model.update_teacher()
        leq_vals.append(float(losses["l_eq"].detach()))

    mean_leq = sum(leq_vals) / len(leq_vals)
    mean_tg = sum(tg_vals) / len(tg_vals)
    late = sum(leq_vals[-5:]) / 5
    assert mean_tg > 0.05, f"T_g image diff collapsed: {mean_tg}"
    assert mean_leq > 0.01, f"mean L_eq over {n_steps} steps={mean_leq} (want > 0.01)"
    assert late > 0.01, f"late L_eq={late} collapsed toward 0"
