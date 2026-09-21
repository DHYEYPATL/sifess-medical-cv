"""DINO + SIFESS SSL training engine.

CLI::
    python -m sifess.engines.train_ssl --config configs/ssl_train.yaml --ablation sifess_full
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from sifess.data.medmnist_loader import build_chestmnist
from sifess.geometry.frame_transforms import (
    apply_frame_transform,
    mean_abs_image_diff,
    sample_frame_group_element,
)
from sifess.geometry.structure_tensor import compute_structure_tensor, pooled_frame_angle
from sifess.losses.combined import SIFESSLoss
from sifess.models.ssl_model import DINOSifessModel
from sifess.utils import load_yaml, resolve_device, save_csv_row, save_json, set_seed


ABLATIONS = ("vanilla_dino", "sifess_full", "no_c_gating", "random_frame")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SIFESS DINO SSL training")
    p.add_argument("--config", type=str, default="configs/ssl_train.yaml")
    p.add_argument("--ablation", type=str, default="sifess_full", choices=ABLATIONS)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None, help="Limit train samples (smoke)")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fp16", action="store_true", default=None)
    p.add_argument(
        "--lambda-eq",
        type=float,
        default=None,
        help="Weight on L_eq (default from config; strong runs use 1.0–2.0)",
    )
    p.add_argument(
        "--lambda-orth",
        type=float,
        default=None,
        help="Weight on norm-preservation orth term (0 for pure SO(2)-on-2D)",
    )
    p.add_argument(
        "--lambda-plane",
        type=float,
        default=None,
        help="Weight on SO(2) plane energy floor",
    )
    p.add_argument(
        "--eq-action",
        type=str,
        default=None,
        choices=("so2_2d", "film"),
        help="Feature action: fixed SO(2) on first 2 dims (default) or learned FiLM",
    )
    return p.parse_args(argv)


def _merge_cfg(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(args.config):
        cfg = load_yaml(args.config)
    cfg.setdefault("ablation", "sifess_full")
    cfg.setdefault("seed", 42)
    cfg.setdefault("epochs", 100)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("backbone", "resnet50")
    cfg.setdefault("lr", 5e-4)
    cfg.setdefault("weight_decay", 0.04)
    cfg.setdefault("lambda_eq", 1.0)
    cfg.setdefault("lambda_orth", 0.0)
    cfg.setdefault("lambda_plane", 0.1)
    cfg.setdefault("plane_min", 0.05)
    cfg.setdefault("eq_action", "so2_2d")
    cfg.setdefault("stopgrad_target", True)
    cfg.setdefault("out_dim", 4096)
    cfg.setdefault("sigma_rho", 1.5)
    cfg.setdefault("n_global", 2)
    cfg.setdefault("n_local", 4)
    cfg.setdefault("image_size", 224)
    cfg.setdefault("fp16", True)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("output_dir", "outputs/ssl")
    cfg.setdefault("subset", None)
    cfg.setdefault("teacher_momentum", 0.996)

    if args.ablation:
        cfg["ablation"] = args.ablation
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.subset is not None:
        cfg["subset"] = args.subset
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.fp16 is True:
        cfg["fp16"] = True
    if args.lambda_eq is not None:
        cfg["lambda_eq"] = float(args.lambda_eq)
    if args.lambda_orth is not None:
        cfg["lambda_orth"] = float(args.lambda_orth)
    if args.lambda_plane is not None:
        cfg["lambda_plane"] = float(args.lambda_plane)
    if args.eq_action is not None:
        cfg["eq_action"] = args.eq_action
    return cfg


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    out_dir = Path(cfg["output_dir"]) / cfg["ablation"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, str(out_dir / "config.json"))

    ablation = cfg["ablation"]
    use_eq = ablation != "vanilla_dino"
    lambda_eq = float(cfg["lambda_eq"]) if use_eq else 0.0
    lambda_orth = float(cfg["lambda_orth"]) if use_eq else 0.0
    lambda_plane = float(cfg.get("lambda_plane", 0.1)) if use_eq else 0.0
    # SO(2)-on-2D: orth unused by default
    if cfg.get("so2_only", False) or str(cfg.get("eq_action", "so2_2d")).lower().startswith("so2"):
        lambda_orth = 0.0

    _, loader = build_chestmnist(
        split="train",
        size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        multicrop=True,
        n_global=int(cfg["n_global"]),
        n_local=int(cfg["n_local"]),
        subset=cfg.get("subset"),
    )

    model = DINOSifessModel(
        backbone=cfg["backbone"],
        out_dim=int(cfg["out_dim"]),
        use_equivariance_head=use_eq,
        momentum=float(cfg["teacher_momentum"]),
        eq_action=str(cfg.get("eq_action", "so2_2d")),
    ).to(device)

    criterion = SIFESSLoss(
        out_dim=int(cfg["out_dim"]),
        lambda_eq=lambda_eq,
        lambda_orth=lambda_orth,
        lambda_plane=lambda_plane,
        plane_min=float(cfg.get("plane_min", 0.05)),
        stopgrad_target=bool(cfg.get("stopgrad_target", True)),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    use_fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_fp16)

    history = []
    t0 = time.time()
    n_global = int(cfg["n_global"])

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = {
            "loss": 0.0,
            "l_dino": 0.0,
            "l_eq": 0.0,
            "l_plane": 0.0,
            "tg_abs": 0.0,
            "n": 0,
        }
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg['epochs']} [{ablation}]", leave=False)
        for batch in pbar:
            crops = [c.to(device, non_blocking=True) for c in batch["crops"]]
            # Teacher sees global crops only
            teacher_crops = crops[:n_global]
            student_crops = crops

            # Frame / equivariance pair from first global crop
            g0 = teacher_crops[0]
            st = compute_structure_tensor(g0, sigma_rho=float(cfg["sigma_rho"]))
            if ablation == "random_frame":
                frame_theta = torch.rand(g0.shape[0], device=device) * (2 * torch.pi)
            else:
                frame_theta = pooled_frame_angle(st)

            if ablation == "no_c_gating":
                cbar = torch.ones(g0.shape[0], device=device)
            else:
                cbar = st.Cbar

            g_el = sample_frame_group_element(g0.shape[0], device=device)
            # Warped view for soft equivariance target
            g0_warp = apply_frame_transform(g0, frame_theta, g_el)
            tg_abs = mean_abs_image_diff(g0, g0_warp)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=use_fp16):
                out = model.forward_multicrop(student_crops, teacher_crops)
                z_src = None
                z_tgt = None
                z_rho = None
                if use_eq:
                    _, z_src = model.embed_student(g0)
                    _, z_tgt = model.embed_student(g0_warp)
                    z_rho = model.apply_rho(z_src, g_el.angle, g_el.reflect)

                losses = criterion(
                    out["student_logits"],
                    out["teacher_logits"],
                    z_src=z_src,
                    z_tgt=z_tgt,
                    z_rho=z_rho,
                    cbar=cbar if use_eq else None,
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            model.update_teacher()

            bs = g0.shape[0]
            running["loss"] += float(loss.detach()) * bs
            running["l_dino"] += float(losses["l_dino"].detach()) * bs
            if "l_eq" in losses:
                running["l_eq"] += float(losses["l_eq"].detach()) * bs
            if "l_plane" in losses:
                running["l_plane"] += float(losses["l_plane"].detach()) * bs
            running["tg_abs"] += float(tg_abs.detach()) * bs
            running["n"] += bs
            pbar.set_postfix(loss=float(loss.detach()), tg=float(tg_abs.detach()))

        n = max(running["n"], 1)
        row = {
            "epoch": epoch + 1,
            "ablation": ablation,
            "loss": running["loss"] / n,
            "l_dino": running["l_dino"] / n,
            "l_eq": running["l_eq"] / n,
            "l_plane": running["l_plane"] / n,
            "tg_abs_mean": running["tg_abs"] / n,
            "seconds": time.time() - t0,
        }
        history.append(row)
        save_csv_row(str(out_dir / "metrics.csv"), row)
        print(
            f"[epoch {epoch+1}] loss={row['loss']:.4f} "
            f"l_dino={row['l_dino']:.4f} l_eq={row['l_eq']:.4f} "
            f"l_plane={row['l_plane']:.4f} |x-Tgx|={row['tg_abs_mean']:.4f}"
        )

    ckpt = {
        "model": model.state_dict(),
        "cfg": cfg,
        "ablation": ablation,
    }
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    summary = {
        "ablation": ablation,
        "epochs": cfg["epochs"],
        "final_loss": history[-1]["loss"] if history else None,
        "final_l_eq": history[-1]["l_eq"] if history else None,
        "final_tg_abs_mean": history[-1]["tg_abs_mean"] if history else None,
        "checkpoint": str(ckpt_path),
        "metrics_csv": str(out_dir / "metrics.csv"),
        "device": str(device),
        "backbone": cfg["backbone"],
        "eq_action": cfg.get("eq_action"),
        "lambda_eq": lambda_eq,
    }
    save_json(summary, str(out_dir / "summary.json"))
    return summary


def main(argv=None):
    args = parse_args(argv)
    cfg = _merge_cfg(args)
    summary = train(cfg)
    print(json_dumps(summary))
    return summary


def json_dumps(obj):
    import json

    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    main()
