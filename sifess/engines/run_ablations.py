"""Run locked ablations: vanilla_dino | sifess_full | no_c_gating | random_frame.

Also supports a ``supervised`` baseline (train backbone with BCE end-to-end)
for reference — not part of the SSL ablation quartet but useful in EXPERIMENT_CARD.

CLI::
    python -m sifess.engines.run_ablations --config configs/ablations.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from sifess.data.medmnist_loader import build_chestmnist
from sifess.engines.train_ssl import ABLATIONS, train
from sifess.models.ssl_model import build_backbone
from sifess.utils import load_yaml, resolve_device, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SIFESS ablation runner")
    p.add_argument("--config", type=str, default="configs/ablations.yaml")
    p.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=None,
        help="Subset of: vanilla_dino sifess_full no_c_gating random_frame [supervised]",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def _base_cfg(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(args.config):
        cfg = load_yaml(args.config)
    # inherit ssl defaults
    ssl_path = cfg.get("ssl_config", "configs/ssl_train.yaml")
    if os.path.isfile(ssl_path):
        ssl = load_yaml(ssl_path)
        for k, v in ssl.items():
            cfg.setdefault(k, v)
    cfg.setdefault("ablations", list(ABLATIONS))
    cfg.setdefault("output_dir", "outputs/ablations")
    cfg.setdefault("seed", 42)
    if args.ablations:
        cfg["ablations"] = args.ablations
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.subset is not None:
        cfg["subset"] = args.subset
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    return cfg


def train_supervised(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end supervised ResNet on ChestMNIST (reference baseline)."""
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    out_dir = Path(cfg["output_dir"]) / "supervised"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader = build_chestmnist(
        split="train",
        size=int(cfg.get("image_size", 224)),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 2)),
        multicrop=False,
        subset=cfg.get("subset"),
    )
    backbone, feat_dim = build_backbone(cfg.get("backbone", "resnet18"))
    head = nn.Linear(feat_dim, 14)
    model = nn.Sequential(backbone, head).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(int(cfg.get("epochs", 1))):
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"supervised {epoch+1}", leave=False):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optim.zero_grad(set_to_none=True)
            loss = bce(model(x), y)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
        print(f"[supervised epoch {epoch+1}] loss={total/max(n,1):.4f}")

    ckpt_path = out_dir / "checkpoint.pt"
    # Save backbone under student_backbone. keys for probe compatibility? use raw
    torch.save({"model": backbone.state_dict(), "cfg": cfg, "ablation": "supervised"}, ckpt_path)
    summary = {"ablation": "supervised", "checkpoint": str(ckpt_path)}
    save_json(summary, str(out_dir / "summary.json"))
    return summary


def main(argv=None):
    args = parse_args(argv)
    cfg = _base_cfg(args)
    results: List[Dict[str, Any]] = []
    for name in cfg["ablations"]:
        print(f"=== ablation: {name} ===")
        if name == "supervised":
            summary = train_supervised(cfg)
        elif name in ABLATIONS:
            run_cfg = dict(cfg)
            run_cfg["ablation"] = name
            run_cfg["output_dir"] = str(Path(cfg["output_dir"]))
            summary = train(run_cfg)
        else:
            raise ValueError(f"unknown ablation {name}; expected {ABLATIONS} or supervised")
        results.append(summary)
    out = {"ablations": results}
    save_json(out, str(Path(cfg["output_dir"]) / "ablations_summary.json"))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
