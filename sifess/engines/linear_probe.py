"""Linear probe on frozen student backbone features (ChestMNIST multi-label).

CLI::
    python -m sifess.engines.linear_probe --checkpoint outputs/ssl/sifess_full/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from tqdm import tqdm

from sifess.data.medmnist_loader import build_chestmnist
from sifess.models.ssl_model import DINOSifessModel, build_backbone
from sifess.utils import load_yaml, resolve_device, save_csv_row, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SIFESS linear probe")
    p.add_argument("--config", type=str, default="configs/linear_probe.yaml")
    p.add_argument("--checkpoint", type=str, required=False, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    return p.parse_args(argv)


def _cfg(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(args.config):
        cfg = load_yaml(args.config)
    cfg.setdefault("seed", 42)
    cfg.setdefault("epochs", 20)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("lr", 1e-3)
    cfg.setdefault("backbone", "resnet50")
    cfg.setdefault("image_size", 224)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("output_dir", "outputs/linear_probe")
    cfg.setdefault("subset", None)
    cfg.setdefault("checkpoint", None)
    cfg.setdefault("fp16", True)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.subset is not None:
        cfg["subset"] = args.subset
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.backbone:
        cfg["backbone"] = args.backbone
    return cfg


@torch.no_grad()
def _extract(backbone, loader, device, use_fp16: bool):
    feats, labels = [], []
    backbone.eval()
    for batch in tqdm(loader, desc="extract", leave=False):
        x = batch["image"].to(device)
        y = batch["label"]
        with autocast(enabled=use_fp16 and device.type == "cuda"):
            f = backbone(x)
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def run_probe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = cfg["backbone"]
    backbone, feat_dim = build_backbone(backbone_name)
    n_classes = 14  # ChestMNIST

    if cfg.get("checkpoint"):
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu")
        # Prefer student backbone weights from DINOSifessModel
        state = ckpt.get("model", ckpt)
        bb_state = {
            k.replace("student_backbone.", ""): v
            for k, v in state.items()
            if k.startswith("student_backbone.")
        }
        if bb_state:
            backbone.load_state_dict(bb_state, strict=True)
            if "cfg" in ckpt and "backbone" in ckpt["cfg"]:
                backbone_name = ckpt["cfg"]["backbone"]
        else:
            # supervised / raw backbone checkpoint
            backbone.load_state_dict(state, strict=False)

    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    _, train_loader = build_chestmnist(
        split="train",
        size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        multicrop=False,
        subset=cfg.get("subset"),
    )
    _, test_loader = build_chestmnist(
        split="test",
        size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        multicrop=False,
        subset=cfg.get("subset"),
    )

    use_fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    x_train, y_train = _extract(backbone, train_loader, device, use_fp16)
    x_test, y_test = _extract(backbone, test_loader, device, use_fp16)

    probe = nn.Linear(feat_dim, n_classes).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=float(cfg["lr"]))
    bce = nn.BCEWithLogitsLoss()

    # Mini-batch over cached features
    bs = int(cfg["batch_size"])
    n = x_train.shape[0]
    history = []
    for epoch in range(int(cfg["epochs"])):
        probe.train()
        perm = torch.randperm(n)
        total = 0.0
        steps = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optim.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = bce(logits, yb)
            loss.backward()
            optim.step()
            total += float(loss.detach())
            steps += 1
        # Eval AUROC (macro over classes with both labels present)
        probe.eval()
        with torch.no_grad():
            logits = probe(x_test.to(device)).cpu()
            probs = torch.sigmoid(logits).numpy()
            yt = y_test.numpy()
        aurocs = []
        for c in range(n_classes):
            if yt[:, c].min() == yt[:, c].max():
                continue
            aurocs.append(roc_auc_score(yt[:, c], probs[:, c]))
        macro = float(sum(aurocs) / len(aurocs)) if aurocs else float("nan")
        row = {
            "epoch": epoch + 1,
            "loss": total / max(steps, 1),
            "macro_auroc": macro,
            "n_classes_scored": len(aurocs),
        }
        history.append(row)
        save_csv_row(str(out_dir / "metrics.csv"), row)
        print(f"[probe epoch {epoch+1}] loss={row['loss']:.4f} macro_auroc={macro:.4f}")

    summary = {
        "macro_auroc": history[-1]["macro_auroc"] if history else None,
        "checkpoint": cfg.get("checkpoint"),
        "backbone": backbone_name,
        "metrics_csv": str(out_dir / "metrics.csv"),
    }
    save_json(summary, str(out_dir / "summary.json"))
    torch.save({"probe": probe.state_dict(), "cfg": cfg}, out_dir / "probe.pt")
    return summary


def main(argv=None):
    import json

    args = parse_args(argv)
    cfg = _cfg(args)
    summary = run_probe(cfg)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
