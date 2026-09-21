"""OOD intensity evaluation: brightness / contrast / gamma.

Logs AUROC under each shift and delta vs clean.
CLI::
    python -m sifess.engines.eval_ood --checkpoint ... --probe ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sifess.data.medmnist_loader import build_chestmnist
from sifess.data.ood_intensity import OODConfig, apply_ood_intensity
from sifess.models.ssl_model import build_backbone
from sifess.utils import load_yaml, resolve_device, save_csv_row, save_json, set_seed


SHIFTS = ("none", "brightness", "contrast", "gamma")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SIFESS OOD intensity eval")
    p.add_argument("--config", type=str, default="configs/ood_eval.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--probe", type=str, default=None, help="probe.pt from linear_probe")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--severity", type=float, default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--batch-size", "--batch_size", type=int, default=None,
                   dest="batch_size", help="Eval dataloader batch size")
    return p.parse_args(argv)


def _cfg(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if os.path.isfile(args.config):
        cfg = load_yaml(args.config)
    cfg.setdefault("seed", 42)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("backbone", "resnet50")
    cfg.setdefault("image_size", 224)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("output_dir", "outputs/ood_eval")
    cfg.setdefault("severity", 0.75)
    cfg.setdefault("subset", None)
    cfg.setdefault("checkpoint", None)
    cfg.setdefault("probe", None)
    cfg.setdefault("n_classes", 14)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.probe:
        cfg["probe"] = args.probe
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.severity is not None:
        cfg["severity"] = args.severity
    if args.subset is not None:
        cfg["subset"] = args.subset
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    return cfg


def _load_backbone(cfg, device):
    backbone, feat_dim = build_backbone(cfg["backbone"])
    if cfg.get("checkpoint"):
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu")
        state = ckpt.get("model", ckpt)
        bb_state = {
            k.replace("student_backbone.", ""): v
            for k, v in state.items()
            if k.startswith("student_backbone.")
        }
        if bb_state:
            backbone.load_state_dict(bb_state, strict=True)
        else:
            backbone.load_state_dict(state, strict=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, feat_dim


def _macro_auroc(probs, yt, n_classes: int) -> float:
    aurocs = []
    for c in range(n_classes):
        if yt[:, c].min() == yt[:, c].max():
            continue
        aurocs.append(roc_auc_score(yt[:, c], probs[:, c]))
    return float(sum(aurocs) / len(aurocs)) if aurocs else float("nan")


@torch.no_grad()
def eval_shift(backbone, probe, loader, device, ood: OODConfig, n_classes: int) -> float:
    probs_all, yt_all = [], []
    for batch in tqdm(loader, desc=f"ood:{ood.shift}", leave=False):
        x = batch["image"].to(device)
        x = apply_ood_intensity(x, ood)
        f = backbone(x)
        logits = probe(f)
        probs_all.append(torch.sigmoid(logits).cpu())
        yt_all.append(batch["label"])
    probs = torch.cat(probs_all, 0).numpy()
    yt = torch.cat(yt_all, 0).numpy()
    return _macro_auroc(probs, yt, n_classes)


def run_ood(cfg: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone, feat_dim = _load_backbone(cfg, device)
    n_classes = int(cfg["n_classes"])
    probe = nn.Linear(feat_dim, n_classes).to(device)
    if cfg.get("probe"):
        pckpt = torch.load(cfg["probe"], map_location="cpu")
        probe.load_state_dict(pckpt["probe"])
    else:
        print("WARNING: no probe checkpoint; using randomly initialized probe (AUROC ~ chance).")
    probe.eval()

    _, test_loader = build_chestmnist(
        split="test",
        size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        multicrop=False,
        subset=cfg.get("subset"),
    )

    results: List[Dict[str, Any]] = []
    clean = None
    for shift in SHIFTS:
        sev = 0.0 if shift == "none" else float(cfg["severity"])
        auroc = eval_shift(
            backbone,
            probe,
            test_loader,
            device,
            OODConfig(shift=shift, severity=sev),  # type: ignore[arg-type]
            n_classes,
        )
        if shift == "none":
            clean = auroc
            delta = 0.0
        else:
            delta = float(auroc - clean) if clean is not None else float("nan")
        row = {
            "shift": shift,
            "severity": sev,
            "macro_auroc": auroc,
            "delta_vs_clean": delta,
        }
        results.append(row)
        save_csv_row(str(out_dir / "ood_metrics.csv"), row)
        print(f"[ood] {shift}: AUROC={auroc:.4f} delta={delta:.4f}")

    summary = {"results": results, "clean_auroc": clean, "severity": cfg["severity"]}
    save_json(summary, str(out_dir / "summary.json"))
    return summary


def main(argv=None):
    import json

    args = parse_args(argv)
    cfg = _cfg(args)
    summary = run_ood(cfg)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
