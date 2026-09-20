# Experiment Card — SIFESS Medical CV

**Status:** scaffolding + Day-0 smoke on ChestMNIST. Full NIH CXR runs: TBD.

## Claim under test

Medical SSL should be soft-equivariant to each image’s structure-tensor /
isophote eigenframe, gated by coherence \(C\) — a foundational training law.

## Day-0 protocol (locked)

| Item | Value |
|------|-------|
| Dataset | ChestMNIST (MedMNIST), `size=224` |
| Backbone | ResNet-18 (smoke) / ResNet-50 (default full) |
| SSL | DINO student–teacher |
| Crops | 2 global + 4 local |
| Batch | 64 |
| Precision | fp16 when CUDA available |
| Seed | 42 |
| \(\sigma_\rho\) | 1.5 |
| \(\lambda_{\mathrm{eq}}\) | 0.5 |
| \(\lambda_{\mathrm{orth}}\) | 0.01 (or 0 if SO(2)-on-2D) |

### Ablations (SSL)

1. `vanilla_dino`
2. `sifess_full`
3. `no_c_gating` (\(\bar C=1\))
4. `random_frame` (random \(\theta\))

Optional reference: `supervised` end-to-end BCE (not part of the SSL quartet).

### Metrics

- Linear probe macro AUROC on ChestMNIST test (14 labels)
- OOD: brightness / contrast / gamma — log AUROC + \(\Delta\) vs clean
- Artifacts: JSON + CSV under `outputs/`

## Results (ChestMNIST Day-0)

| Ablation | Probe macro AUROC | OOD Δ (brightness) | OOD Δ (contrast) | OOD Δ (gamma) |
|----------|-------------------:|-------------------:|-----------------:|--------------:|
| vanilla_dino | TBD | TBD | TBD | TBD |
| sifess_full | TBD | TBD | TBD | TBD |
| no_c_gating | TBD | TBD | TBD | TBD |
| random_frame | TBD | TBD | TBD | TBD |

*Do not invent numbers. Fill after `run_ablations` + `linear_probe` + `eval_ood`.*

## Full-scale: NIH ChestX-ray14

| Item | Notes |
|------|-------|
| Data | NIH CXR14 (local / Kaggle). **Not auto-downloaded** (license + size). |
| Loader | `sifess/data/chestxray_loader.py` — expects `Data_Entry_2017.csv` + image folders |
| Backbone | ResNet-50 recommended |
| Protocol | Same DINO + SIFESS losses / ablations; longer epochs — TBD |
| Results | TBD |

### Known gaps for Kaggle CXR training

- NIH images must be attached manually (Kaggle dataset or GCS).
- Official train/val/test lists should be pinned in config (paths TBD).
- Multi-crop I/O may need more workers / cached decoded tensors on Kaggle.
- fp16 + batch 64 may need gradient accumulation on smaller GPUs.
- Logging to W&B optional; not wired by default.

## Repro commands

```bash
pip install -e ".[dev]"
pytest -q
bash scripts/smoke_test.sh

python -m sifess.engines.train_ssl --ablation sifess_full --backbone resnet50
python -m sifess.engines.linear_probe --checkpoint outputs/ssl/sifess_full/checkpoint.pt
python -m sifess.engines.eval_ood --checkpoint ... --probe outputs/linear_probe/probe.pt
python -m sifess.engines.run_ablations --config configs/ablations.yaml
```
