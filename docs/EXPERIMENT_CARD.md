# Experiment Card — SIFESS Medical CV

**Status:** scaffolding + Day-0 ChestMNIST (Kaggle V2 with OOD). Full NIH CXR runs: TBD.

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
| Batch | 64 (Kaggle Day-0 V2 used 32) |
| Precision | fp16 when CUDA available |
| Seed | 42 |
| \(\sigma_\rho\) | 1.5 |
| \(\lambda_{\mathrm{eq}}\) | 1.0 (strong: 2.0) |
| \(\lambda_{\mathrm{orth}}\) / eq_action | 0 / `so2_2d` |
| \(\lambda_{\mathrm{plane}}\) | 0.1 |

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

**Source:** [Kaggle `sifess-day0-chestmnist` Version 2](https://www.kaggle.com/code/dbpdhyey/sifess-day0-chestmnist)
— ResNet-18, **8 epochs**, subset **8192**, batch **32**, GPU T4×2 (~4990 s).
Macro AUROC at 1% / 10% / 100% labeled. OOD severity 0.75.

| Ablation | 1% | 10% | 100% | OOD Δ (brightness) | OOD Δ (contrast) | OOD Δ (gamma) |
|----------|-----:|-----:|------:|-------------------:|-----------------:|--------------:|
| vanilla_dino | 0.6657 | 0.6711 | 0.7036 | −0.0557 | −0.0076 | −0.0008 |
| sifess_full | 0.6590 | 0.6663 | 0.6989 | −0.0458 | −0.0060 | −0.0019 |
| no_c_gating | 0.6641 | 0.6663 | 0.7006 | — | — | — |
| random_frame | 0.6573 | 0.6619 | 0.6933 | — | — | — |

Notes: V2 \(L_{\mathrm{eq}}\) ~5e-4 on epoch 1 then 0 for epochs 2–8 (collapse; see METHOD.md). Method fix on main; re-run `kaggle_day0_strong.sh`.
OOD eval ran for `sifess_full` and `vanilla_dino` only.

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
