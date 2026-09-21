# SIFESS — Soft Isophote-Frame Equivariant Self-Supervision

Medical self-supervised learning should be **soft-equivariant** to each image’s
own **structure-tensor / isophote eigenframe**, gated by **coherence \(C\)**.
That constraint is treated here as a *foundational training law*, not as an
ensemble, freeze schedule, or architecture footnote.

This repository implements a clean **DINO student–teacher** baseline plus a
coherence-weighted soft equivariance residual \(L_{\mathrm{eq}}\).

\[
L = L_{\mathrm{DINO}} + \lambda_{\mathrm{eq}} L_{\mathrm{eq}} + \lambda_{\mathrm{orth}} L_{\mathrm{orth}}
\]

Defaults: \(\lambda_{\mathrm{eq}}=0.5\), \(\lambda_{\mathrm{orth}}=0.01\)
(set to 0 for pure SO(2)-on-2D action). Structure tensor scale \(\sigma_\rho=1.5\).

## Quickstart

```bash
cd sifess-medical-cv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
bash scripts/smoke_test.sh
```

Day-0 data: **ChestMNIST** (MedMNIST), `size=224`. NIH ChestX-ray14 is documented
for full runs (`sifess/data/chestxray_loader.py`) but is **not** auto-downloaded.

## Ablations (locked)

| Flag | Behavior |
|------|----------|
| `vanilla_dino` | DINO only (\(\lambda_{\mathrm{eq}}=0\)) |
| `sifess_full` | DINO + \(C\)-gated \(L_{\mathrm{eq}}\) |
| `no_c_gating` | \(L_{\mathrm{eq}}\) with \(\bar C := 1\) |
| `random_frame` | replace eigenframe \(\theta\) by random angle |

```bash
python -m sifess.engines.train_ssl --ablation sifess_full --backbone resnet18 --epochs 2 --subset 256
python -m sifess.engines.run_ablations --config configs/ablations.yaml
python -m sifess.engines.linear_probe --checkpoint outputs/ssl/sifess_full/checkpoint.pt
python -m sifess.engines.eval_ood --checkpoint ... --probe outputs/linear_probe/probe.pt --batch-size 32
```

## Day-0 linear-probe + OOD results (REAL)

Source: Kaggle notebook [`sifess-day0-chestmnist`](https://www.kaggle.com/code/dbpdhyey/sifess-day0-chestmnist) **Version 2**
(`scripts/kaggle_day0.sh`, ResNet-18, **8 epochs**, subset **8192**, batch **32**, GPU T4×2, ~83 min).
Macro AUROC on ChestMNIST test at 1% / 10% / 100% labeled fractions.
OOD: intensity shifts at severity **0.75** (AUROC and \(\Delta\) vs clean).

### Linear probe

| Ablation | 1% | 10% | 100% |
|----------|-----:|-----:|------:|
| `vanilla_dino` | 0.6657 | 0.6711 | 0.7036 |
| `sifess_full` | 0.6590 | 0.6663 | 0.6989 |
| `no_c_gating` | 0.6641 | 0.6663 | 0.7006 |
| `random_frame` | 0.6573 | 0.6619 | 0.6933 |

### OOD \(\Delta\) vs clean (severity 0.75)

| Ablation | clean AUROC | \(\Delta\) brightness | \(\Delta\) contrast | \(\Delta\) gamma |
|----------|------------:|---------------------:|-------------------:|-----------------:|
| `sifess_full` | 0.6989 | −0.0458 | −0.0060 | −0.0019 |
| `vanilla_dino` | 0.7036 | −0.0557 | −0.0076 | −0.0008 |

\(L_{\mathrm{eq}}\) logging works (nonzero on epoch 1 for equivariance ablations:
~0.0005–0.0006), then collapses to 0.0000 for epochs 2–8 — magnitude still
needs investigation.

> Measured Day-0 numbers, not invented. Longer ResNet-50 / full-epoch tables remain open.
> Prior Version 1 was a 5-epoch / 4096-subset pass without OOD (see git history).

## Layout

```
sifess/
  geometry/     # structure tensor, frame transforms
  models/       # DINO student–teacher, rho(g)
  losses/       # L_DINO, L_eq, combined
  engines/      # train_ssl, linear_probe, eval_ood, run_ablations
  data/         # ChestMNIST, NIH stub, OOD intensity
configs/        # yaml
docs/METHOD.md  docs/EXPERIMENT_CARD.md
notebooks/kaggle_train.ipynb
scripts/smoke_test.sh  scripts/kaggle_day0.sh
tests/
```

## Day-0 defaults

- Backbone: ResNet-18 (smoke) / ResNet-50 (default full)
- Batch 64, seed 42, fp16 on CUDA
- 2 global + 4 local crops
- OOD shifts: brightness / contrast / gamma (AUROC + Δ)

See `docs/METHOD.md` for equations and `docs/EXPERIMENT_CARD.md` for the
locked protocol.

## License

MIT — see `LICENSE`.
