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

> **Honesty:** result tables are **TBD**. This repo ships code, configs, tests,
> and a MedMNIST smoke path — not invented AUROCs.

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
python -m sifess.engines.eval_ood --checkpoint ... --probe outputs/linear_probe/probe.pt
```

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
scripts/smoke_test.sh
tests/
```

## Day-0 defaults

- Backbone: ResNet-18 (smoke) / ResNet-50 (default full)
- Batch 64, seed 42, fp16 on CUDA
- 2 global + 4 local crops
- OOD shifts: brightness / contrast / gamma (AUROC + Δ)

See `docs/METHOD.md` for equations and `docs/EXPERIMENT_CARD.md` for the
locked protocol and TBD tables.

## License

MIT — see `LICENSE`.
