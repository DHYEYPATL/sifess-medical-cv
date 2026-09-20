# Contributing to SIFESS

Thank you for interest in contributing. This repository is a research codebase for
**Soft Isophote-Frame Equivariant Self-Supervision** in medical imaging.

## Principles

1. **No fake results.** Do not commit invented metrics, tables, or plots that imply
   completed experiments. Use TBD placeholders until numbers are measured.
2. **Reproducibility.** Every training/eval entrypoint must accept a seed and write
   JSON/CSV metrics with config fingerprints.
3. **Single backbone.** Keep the default backbone path clean (ResNet-18/50). New
   architectures belong behind a clear flag, not as silent defaults.
4. **Honest ablations.** Ablation names (`supervised`, `vanilla_ssl`, `sifess_full`,
   `sifess_no_gate`, `random_frame`) must mean what `docs/METHOD.md` says.

## Dev setup

```bash
cd sifess-medical-cv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
bash scripts/smoke_test.sh
```

## Code style

- Python 3.9+; type hints on public APIs where practical.
- Prefer small, testable modules (`structure_tensor`, `frame_transforms`, `equivariance`).
- Do not hard-code absolute paths; use config `data_root` / env vars.

## Pull requests

- Describe the scientific intent (what claim does the change test?).
- Include or update tests when changing ST / frame / residual math.
- Do not push large data or NIH CXR binaries into the repo.

## Issues

Open an issue for: dataset download blockers, numerical instability in ST/coherence,
or missing ablation coverage. Tag with `method`, `data`, or `infra`.
