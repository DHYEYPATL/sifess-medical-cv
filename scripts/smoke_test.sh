#!/usr/bin/env bash
# Day-0 smoke: tiny ChestMNIST subset, ResNet-18, 1 epoch, sifess_full + vanilla_dino.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
OUT="${SMOKE_OUT:-outputs/smoke}"
mkdir -p "$OUT"

echo "== SIFESS smoke test =="
echo "root=$ROOT"
python -c "import sifess, torch; print('sifess', sifess.__version__, 'torch', torch.__version__)"

echo "== pytest =="
python -m pytest -q

echo "== train vanilla_dino (1 epoch, subset 64) =="
python -m sifess.engines.train_ssl \
  --config configs/ssl_train.yaml \
  --ablation vanilla_dino \
  --backbone resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --subset 64 \
  --seed 42 \
  --output-dir "$OUT/ssl"

echo "== train sifess_full (1 epoch, subset 64) =="
python -m sifess.engines.train_ssl \
  --config configs/ssl_train.yaml \
  --ablation sifess_full \
  --backbone resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --subset 64 \
  --seed 42 \
  --output-dir "$OUT/ssl"

echo "== linear probe (tiny) =="
python -m sifess.engines.linear_probe \
  --config configs/linear_probe.yaml \
  --checkpoint "$OUT/ssl/sifess_full/checkpoint.pt" \
  --backbone resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --subset 64 \
  --seed 42 \
  --output-dir "$OUT/probe"

echo "== ood eval (tiny) =="
python -m sifess.engines.eval_ood \
  --config configs/ood_eval.yaml \
  --checkpoint "$OUT/ssl/sifess_full/checkpoint.pt" \
  --probe "$OUT/probe/probe.pt" \
  --backbone resnet18 \
  --subset 64 \
  --severity 0.75 \
  --output-dir "$OUT/ood"

echo "== smoke OK =="
echo "artifacts under $OUT"
ls -la "$OUT/ssl/sifess_full" "$OUT/probe" "$OUT/ood" || true
