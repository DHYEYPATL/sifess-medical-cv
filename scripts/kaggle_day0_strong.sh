#!/usr/bin/env bash
# Strong Day-0 Kaggle/T4 run: sustained L_eq signal (fixed SO(2), lambda_eq=2.0).
# EPOCHS=15 SUBSET=16384 — method fix validation after Day-0 V2 collapse.
set -euo pipefail

REPO_URL="https://github.com/DHYEYPATL/sifess-medical-cv.git"
WORKING_ROOT="${KAGGLE_WORKING:-/kaggle/working}"
REPO="${WORKING_ROOT}/sifess-medical-cv"
OUT="${DAY0_OUT:-outputs/day0_strong}"
BACKBONE="${DAY0_BACKBONE:-resnet18}"
EPOCHS="${DAY0_EPOCHS:-15}"
BATCH_SIZE="${DAY0_BATCH_SIZE:-32}"
NUM_WORKERS="${DAY0_NUM_WORKERS:-2}"
SUBSET="${DAY0_SUBSET:-16384}"
SEED="${DAY0_SEED:-42}"
LAMBDA_EQ="${DAY0_LAMBDA_EQ:-2.0}"
EQ_ACTION="${DAY0_EQ_ACTION:-so2_2d}"

mkdir -p "$WORKING_ROOT"
if [[ ! -f "$REPO/pyproject.toml" ]]; then
  git clone "$REPO_URL" "$REPO"
else
  echo "Using existing checkout: $REPO"
  git -C "$REPO" pull --ff-only || true
fi
cd "$REPO"
python -m pip install -e .
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "A Kaggle GPU is required; enable a T4 accelerator."
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUT"
TRAIN_SIZE="$(python - <<'PY'
from medmnist import ChestMNIST
print(len(ChestMNIST(split="train", download=True, size=224, as_rgb=True)))
PY
)"
echo "ChestMNIST train size=$TRAIN_SIZE"
echo "Strong run: EPOCHS=$EPOCHS SUBSET=$SUBSET LAMBDA_EQ=$LAMBDA_EQ EQ_ACTION=$EQ_ACTION"

train_args=(
  --config configs/ssl_train.yaml
  --backbone "$BACKBONE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --fp16
  --lambda-eq "$LAMBDA_EQ"
  --eq-action "$EQ_ACTION"
  --subset "$SUBSET"
  --output-dir "$OUT"
)

for ablation in vanilla_dino sifess_full no_c_gating random_frame; do
  echo "=== SSL: $ablation ==="
  python -m sifess.engines.train_ssl "${train_args[@]}" --ablation "$ablation"

  for frac in 0.01 0.1 1.0; do
    n_labeled="$TRAIN_SIZE"
    if [[ "$frac" == "0.01" ]]; then n_labeled="$(( (TRAIN_SIZE + 50) / 100 ))"; fi
    if [[ "$frac" == "0.1" ]]; then n_labeled="$(( (TRAIN_SIZE + 5) / 10 ))"; fi
    frac_tag="$frac"
    if [[ "$frac" == "1.0" ]]; then frac_tag="1"; fi
    probe_dir="$OUT/$ablation/probe_frac_$frac_tag"
    echo "--- linear probe: $ablation label_frac=$frac n_labeled=$n_labeled ---"
    python -m sifess.engines.linear_probe \
      --config configs/linear_probe.yaml \
      --checkpoint "$OUT/$ablation/checkpoint.pt" \
      --backbone "$BACKBONE" \
      --batch-size "$BATCH_SIZE" \
      --subset "$n_labeled" \
      --seed "$SEED" \
      --output-dir "$probe_dir"
  done
done

python - "$OUT" "$TRAIN_SIZE" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
train_size = int(sys.argv[2])
rows = []
for ablation in ("vanilla_dino", "sifess_full", "no_c_gating", "random_frame"):
    metrics = out / ablation / "metrics.csv"
    if metrics.is_file():
        print(f"=== SSL metrics: {ablation} ===")
        print(metrics.read_text())
    for frac in (0.01, 0.1, 1.0):
        n_labeled = train_size if frac == 1.0 else max(1, round(train_size * frac))
        tag = f"{frac:g}"
        summary_path = out / ablation / f"probe_frac_{tag}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        rows.append({
            "ablation": ablation,
            "label_frac": frac,
            "n_labeled": n_labeled,
            "macro_auroc": summary.get("macro_auroc"),
            "metrics_csv": str(summary_path.parent / "metrics.csv"),
        })
with (out / "linear_probe_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out / 'linear_probe_metrics.csv'}")
PY

for ablation in sifess_full vanilla_dino; do
  echo "=== OOD: $ablation ==="
  python -m sifess.engines.eval_ood \
    --config configs/ood_eval.yaml \
    --checkpoint "$OUT/$ablation/checkpoint.pt" \
    --probe "$OUT/$ablation/probe_frac_1/probe.pt" \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --brightness 0.75 \
    --output-dir "$OUT/$ablation/ood"
done

python - "$OUT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1])
print("=== observed linear-probe results ===")
print((out / "linear_probe_metrics.csv").read_text())
for ablation in ("sifess_full", "vanilla_dino"):
    summary = json.loads((out / ablation / "ood" / "summary.json").read_text())
    print(f"=== observed OOD results: {ablation} ===")
    print(json.dumps(summary, indent=2))
archive = Path(shutil.make_archive("day0_strong_outputs", "zip", root_dir=out))
print(f"Download archive: {archive.resolve()}")
PY
