#!/bin/bash
set -e

echo " Running all fine-tuning experiments "

cd "$(dirname "$0")"

mkdir -p outputs/results
mkdir -p outputs/runs

# Run full fine-tuning
echo "[1/6] Full fine-tuning"
python -m scripts.run_comparison \
  --mode full_ft \
  --epochs 2

# Run LoRA tunung with different ranks
for r in 1 2 4 8; do
  echo "[LoRA] r=${r}"
  python -m scripts.run_comparison \
    --mode lora \
    --r ${r} \
    --epochs 2
done

# Run Adapter Tuning
echo "[Adapter]"
python -m scripts.run_comparison \
  --mode adapter \
  --epochs 2

echo " All experiments completed successfully"
