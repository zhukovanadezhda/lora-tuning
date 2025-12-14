#!/bin/bash
set -e

echo " Running all fine-tuning experiments "

# Move to repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Make Python see the project
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

export RESULTS_DIR="/content/drive/MyDrive/lora-tuning/results"
mkdir -p "$RESULTS_DIR"

# Run full fine-tuning
echo "[1/6] Full fine-tuning"
if [ ! -f "$RESULTS_DIR/full_ft.json" ]; then
  python -m scripts.run_comparison \
    --mode full_ft \
    --epochs 2 \
    --results-path "$RESULTS_DIR/full_ft.json"
else
  echo "Skipping full_ft (already exists)"
fi

# Run LoRA tunung with different ranks
for r in 1 2 4 8 16; do
  echo "[LoRA] r=${r}"
  if [ ! -f "$RESULTS_DIR/lora_r_${r}.json" ]; then
    python -m scripts.run_comparison \
      --mode lora \
      --r ${r} \
      --epochs 2 \
      --results-path "$RESULTS_DIR/lora_r_${r}.json"
  else
    echo "Skipping LoRA r=${r} (already exists)"
  fi
done

# Run Adapter Tuning
echo "[Adapter]"
if [ ! -f "$RESULTS_DIR/adapter.json" ]; then
  python -m scripts.run_comparison \
    --mode adapter \
    --epochs 2 \
    --results-path "$RESULTS_DIR/adapter.json"
else
  echo "Skipping adapter (already exists)"
fi

echo " All experiments completed successfully"
