#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${ARCHIVE:-/workspace/beer_v2_qwen_eval.tgz}"
WORKDIR="${WORKDIR:-/workspace/beer_v2_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/v2_qwen_outputs}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HOME

mkdir -p "$WORKDIR" "$OUTPUT_DIR"
tar -xzf "$ARCHIVE" -C "$WORKDIR"
cd "$WORKDIR"

python -m pip install -q -e . \
  'transformers==5.14.1' 'peft==0.20.0' 'accelerate>=1.13' \
  'safetensors>=0.7' sentencepiece

python scripts/eval_live_y_domain_randomized_grpo_v2_local.py \
  --label untrained_qwen_v2_runpod \
  --model-name Qwen/Qwen3.5-4B \
  --output "$OUTPUT_DIR/untrained_qwen_v2_runpod.json"

python scripts/eval_live_y_domain_randomized_grpo_v2_local.py \
  --label trained_qwen_grpo_v2_runpod \
  --model-name Qwen/Qwen3.5-4B \
  --adapter "$WORKDIR/artifacts/live_y_best_adapter" \
  --output "$OUTPUT_DIR/trained_qwen_grpo_v2_runpod.json"

sha256sum "$OUTPUT_DIR"/*.json > "$OUTPUT_DIR/SHA256SUMS"
touch "$OUTPUT_DIR/COMPLETE"
