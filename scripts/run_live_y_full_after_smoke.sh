#!/usr/bin/env bash
# Queue the long run only after the isolated no-teacher smoke has saved an adapter.
set -euo pipefail

smoke_pid="${1:?usage: run_live_y_full_after_smoke.sh <smoke-pid>}"
smoke_output="/workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl-smoke-fresh"
full_output="/workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl"

while kill -0 "$smoke_pid" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$smoke_output/adapter/adapter_config.json" ]]; then
  echo "Smoke run did not save an adapter; refusing to launch full training."
  exit 1
fi

cd /workspace/beer_distribution_live_y_rl
git pull --ff-only origin codex/live-y-qwen-rl
exec env PYTHONPATH=.:environments/beer_distribution_game \
  python scripts/train_qwen35_live_y_rl.py \
    --updates 6 \
    --group-size 4 \
    --train-minibatch 4 \
    --output-dir "$full_output"
