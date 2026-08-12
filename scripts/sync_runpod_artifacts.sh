#!/usr/bin/env bash
# Mirror a live RunPod training output directory into the local artifacts tree.
#
# A pod can vanish (spend cap, preemption) between checkpoints, so on-pod
# snapshots alone are not "saved".
#
# Fetches by name rather than with a recursive rsync.  The pod's /workspace is
# MooseFS over FUSE, where stat-ing a tree of adapter shards is far slower than
# moving the bytes: an `rsync` of the whole output directory stalled for hours
# while a targeted copy of the same bookkeeping files took five seconds.  Large
# directories are therefore pulled exactly once, when they first appear.
set -uo pipefail

HOST="${HOST:?set HOST, e.g. root@69.30.85.5}"
PORT="${PORT:-22}"
REMOTE="${REMOTE:?set REMOTE, e.g. /workspace/outputs/live-y-corrected-v3}"
LOCAL="${LOCAL:?set LOCAL, e.g. artifacts/live_y_corrected_v3}"
KEY="${KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
INTERVAL="${INTERVAL:-600}"

# scp's port flag is -P, not -p (-p means "preserve timestamps"), and BSD getopt
# on macOS stops parsing options at the first non-option argument.  Sharing one
# option array between ssh and scp therefore turns the identity-file path into a
# *source file to copy* -- which silently deposits the private key in $LOCAL.
# Keep the two option sets separate, and keep every flag ahead of the paths.
COMMON_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
             -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
SCP_OPTS=(-P "$PORT" "${COMMON_OPTS[@]}")

mkdir -p "$LOCAL"

get_file() {  # remote-relative path
  scp "${SCP_OPTS[@]}" "$HOST:$REMOTE/$1" "$LOCAL/" 2>/dev/null
}

get_dir() {   # remote-relative dir -> pulled once, but only counted as done when complete
  local name="$1"
  # "The directory exists" is not "the copy succeeded": an scp interrupted
  # mid-transfer leaves a partial directory behind, and skipping on mere
  # existence means it is never retried.  Gate on a sentinel written only after
  # the local byte count matches the remote.
  [ -f "$LOCAL/$name/.complete" ] && return 0

  local remote_bytes
  remote_bytes=$(ssh "${COMMON_OPTS[@]}" -p "$PORT" "$HOST" \
      "du -sb '$REMOTE/$name' 2>/dev/null | cut -f1" 2>/dev/null)
  [ -z "$remote_bytes" ] && return 1

  rm -rf "$LOCAL/$name"
  mkdir -p "$LOCAL/$(dirname "$name")"
  scp -r "${SCP_OPTS[@]}" "$HOST:$REMOTE/$name" "$LOCAL/$(dirname "$name")/" 2>/dev/null

  local local_bytes
  local_bytes=$(find "$LOCAL/$name" -type f -exec stat -f%z {} + 2>/dev/null \
                | awk '{s+=$1} END{print s+0}')
  if [ "${local_bytes:-0}" = "$remote_bytes" ]; then
    touch "$LOCAL/$name/.complete"
    echo "[$(date -u +%H:%M:%S)] pulled $name ($local_bytes bytes, verified)"
  else
    echo "[$(date -u +%H:%M:%S)] INCOMPLETE $name (local ${local_bytes:-0} vs remote $remote_bytes) - will retry"
    return 1
  fi
}

while true; do
  for f in training_metrics.json rollouts.jsonl checkpoint_state.json \
           run_config.json research_protocol.json \
           eval_pre_development.json eval_post_development.json; do
    get_file "$f"
  done

  # Numbered snapshots: copy each exactly once, the round after it appears.
  done_updates=$(python3 -c "
import json,sys
try:
    print(json.load(open('$LOCAL/checkpoint_state.json'))['completed_updates'])
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  for n in $(seq 1 "${done_updates:-0}"); do
    get_dir "checkpoints/update_$(printf '%03d' "$n")"
  done

  # The best adapter moves, so re-pull it whenever the recorded best changes.
  best=$(python3 -c "
import json
try:
    print(json.load(open('$LOCAL/checkpoint_state.json')).get('best',{}).get('update') or 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  if [ "${best:-0}" != "0" ] && [ "${best:-0}" != "$(cat "$LOCAL/.best_synced" 2>/dev/null || echo -)" ]; then
    remote_bytes=$(ssh "${COMMON_OPTS[@]}" -p "$PORT" "$HOST" \
        "du -sb '$REMOTE/adapter_best' 2>/dev/null | cut -f1" 2>/dev/null)
    rm -rf "$LOCAL/adapter_best"
    scp -r "${SCP_OPTS[@]}" "$HOST:$REMOTE/adapter_best" "$LOCAL/" 2>/dev/null
    local_bytes=$(find "$LOCAL/adapter_best" -type f -exec stat -f%z {} + 2>/dev/null \
                  | awk '{s+=$1} END{print s+0}')
    # Only record the best as synced once the bytes match, so a partial copy is
    # retried on the next round instead of masquerading as the saved result.
    if [ -n "$remote_bytes" ] && [ "${local_bytes:-0}" = "$remote_bytes" ]; then
      echo "$best" > "$LOCAL/.best_synced"
      echo "[$(date -u +%H:%M:%S)] pulled adapter_best (update $best, verified $local_bytes bytes)"
    else
      echo "[$(date -u +%H:%M:%S)] INCOMPLETE adapter_best (local ${local_bytes:-0} vs remote ${remote_bytes:-?}) - will retry"
    fi
  fi

  echo "[$(date -u +%H:%M:%S)] synced through update ${done_updates:-?} ($(du -sh "$LOCAL" 2>/dev/null | cut -f1))"
  sleep "$INTERVAL"
done
