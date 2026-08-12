#!/usr/bin/env bash
# Wait for a RunPod training job to finish, pull its artifacts with verification,
# and only then terminate the pod.
#
# Runs detached on the local machine, independent of any agent session, so the
# pod does not keep billing after the job ends.
#
# TERMINATION IS DESTRUCTIVE: the pod's volume goes with it.  The teardown is
# therefore gated on a verified local copy -- every file byte-matched against the
# pod and the adapter proven loadable.  If verification fails the pod is LEFT
# RUNNING and the failure is logged, because paying for an idle GPU is a far
# cheaper mistake than deleting the only copy of the weights.
set -uo pipefail

HOST="${HOST:?}"; PORT="${PORT:-22}"; PID="${PID:?trainer pid on the pod}"
REMOTE="${REMOTE:?}"; LOCAL="${LOCAL:?}"; POD_ID="${POD_ID:?}"
KEY="${KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
DURABLE="${DURABLE:-artifacts/live_y_best_adapter}"
POLL="${POLL:-120}"

COMMON=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
SSH_OPTS=("${COMMON[@]}" -p "$PORT")
SCP_OPTS=("${COMMON[@]}" -P "$PORT")

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

remote_alive() { ssh "${SSH_OPTS[@]}" "$HOST" true 2>/dev/null; }
trainer_running() { ssh "${SSH_OPTS[@]}" "$HOST" "test -d /proc/$PID" 2>/dev/null; }

pull_verified() {  # remote-relative path (file or dir) -> 0 on byte-verified copy
  local name="$1" rbytes lbytes
  rbytes=$(ssh "${SSH_OPTS[@]}" "$HOST" "du -sb '$REMOTE/$name' 2>/dev/null | cut -f1" 2>/dev/null)
  [ -z "$rbytes" ] && { log "  MISSING on pod: $name"; return 1; }
  rm -rf "${LOCAL:?}/$name"
  mkdir -p "$LOCAL/$(dirname "$name")"
  scp -r "${SCP_OPTS[@]}" "$HOST:$REMOTE/$name" "$LOCAL/$(dirname "$name")/" 2>/dev/null
  lbytes=$(find "$LOCAL/$name" -type f ! -name '.complete' -exec stat -f%z {} + 2>/dev/null \
           | awk '{s+=$1} END{print s+0}')
  if [ "${lbytes:-0}" = "$rbytes" ]; then
    log "  verified $name ($lbytes bytes)"; return 0
  fi
  log "  FAILED $name (local ${lbytes:-0} vs remote $rbytes)"; return 1
}

log "watching trainer pid $PID on $HOST (poll ${POLL}s)"
misses=0
while true; do
  if ! remote_alive; then
    misses=$((misses + 1))
    log "pod unreachable ($misses consecutive)"
    # The pod may already be gone (balance, preemption); nothing left to bill.
    [ "$misses" -ge 10 ] && { log "pod gone for 10 checks; nothing to terminate, exiting"; exit 0; }
    sleep "$POLL"; continue
  fi
  misses=0
  trainer_running || break
  sleep "$POLL"
done

log "trainer exited; pulling final artifacts"
ok=0
for f in training_metrics.json rollouts.jsonl checkpoint_state.json run_config.json \
         research_protocol.json eval_pre_development.json eval_post_development.json; do
  pull_verified "$f" || ok=1
done
pull_verified "adapter_best" || ok=1
pull_verified "adapter" || ok=1

planned=$(python3 -c "
import json
try: print(json.load(open('$LOCAL/checkpoint_state.json'))['planned_updates'])
except Exception: print(0)" 2>/dev/null)
for n in $(seq 1 "${planned:-0}"); do
  d="checkpoints/update_$(printf '%03d' "$n")"
  [ -f "$LOCAL/$d/.complete" ] && continue
  pull_verified "$d" && touch "$LOCAL/$d/.complete" || log "  (checkpoint $n unavailable)"
done

# The adapter must actually load, not merely weigh the right number of bytes.
if ! python3 -c "
from safetensors import safe_open
import sys
with safe_open('$LOCAL/adapter_best/adapter_model.safetensors','pt') as f:
    n=len(list(f.keys()))
sys.exit(0 if n>0 else 1)
" 2>/dev/null; then
  log "FATAL: adapter_best does not load. LEAVING POD $POD_ID RUNNING."
  exit 1
fi
log "adapter_best loads cleanly"

if [ "$ok" -ne 0 ]; then
  log "FATAL: one or more artifacts failed verification. LEAVING POD $POD_ID RUNNING."
  exit 1
fi

mkdir -p "$DURABLE"
rm -rf "${DURABLE:?}"/*
cp -R "$LOCAL/adapter_best/." "$DURABLE/"
python3 - <<PY
import json, hashlib, pathlib, datetime
d = pathlib.Path("$DURABLE")
h = hashlib.sha256((d/"adapter_model.safetensors").read_bytes()).hexdigest()
sel = json.loads((d/"selection.json").read_text())
m = {"status": "FINAL - training completed",
     "selected_update": sel["update"],
     "selection_criterion": "lowest mean training-rollout local cost among 100%-protocol-clean updates",
     "training_rollout_local_cost_mean": sel["cost"],
     "base_model": "Qwen/Qwen3.5-4B",
     "saved_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "sha256_adapter_model_safetensors": h,
     "caveat": "Selected on TRAINING rollout cost; not validated on development or held-out seeds."}
(d/"MANIFEST.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
print("durable manifest written, update", sel["update"])
PY

log "all artifacts verified; terminating pod $POD_ID"
runpodctl remove pod "$POD_ID" 2>&1 | tail -2
log "DONE"
