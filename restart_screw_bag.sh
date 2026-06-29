#!/bin/bash
# Restart screw_bag full pipeline with seed retry until ≥1 question passes Stage 3b.
#
# Strategy:
#   - Each attempt uses a different random seed (controls which 3 normal images
#     are sampled for few-shot setup). Different normals → different Stage 2
#     summary → different Stage 3 candidates → higher chance some pass filter.
#   - n_questions is increased to 10 (vs 6 in baseline) for more candidates.
#   - Waits for GPU memory before each attempt; uses the same flock mechanism
#     as run_full_parallel.sh so concurrent sessions aren't disrupted.
#
# EXTENSIBILITY: future "improved framework" queue jobs can be appended to
#   /tmp/logicqa_run_queue.txt as "class:config:threshold_MiB:output_base"
#   and processed by a separate run_queued_jobs.sh (not yet implemented).
#
# Usage:
#   bash restart_screw_bag.sh               # launch in current shell
#   tmux new-session -d -s retry_screw_bag "bash restart_screw_bag.sh; bash"

set -euo pipefail

CLASS="screw_bag"
CONFIG="config_baseline_full.yaml"
DATA_DIR="/home/chikibriki/LoqicQA/dataset-ninja/"
OUT_BASE="results_baseline_full"
OUT_DIR="${OUT_BASE}/${CLASS}"
THRESHOLD=25000        # MiB free on GPU before launching
LOCK_FILE="/tmp/logicqa_gpu_launch.lock"
LOCK_HOLD_SECONDS=123  # seconds to hold lock after launch (InternVL load time)
CHECK_INTERVAL=60      # seconds between GPU memory polls
MAX_SEEDS=10
N_QUESTIONS=10         # more candidates → better chance of passing filter

DONE_FLAG="${OUT_DIR}/.done_retry"

mkdir -p "$OUT_DIR"
rm -f "$DONE_FLAG"

log() { echo "[$(date '+%H:%M:%S')] [${CLASS}] $*"; }

# ------------------------------------------------------------------ #
# Show other active full_* sessions for situational awareness
# ------------------------------------------------------------------ #
log "Active full_* tmux sessions:"
tmux ls 2>/dev/null | grep "^full_" | sed 's/^/    /' || echo "    (none)"
echo ""

# ------------------------------------------------------------------ #
# Seed retry loop
# ------------------------------------------------------------------ #
for SEED in $(seq 1 $MAX_SEEDS); do
    log "=== Attempt seed=${SEED} / ${MAX_SEEDS} ==="

    # ---- Wait for GPU memory + launch lock ----
    log "Waiting for ${THRESHOLD} MiB free on GPU..."
    while true; do
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                   | head -1 | tr -d ' ')
        log "GPU free: ${FREE_MIB} MiB / need: ${THRESHOLD} MiB"

        if [ "${FREE_MIB}" -ge "${THRESHOLD}" ]; then
            exec 9>"$LOCK_FILE"
            if flock -n 9; then
                log "Got launch lock — launching seed=${SEED}"
                # Release lock in background after model is loaded into VRAM
                ( sleep $LOCK_HOLD_SECONDS; flock -u 9 ) &
                break
            else
                log "Lock busy (another model loading), waiting..."
            fi
        fi
        sleep $CHECK_INTERVAL
    done

    # ---- Run pipeline for this seed ----
    ATTEMPT_DIR="${OUT_DIR}/attempt_seed${SEED}"
    mkdir -p "$ATTEMPT_DIR"

    log "Output dir: ${ATTEMPT_DIR}"
    .venv/bin/python3 scripts/run_pipeline.py \
        --class_name "$CLASS" \
        --config     "$CONFIG" \
        --data_dir   "$DATA_DIR" \
        --output_dir "$ATTEMPT_DIR" \
        --n_questions "$N_QUESTIONS" \
        --seed        "$SEED" \
        --save_questions \
        || { log "Pipeline exited with error, trying next seed..."; continue; }

    # ---- Check main_questions count ----
    Q_FILE="${ATTEMPT_DIR}/${CLASS}_questions.json"
    if [ ! -f "$Q_FILE" ]; then
        log "Questions file not found at ${Q_FILE}, trying next seed..."
        continue
    fi

    N_MAIN=$(.venv/bin/python3 - <<PYEOF
import json
with open("${Q_FILE}") as f:
    d = json.load(f)
print(len(d.get("main_questions", [])))
PYEOF
)

    log "Seed ${SEED}: ${N_MAIN} main question(s) passed Stage 3b filtering"

    if [ "${N_MAIN}" -gt 0 ]; then
        # ---- Success: promote results to canonical OUT_DIR location ----
        cp "$Q_FILE" "${OUT_DIR}/${CLASS}_questions.json"
        if [ -f "${ATTEMPT_DIR}/${CLASS}_results.json" ]; then
            cp "${ATTEMPT_DIR}/${CLASS}_results.json" "${OUT_DIR}/${CLASS}_results.json"
        fi
        touch "$DONE_FLAG"
        log "SUCCESS — seed=${SEED}, ${N_MAIN} question(s). Results in: ${ATTEMPT_DIR}"
        log "Canonical outputs copied to: ${OUT_DIR}/"
        echo ""
        echo "========================================"
        echo "  screw_bag DONE (seed=${SEED}, n_main=${N_MAIN})"
        echo "  questions : ${OUT_DIR}/${CLASS}_questions.json"
        echo "  results   : ${OUT_DIR}/${CLASS}_results.json"
        echo "  run dir   : ${ATTEMPT_DIR}"
        echo "========================================"
        exit 0
    fi

    log "Zero questions with seed=${SEED} — results discarded, trying next seed..."
    # Stage 4 results from this attempt are trivial (0 questions) — kept for logs only
done

# ------------------------------------------------------------------ #
# All seeds exhausted
# ------------------------------------------------------------------ #
log "FAILED: no valid questions after ${MAX_SEEDS} seed attempts."
log "Consider:"
log "  - Lowering question_filter_threshold (currently 0.8) in ${CONFIG}"
log "  - Increasing n_shots (currently 3) in ${CONFIG}"
log "  - Checking Stage 3a logs: ${OUT_DIR}/attempt_seed*/stage3a_questions.json"
exit 1
