#!/usr/bin/env bash
# run_r27_r30_sequential.sh — Sequential Runs 27–30
# Classes: juice_bottle → pushpins → screw_bag → splicing_connectors
# Model: InternVL2.5-8B | Pre-built questions (skip Stage 1-3) | exclude structural_anomaly
#
# Usage:
#   bash run_r27_r30_sequential.sh          # launch in new tmux session
#   bash run_r27_r30_sequential.sh --attach # launch and attach immediately
#
# The script runs inside a detached tmux session "lqa_r27_r30".
# Logs: /tmp/r27_juice_bottle.log ... /tmp/r30_splicing_connectors.log
# Sentinels: /tmp/r27.done ... /tmp/r30.done

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────

MAIN_REPO="/home/chikibriki/LoqicQA"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
CONFIGS_DIR="${MAIN_REPO}/configs"
QUESTIONS_DIR="${MAIN_REPO}/questions"

GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30

TMUX_SESSION="lqa_r27_r30"

# ── tmux launcher ────────────────────────────────────────────────────────────

ATTACH=0
if [[ "${1:-}" == "--attach" ]]; then
    ATTACH=1
fi

# If we are NOT already inside the tmux session, create it and re-run there.
if [[ -z "${TMUX:-}" ]]; then
    echo "[launcher] Starting tmux session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" \
        -x 220 -y 50 \
        "bash ${MAIN_REPO}/run_r27_r30_sequential.sh --_inside_tmux; exec bash"
    echo "[launcher] Session started. Logs:"
    echo "  /tmp/r27_juice_bottle.log"
    echo "  /tmp/r28_pushpins.log"
    echo "  /tmp/r29_screw_bag.log"
    echo "  /tmp/r30_splicing_connectors.log"
    echo ""
    echo "  Attach:  tmux attach -t ${TMUX_SESSION}"
    echo "  Monitor: tail -f /tmp/r27_juice_bottle.log"
    if [[ "${ATTACH}" -eq 1 ]]; then
        tmux attach -t "${TMUX_SESSION}"
    fi
    exit 0
fi

# Guard: only proceed when called from inside tmux (--_inside_tmux or TMUX set)
if [[ "${1:-}" != "--_inside_tmux" && -z "${TMUX:-}" ]]; then
    echo "[error] Must run via tmux. Use: bash run_r27_r30_sequential.sh"
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    local label="$1"
    echo "[queue] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory (${label})..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[queue] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[queue] GPU ready for ${label}."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

run_one() {
    local RUN_ID="$1"          # e.g. r27
    local CLASS="$2"           # e.g. juice_bottle
    local CONFIG="$3"          # e.g. configs/r27_juice_bottle.yaml
    local QUESTIONS="$4"       # e.g. questions/juice_bottle_questions.json
    local OUTPUT_DIR="$5"      # e.g. results/r27_juice_bottle
    local LOG="$6"             # e.g. /tmp/r27_juice_bottle.log
    local SENTINEL="$7"        # e.g. /tmp/r27.done

    rm -f "${SENTINEL}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " ${RUN_ID^^}: ${CLASS}  |  8B model  |  pre-built questions"
    echo " Config:    ${CONFIG}"
    echo " Questions: ${QUESTIONS}"
    echo " Output:    ${OUTPUT_DIR}"
    echo " Log:       ${LOG}"
    echo "════════════════════════════════════════════════════════"

    wait_for_gpu "${RUN_ID}"

    echo "[${RUN_ID}] Starting at $(date)"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1

    cd "${MAIN_REPO}"
    "${VENV_PYTHON}" -u "${SCRIPT}" \
        --class_name "${CLASS}" \
        --config "${CONFIG}" \
        --questions_file "${QUESTIONS}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        2>&1 | tee "${LOG}"

    PIPELINE_EXIT=${PIPESTATUS[0]}
    echo "[${RUN_ID}] Pipeline exited with code ${PIPELINE_EXIT} at $(date)"
    touch "${SENTINEL}"
}

# ── Clear old sentinels ──────────────────────────────────────────────────────

rm -f /tmp/r27.done /tmp/r28.done /tmp/r29.done /tmp/r30.done

# ── Run 27: juice_bottle ─────────────────────────────────────────────────────

run_one "r27" "juice_bottle" \
    "${CONFIGS_DIR}/r27_juice_bottle.yaml" \
    "${QUESTIONS_DIR}/juice_bottle_questions.json" \
    "${MAIN_REPO}/results/r27_juice_bottle" \
    "/tmp/r27_juice_bottle.log" \
    "/tmp/r27.done"

# ── Run 28: pushpins ─────────────────────────────────────────────────────────

run_one "r28" "pushpins" \
    "${CONFIGS_DIR}/r28_pushpins.yaml" \
    "${QUESTIONS_DIR}/pushpins_questions.json" \
    "${MAIN_REPO}/results/r28_pushpins" \
    "/tmp/r28_pushpins.log" \
    "/tmp/r28.done"

# ── Run 29: screw_bag ────────────────────────────────────────────────────────

run_one "r29" "screw_bag" \
    "${CONFIGS_DIR}/r29_screw_bag.yaml" \
    "${QUESTIONS_DIR}/screw_bag_questions.json" \
    "${MAIN_REPO}/results/r29_screw_bag" \
    "/tmp/r29_screw_bag.log" \
    "/tmp/r29.done"

# ── Run 30: splicing_connectors ──────────────────────────────────────────────

run_one "r30" "splicing_connectors" \
    "${CONFIGS_DIR}/r30_splicing_connectors.yaml" \
    "${QUESTIONS_DIR}/splicing_connectors_questions.json" \
    "${MAIN_REPO}/results/r30_splicing_connectors" \
    "/tmp/r30_splicing_connectors.log" \
    "/tmp/r30.done"

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo " ALL RUNS COMPLETE"
echo "════════════════════════════════════════════════════════"
echo "  r27 juice_bottle      → results/r27_juice_bottle/       log: /tmp/r27_juice_bottle.log"
echo "  r28 pushpins          → results/r28_pushpins/           log: /tmp/r28_pushpins.log"
echo "  r29 screw_bag         → results/r29_screw_bag/          log: /tmp/r29_screw_bag.log"
echo "  r30 splicing_conn     → results/r30_splicing_connectors/ log: /tmp/r30_splicing_connectors.log"
