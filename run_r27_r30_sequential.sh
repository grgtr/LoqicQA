#!/usr/bin/env bash
# run_r27_r30_sequential.sh — Sequential Runs 27–30
# Classes: juice_bottle → pushpins → screw_bag → splicing_connectors
# Model: InternVL2.5-8B | Pre-built questions (skip Stage 1-3) | exclude structural_anomaly
# Branch: feature/rc4a-rc5 @ 05cfbb3
#
# Usage:
#   bash run_r27_r30_sequential.sh          # launch in new tmux session
#   bash run_r27_r30_sequential.sh --attach # launch and attach immediately
#
# Session: lqa_r27_r30
# Logs:    /tmp/r27_juice_bottle.log  ...  /tmp/r30_splicing_connectors.log
# Sentinels: /tmp/r27.done  ...  /tmp/r30.done

# ── Config ──────────────────────────────────────────────────────────────────

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT="05cfbb3"
WORKTREE="/home/chikibriki/.lqa_worktrees/run_r27_r30"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${WORKTREE}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
CONFIGS_DIR="${WORKTREE}/configs"
QUESTIONS_DIR="${WORKTREE}/questions"

GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30

TMUX_SESSION="lqa_r27_r30"

# ── tmux launcher ────────────────────────────────────────────────────────────

ATTACH=0
for arg in "$@"; do
    [[ "${arg}" == "--attach" ]] && ATTACH=1
done

if [[ -z "${TMUX:-}" ]] && [[ "${1:-}" != "--_inside_tmux" ]]; then
    echo "[launcher] Starting tmux session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" \
        -x 220 -y 50 \
        "bash ${MAIN_REPO}/run_r27_r30_sequential.sh --_inside_tmux; exec bash"
    echo "[launcher] Session started."
    echo "  Attach:    tmux attach -t ${TMUX_SESSION}"
    echo "  Monitor:   tail -f /tmp/r27_juice_bottle.log"
    echo "  All logs:  /tmp/r2{7,8,9}_*.log  /tmp/r30_*.log"
    if [[ "${ATTACH}" -eq 1 ]]; then
        tmux attach -t "${TMUX_SESSION}"
    fi
    exit 0
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
        echo "[queue] GPU free: ${FREE} MB  (need ${GPU_THRESHOLD_MB})"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[queue] GPU ready for ${label}."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

# ── Worktree setup ───────────────────────────────────────────────────────────

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[queue] Removing stale worktree at ${WORKTREE}..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[queue] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"
echo "[queue] Worktree ready: ${WORKTREE}"

# ── Run function ─────────────────────────────────────────────────────────────

rm -f /tmp/r27.done /tmp/r28.done /tmp/r29.done /tmp/r30.done

run_one() {
    local RUN_ID="$1"
    local CLASS="$2"
    local CONFIG="${CONFIGS_DIR}/$3"
    local QUESTIONS="${QUESTIONS_DIR}/$4"
    local OUTPUT_DIR="${MAIN_REPO}/results/$5"
    local LOG="$6"
    local SENTINEL="$7"

    rm -f "${SENTINEL}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "════════════════════════════════════════════════════════"
    printf " %-4s: %-26s  commit=%s\n" "${RUN_ID^^}" "${CLASS}" "${COMMIT}"
    echo " Config:    ${CONFIG}"
    echo " Questions: ${QUESTIONS}"
    echo " Output:    ${OUTPUT_DIR}"
    echo " Log:       ${LOG}"
    echo "════════════════════════════════════════════════════════"

    wait_for_gpu "${RUN_ID}"

    echo "[${RUN_ID}] Starting at $(date)"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1

    "${VENV_PYTHON}" -u "${SCRIPT}" \
        --class_name "${CLASS}" \
        --config "${CONFIG}" \
        --questions_file "${QUESTIONS}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        2>&1 | tee "${LOG}"

    local EXIT_CODE=${PIPESTATUS[0]}
    echo "[${RUN_ID}] Pipeline exited with code ${EXIT_CODE} at $(date)"
    touch "${SENTINEL}"
    return ${EXIT_CODE}
}

# ── Sequential runs ──────────────────────────────────────────────────────────

run_one "r27" "juice_bottle" \
    "r27_juice_bottle.yaml" \
    "juice_bottle_questions.json" \
    "r27_juice_bottle" \
    "/tmp/r27_juice_bottle.log" \
    "/tmp/r27.done"

run_one "r28" "pushpins" \
    "r28_pushpins.yaml" \
    "pushpins_questions.json" \
    "r28_pushpins" \
    "/tmp/r28_pushpins.log" \
    "/tmp/r28.done"

run_one "r29" "screw_bag" \
    "r29_screw_bag.yaml" \
    "screw_bag_questions.json" \
    "r29_screw_bag" \
    "/tmp/r29_screw_bag.log" \
    "/tmp/r29.done"

run_one "r30" "splicing_connectors" \
    "r30_splicing_connectors.yaml" \
    "splicing_connectors_questions.json" \
    "r30_splicing_connectors" \
    "/tmp/r30_splicing_connectors.log" \
    "/tmp/r30.done"

# ── Cleanup & summary ────────────────────────────────────────────────────────

git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo ""
echo "════════════════════════════════════════════════════════"
echo " ALL RUNS COMPLETE  (commit ${COMMIT})"
echo "════════════════════════════════════════════════════════"
echo "  r27 juice_bottle       → results/r27_juice_bottle/          log: /tmp/r27_juice_bottle.log"
echo "  r28 pushpins           → results/r28_pushpins/              log: /tmp/r28_pushpins.log"
echo "  r29 screw_bag          → results/r29_screw_bag/             log: /tmp/r29_screw_bag.log"
echo "  r30 splicing_connectors→ results/r30_splicing_connectors/   log: /tmp/r30_splicing_connectors.log"
