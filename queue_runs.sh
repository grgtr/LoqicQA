#!/usr/bin/env bash
# queue_runs.sh — Sequential Run 9 + Run 10 with GPU wait
# Run 9: dev_evaluation_framework (main repo, no grounding)
# Run 10: feature/visual_grounding (worktree, with grounding)
set -euo pipefail

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE_DIR="${HOME}/.lqa_worktrees"
WORKTREE_R10="${WORKTREE_DIR}/run10_vg"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"

GPU_THRESHOLD_MB=30000
POLL_INTERVAL=30

SENTINEL_R9="/tmp/run9.done"
SENTINEL_R10="/tmp/run10.done"
LOG_R9="/tmp/run9_dev_eval.log"
LOG_R10="/tmp/run10_vg.log"

rm -f "${SENTINEL_R9}" "${SENTINEL_R10}"

# ── helpers ────────────────────────────────────────────────────────────────

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

setup_worktree() {
    mkdir -p "${WORKTREE_DIR}"
    if [ -d "${WORKTREE_R10}" ]; then
        echo "[queue] Worktree already exists at ${WORKTREE_R10}, skipping creation."
    else
        echo "[queue] Creating worktree for feature/visual_grounding..."
        git -C "${MAIN_REPO}" worktree add "${WORKTREE_R10}" feature/visual_grounding
    fi
}

cleanup_worktree() {
    if [ -d "${WORKTREE_R10}" ]; then
        echo "[queue] Removing worktree ${WORKTREE_R10}..."
        git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE_R10}" || true
    fi
}

# ── Run 9 ──────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════"
echo " RUN 9: dev_evaluation_framework (no grounding)"
echo "════════════════════════════════════════════"

wait_for_gpu "Run 9"

echo "[queue] Starting Run 9 at $(date)"
cd "${MAIN_REPO}"
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${MAIN_REPO}/config_improved_test_bb50.yaml" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG_R9}"

touch "${SENTINEL_R9}"
echo "[queue] Run 9 finished at $(date). Sentinel: ${SENTINEL_R9}"

# ── Run 10 ─────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════"
echo " RUN 10: feature/visual_grounding (with grounding)"
echo "════════════════════════════════════════════"

setup_worktree
wait_for_gpu "Run 10"

echo "[queue] Starting Run 10 at $(date)"
cd "${WORKTREE_R10}"
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE_R10}/config_visual_grounding_bb50.yaml" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG_R10}"

touch "${SENTINEL_R10}"
echo "[queue] Run 10 finished at $(date). Sentinel: ${SENTINEL_R10}"

# ── cleanup ────────────────────────────────────────────────────────────────

cleanup_worktree

echo ""
echo "[queue] All runs complete."
echo "  Run 9 log:  ${LOG_R9}"
echo "  Run 10 log: ${LOG_R10}"
echo "  Run 9 results:  ${MAIN_REPO}/results/ (check config for output_dir)"
echo "  Run 10 results: ${WORKTREE_R10}/results/visual_grounding_bb50/"
