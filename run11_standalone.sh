#!/usr/bin/env bash
# Run 11: dev_evaluation_framework (Priorities 1+2, no grounding)
# Ждёт завершения любого активного пайплайна, затем запускается.
set -euo pipefail

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run11_dev_eval"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
GPU_THRESHOLD_MB=30000
POLL_INTERVAL=30
LOG="/tmp/run11_dev_eval.log"
SENTINEL="/tmp/run11.done"

rm -f "${SENTINEL}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

pipeline_running() {
    pgrep -f "run_pipeline.py" > /dev/null 2>&1
}

wait_for_pipeline_finish() {
    echo "[run11] Waiting for active pipeline runs to finish..."
    while pipeline_running; do
        echo "[run11] Pipeline still running, waiting ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    echo "[run11] No active pipeline detected."
}

wait_for_gpu() {
    echo "[run11] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run11] GPU free: ${FREE} MB"
        [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ] && { echo "[run11] GPU ready."; return; }
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 11: dev_evaluation_framework (P1+P2, no grounding)"
echo "════════════════════════════════════════════"

wait_for_pipeline_finish
wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ ! -d "${WORKTREE}" ]; then
    echo "[run11] Creating worktree for dev_evaluation_framework..."
    git -C "${MAIN_REPO}" worktree add "${WORKTREE}" dev_evaluation_framework
else
    echo "[run11] Worktree already exists at ${WORKTREE}"
fi

echo "[run11] Starting at $(date)"
cd "${WORKTREE}"
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_improved_test_bb50.yaml" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run11] Finished at $(date). Log: ${LOG}"
