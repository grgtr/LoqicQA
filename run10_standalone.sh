#!/usr/bin/env bash
# Run 10: feature/visual_grounding (Priorities 1+2+3, with grounding)
set -euo pipefail

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run10_vg"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
GPU_THRESHOLD_MB=30000
POLL_INTERVAL=30
LOG="/tmp/run10_vg.log"
SENTINEL="/tmp/run10.done"

rm -f "${SENTINEL}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    echo "[run10] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run10] GPU free: ${FREE} MB"
        [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ] && { echo "[run10] GPU ready."; return; }
        sleep "${POLL_INTERVAL}"
    done
}

mkdir -p "$(dirname "${WORKTREE}")"
VG_COMMIT=$(git -C "${MAIN_REPO}" rev-parse feature/visual_grounding)  # e8e87c1 — includes constraint filter fix
if [ ! -d "${WORKTREE}" ]; then
    echo "[run10] Creating worktree at commit ${VG_COMMIT} (detached)..."
    git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${VG_COMMIT}"
else
    echo "[run10] Worktree already exists at ${WORKTREE}"
fi

echo ""
echo "════════════════════════════════════════════"
echo " RUN 10: feature/visual_grounding (grounding + P1+P2+P3)"
echo "════════════════════════════════════════════"

wait_for_gpu

echo "[run10] Starting at $(date)"
cd "${WORKTREE}"
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_visual_grounding_bb50.yaml" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run10] Finished at $(date). Log: ${LOG}"
