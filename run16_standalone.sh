#!/usr/bin/env bash
# Run 16: Decomposed Stage 1/2/4 — per-component sequential VLM calls
# Fixes: Stage 2 stochastic component loss (Run15 dropped cereals/bananas/almonds).
# Results saved to MAIN REPO — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT=$(git -C "${MAIN_REPO}" rev-parse decomposed-stage1)
WORKTREE="/home/chikibriki/.lqa_worktrees/run16_decomposed"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/decomposed_bb50"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run16_decomposed.log"
SENTINEL="/tmp/run16.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    echo "[run16] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run16] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run16] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 16: Decomposed Stage 1/2/4"
echo " per-component sequential VLM calls"
echo " decomposed_description=true, max_tiles=12"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run16] Removing stale worktree..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run16] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

echo "[run16] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_decomposed_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run16] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run16] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
