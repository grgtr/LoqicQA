#!/usr/bin/env bash
# Run 18: Decomposed — fixes from Run 17 analysis
# Fixes:
#   1. Remove Phase A (components from NORMALITY_COMPONENTS directly)
#   2. Countable/uncountable split — Coverage prompt for bulk items
#   3. Dedup guard — questions about different components are never duplicates
#   4. Single ensemble seed [42]
# Results saved to MAIN REPO — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT=$(git -C "${MAIN_REPO}" rev-parse decomposed-stage1)
WORKTREE="/home/chikibriki/.lqa_worktrees/run18_decomposed"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/decomposed_bb50_r18"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run18_decomposed.log"
SENTINEL="/tmp/run18.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    echo "[run18] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run18] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run18] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 18: Decomposed — fixes from Run 17"
echo " 1. No Phase A (NORMALITY_COMPONENTS direct)"
echo " 2. Countable/uncountable count instruction"
echo " 3. Dedup component-identity guard"
echo " 4. Single ensemble seed [42]"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run18] Removing stale worktree..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run18] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

echo "[run18] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
"${VENV_PYTHON}" -u "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_decomposed_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run18] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run18] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
