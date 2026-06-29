#!/usr/bin/env bash
# Run 19: Decomposed — seed 42, all fixes from Run 18 analysis
# Fixes vs Run 18:
#   1. NORMALITY_COMPONENTS dict bug fixed (no countable/uncountable fake components)
#   2. SUMMARIZE_COMPONENT_PROMPT — uncountable items get Coverage not Count
#   3. Sub-question generation: normality context + polarity check + fallback templates
#   4. ATOMIC_CONSTRAINTS expanded with almonds/banana chips standalone constraints
# Results saved to MAIN REPO — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT="8736a4874caf206ec6209bd814b1bf8626bdce4e"
WORKTREE="/home/chikibriki/.lqa_worktrees/run19_decomposed"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/decomposed_bb50_r19"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run19_decomposed.log"
SENTINEL="/tmp/run19.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{ sum += $1 } END { print sum }'
}

wait_for_gpu() {
    echo "[run19] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run19] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run19] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 19: Decomposed — seed 42, all fixes"
echo " Commit: ${COMMIT}"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run19] Removing stale worktree..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run19] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

echo "[run19] Starting at $(date)"
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
echo "[run19] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run19] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
