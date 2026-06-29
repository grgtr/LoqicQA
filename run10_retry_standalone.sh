#!/usr/bin/env bash
# Run 10 retry: feature/visual_grounding at commit e8e87c1
# Same code/config as original Run 10 (e8e87c1 — constraint filter fix).
# Saves artifacts to MAIN REPO (survives worktree deletion).
# Log: /tmp/run10_retry.log  (does NOT overwrite /tmp/run10_vg.log)

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT="e8e87c1"
WORKTREE="/home/chikibriki/.lqa_worktrees/run10_retry"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/visual_grounding_bb50_retry"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run10_retry.log"
SENTINEL="/tmp/run10_retry.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    echo "[run10_retry] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run10_retry] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run10_retry] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 10 RETRY: e8e87c1 (constraint filter fix)"
echo " config_visual_grounding_bb50.yaml"
echo " use_grounded_reasoning=true, use_llm_judge=true"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run10_retry] Removing stale worktree at ${WORKTREE}..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run10_retry] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

echo "[run10_retry] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_visual_grounding_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run10_retry] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run10_retry] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
