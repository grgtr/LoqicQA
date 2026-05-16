#!/usr/bin/env bash
# Run 12: self_consistency ablation (feature/visual_grounding)
# Compares sub_question_mode="self_consistency" vs Run 10 "rephrase"
#
# IMPORTANT: results saved to MAIN REPO (absolute path), not worktree.
# Worktree is only used for code isolation; it is cleaned up after the run.

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run12_sc"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
# Results go to MAIN REPO — survives worktree deletion
OUTPUT_DIR="${MAIN_REPO}/results/self_consistency_bb50"
GPU_THRESHOLD_MB=18000  # InternVL2.5-8B bf16 ~16GB; no LLMJudge loaded (disabled in config)
POLL_INTERVAL=30
LOG="/tmp/run12_sc.log"
SENTINEL="/tmp/run12.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    echo "[run12] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run12] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run12] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 12: self_consistency ablation"
echo " sub_question_mode=self_consistency + grounding"
echo "════════════════════════════════════════════"

# Create worktree at current feature/visual_grounding HEAD
SC_COMMIT=$(git -C "${MAIN_REPO}" rev-parse feature/visual_grounding)
mkdir -p "$(dirname "${WORKTREE}")"
if [ ! -d "${WORKTREE}" ]; then
    echo "[run12] Creating worktree at commit ${SC_COMMIT} ..."
    git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${SC_COMMIT}"
else
    echo "[run12] Worktree already exists at ${WORKTREE}"
fi

wait_for_gpu

echo "[run12] Starting at $(date)"
# Reduce allocator fragmentation to avoid OOM on tight memory budgets
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Note: no 'set -e' — script survives pipeline errors; results always saved
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_self_consistency_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run12] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run12] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
