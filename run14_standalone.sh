#!/usr/bin/env bash
# Run 14: perceptual probe sub-questions + visual grounding
# Waits for Run 13 sentinel, then runs.
#
# Key change vs Run 8/10: SUBQUESTION_AUGMENT_PROMPT now generates diverse
# visual probes (PRESENCE/ABSENCE/FEATURE/COUNT/SPATIAL) instead of rephrasing.
# Results saved to MAIN REPO — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run14_pp"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/perceptual_probes_bb50"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run14_pp.log"
SENTINEL="/tmp/run14.done"
RUN13_SENTINEL="/tmp/run13.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_run13() {
    echo "[run14] Waiting for Run 13 to finish (sentinel: ${RUN13_SENTINEL})..."
    while [ ! -f "${RUN13_SENTINEL}" ]; do
        echo "[run14] Run 13 still running, checking in ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    echo "[run14] Run 13 done — proceeding."
}

wait_for_gpu() {
    echo "[run14] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run14] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run14] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 14: perceptual probe sub-questions"
echo " rephrase→probes, grounding, no LLMJudge"
echo "════════════════════════════════════════════"

wait_for_run13
wait_for_gpu

SC_COMMIT=$(git -C "${MAIN_REPO}" rev-parse feature/visual_grounding)
mkdir -p "$(dirname "${WORKTREE}")"
if [ ! -d "${WORKTREE}" ]; then
    echo "[run14] Creating worktree at commit ${SC_COMMIT} ..."
    git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${SC_COMMIT}"
else
    echo "[run14] Worktree already exists at ${WORKTREE}"
fi

echo "[run14] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_perceptual_probes_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run14] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run14] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
