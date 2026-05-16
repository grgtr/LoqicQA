#!/usr/bin/env bash
# Run 15: perceptual probes + inversion fix (3d0a054)
# Same as Run 14 but with fixed sub-question prompt:
#   - No hallucinated objects (from f42bb84, same as run14)
#   - Inverted Yes/No absence questions fixed (3d0a054, NEW)
#     WRONG: "Is there an empty spot where X should be?" (Yes = anomaly)
#     RIGHT: "Can you see X?" (Yes = normal)
# Results saved to MAIN REPO — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run15_pp2"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/perceptual_probes_v2_bb50"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run15_pp2.log"
SENTINEL="/tmp/run15.done"
RUN14_SENTINEL="/tmp/run14.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_run14() {
    echo "[run15] Waiting for Run 14 to finish (sentinel: ${RUN14_SENTINEL})..."
    while [ ! -f "${RUN14_SENTINEL}" ]; do
        echo "[run15] Run 14 still running, checking in ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    echo "[run15] Run 14 done — proceeding."
}

wait_for_gpu() {
    echo "[run15] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run15] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run15] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 15: perceptual probes v2 (inversion fix)"
echo " no hallucinations + Yes=normal enforced"
echo "════════════════════════════════════════════"

wait_for_run14
wait_for_gpu

SC_COMMIT=$(git -C "${MAIN_REPO}" rev-parse feature/visual_grounding)
mkdir -p "$(dirname "${WORKTREE}")"
if [ ! -d "${WORKTREE}" ]; then
    echo "[run15] Creating worktree at commit ${SC_COMMIT} ..."
    git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${SC_COMMIT}"
else
    echo "[run15] Worktree already exists at ${WORKTREE}"
fi

echo "[run15] Starting at $(date)"
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
echo "[run15] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run15] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
