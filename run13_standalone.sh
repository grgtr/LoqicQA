#!/usr/bin/env bash
# Run 13: self_consistency + full resolution (max_tiles=12, max_new_tokens=512)
# Waits for Run 12 to finish, then runs.
#
# Differences vs Run 12:
#   max_tiles:      4  → 12  (original resolution)
#   max_new_tokens: 128 → 512 (full Stage 1 descriptions)
#   LLMJudge:       disabled (same as Run 12, keeps GPU budget)
#
# Goal: isolate whether Run 12 quality difference is from self_consistency
# mode itself or from reduced resolution/response length.
#
# Results saved to MAIN REPO (absolute path) — survives worktree deletion.

MAIN_REPO="/home/chikibriki/LoqicQA"
WORKTREE="/home/chikibriki/.lqa_worktrees/run13_sc_full"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/self_consistency_fullres_bb50"
GPU_THRESHOLD_MB=18000  # max_tiles=6 needs ~18-19 GB; well within 22 GB free
POLL_INTERVAL=30
LOG="/tmp/run13_sc_full.log"
SENTINEL="/tmp/run13.done"
RUN12_SENTINEL="/tmp/run12.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_run12() {
    echo "[run13] Waiting for Run 12 to finish (sentinel: ${RUN12_SENTINEL})..."
    while [ ! -f "${RUN12_SENTINEL}" ]; do
        echo "[run13] Run 12 still running, checking again in ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    echo "[run13] Run 12 sentinel detected — proceeding."
}

wait_for_gpu() {
    echo "[run13] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run13] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run13] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 13: self_consistency + full resolution"
echo " max_tiles=12, max_new_tokens=512, no LLMJudge"
echo "════════════════════════════════════════════"

wait_for_run12
wait_for_gpu

SC_COMMIT=$(git -C "${MAIN_REPO}" rev-parse feature/visual_grounding)
mkdir -p "$(dirname "${WORKTREE}")"
if [ ! -d "${WORKTREE}" ]; then
    echo "[run13] Creating worktree at commit ${SC_COMMIT} ..."
    git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${SC_COMMIT}"
else
    echo "[run13] Worktree already exists at ${WORKTREE}"
fi

echo "[run13] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${VENV_PYTHON}" "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_self_consistency_fullres_bb50.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run13] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run13] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
