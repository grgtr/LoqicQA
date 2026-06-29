#!/usr/bin/env bash
# Run 23: RC4-A + Fix1/2/3 (re.findall, no multi-img SQ, main-Q anchor)
# Branch: feature/rc4a-rc5 @ ffa8cc4
# Seed 42 — same as run21 for apples-to-apples comparison.

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT="ffa8cc4"
WORKTREE="/home/chikibriki/.lqa_worktrees/run23_rc4a_v2"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${WORKTREE}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/decomposed_bb50_r23"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run23_decomposed.log"
SENTINEL="/tmp/run23.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{ sum += $1 } END { print sum }'
}

wait_for_gpu() {
    echo "[run23] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run23] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run23] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 23: RC4-A + Fix1/2/3 (re.findall, no multi-img SQ, main-Q anchor)"
echo " Commit: ${COMMIT}"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run23] Removing stale worktree..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run23] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

echo "[run23] Starting at $(date)"
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
echo "[run23] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run23] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
