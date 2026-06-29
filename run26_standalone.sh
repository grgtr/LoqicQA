#!/usr/bin/env bash
# Run 26: RC4-A + Fix4/5 + InternVL2.5-38B-AWQ (model strength test)
# Branch: feature/rc4a-rc5 @ 65de3fd (AWQ config.json patch + eos_token_id fix)
# Seed 42 — same as run25 for 8B vs 38B comparison.

MAIN_REPO="/home/chikibriki/LoqicQA"
COMMIT="b5e2597"
WORKTREE="/home/chikibriki/.lqa_worktrees/run26_rc4a_38b"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${WORKTREE}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
OUTPUT_DIR="${MAIN_REPO}/results/decomposed_bb50_r26"
GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
LOG="/tmp/run26_decomposed.log"
SENTINEL="/tmp/run26.done"

rm -f "${SENTINEL}"
mkdir -p "${OUTPUT_DIR}"

gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{ sum += $1 } END { print sum }'
}

wait_for_gpu() {
    echo "[run26] Waiting for ≥${GPU_THRESHOLD_MB} MB free GPU memory..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[run26] GPU free: ${FREE} MB"
        if [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ]; then
            echo "[run26] GPU ready."
            return
        fi
        sleep "${POLL_INTERVAL}"
    done
}

echo ""
echo "════════════════════════════════════════════"
echo " RUN 26: RC4-A + Fix4/5 + InternVL2.5-38B-AWQ"
echo " Commit: ${COMMIT}"
echo "════════════════════════════════════════════"

wait_for_gpu

mkdir -p "$(dirname "${WORKTREE}")"
if [ -d "${WORKTREE}" ]; then
    echo "[run26] Removing stale worktree..."
    git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || rm -rf "${WORKTREE}"
    git -C "${MAIN_REPO}" worktree prune
fi

echo "[run26] Creating worktree at commit ${COMMIT}..."
git -C "${MAIN_REPO}" worktree add --detach "${WORKTREE}" "${COMMIT}"

# Copy 38B config into worktree
cp "${MAIN_REPO}/config_decomposed_bb50_38b.yaml" "${WORKTREE}/config_decomposed_bb50_38b.yaml"

echo "[run26] Starting at $(date)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
"${VENV_PYTHON}" -u "${SCRIPT}" \
    --class_name breakfast_box \
    --config "${WORKTREE}/config_decomposed_bb50_38b.yaml" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --save_questions \
    2>&1 | tee "${LOG}"

PIPELINE_EXIT=${PIPESTATUS[0]}
echo "[run26] Pipeline exited with code ${PIPELINE_EXIT}"

touch "${SENTINEL}"
git -C "${MAIN_REPO}" worktree remove --force "${WORKTREE}" || true

echo "[run26] Finished at $(date). Log: ${LOG}. Results: ${OUTPUT_DIR}"
