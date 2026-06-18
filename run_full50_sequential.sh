#!/usr/bin/env bash
# run_full50_sequential.sh — Sequential FULL-pipeline runs on 50-image subsets
# Classes: juice_bottle -> pushpins -> screw_bag -> splicing_connectors
# Model: InternVL2.5-8B | FULL pipeline (Stages 1-4, auto question generation)
# Subset: 25 good + 25 logical anomalies per class (configs/full50_*.yaml)
# Normality definitions: rewritten (logicqa/data/normality_definitions.py)
# Runs from MAIN repo (uses current working tree, incl. rewritten definitions).
#
# Usage:
#   bash run_full50_sequential.sh            # launch in persistent tmux session
#   bash run_full50_sequential.sh --attach   # launch and attach
#
# Session: lqa_full4   Logs: /tmp/full_<class>.log   Sentinels: /tmp/full_<class>.done

MAIN_REPO="/home/chikibriki/LoqicQA"
VENV_PYTHON="${MAIN_REPO}/.venv/bin/python3"
SCRIPT="${MAIN_REPO}/scripts/run_pipeline.py"
DATA_DIR="${MAIN_REPO}/dataset-ninja/"
CONFIGS_DIR="${MAIN_REPO}/configs"

GPU_THRESHOLD_MB=18000
POLL_INTERVAL=30
TMUX_SESSION="lqa_full4"

# ── tmux launcher (persistent: session survives detach; stays open at end) ────
ATTACH=0
for arg in "$@"; do [[ "${arg}" == "--attach" ]] && ATTACH=1; done

if [[ -z "${TMUX:-}" ]] && [[ "${1:-}" != "--_inside_tmux" ]]; then
    echo "[launcher] Starting tmux session '${TMUX_SESSION}'..."
    tmux new-session -d -s "${TMUX_SESSION}" -x 220 -y 50 \
        "bash ${MAIN_REPO}/run_full50_sequential.sh --_inside_tmux; exec bash"
    echo "[launcher] Started."
    echo "  Attach:  tmux attach -t ${TMUX_SESSION}"
    echo "  Monitor: tail -f /tmp/full_juice_bottle.log"
    [[ "${ATTACH}" -eq 1 ]] && tmux attach -t "${TMUX_SESSION}"
    exit 0
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk '{sum += $1} END {print sum}'
}

wait_for_gpu() {
    local label="$1"
    echo "[queue] Waiting for >=${GPU_THRESHOLD_MB} MB free GPU (${label})..."
    while true; do
        FREE=$(gpu_free_mb)
        echo "[queue] GPU free: ${FREE} MB (need ${GPU_THRESHOLD_MB})"
        [ "${FREE}" -ge "${GPU_THRESHOLD_MB}" ] && { echo "[queue] GPU ready for ${label}."; return; }
        sleep "${POLL_INTERVAL}"
    done
}

# ── Run function (FULL pipeline: no --questions_file) ────────────────────────
rm -f /tmp/full_juice_bottle.done /tmp/full_pushpins.done \
      /tmp/full_screw_bag.done /tmp/full_splicing_connectors.done

run_one() {
    local CLASS="$1"
    local CONFIG="${CONFIGS_DIR}/full50_${CLASS}.yaml"
    local OUTPUT_DIR="${MAIN_REPO}/results/full50_${CLASS}"
    local LOG="/tmp/full_${CLASS}.log"
    local SENTINEL="/tmp/full_${CLASS}.done"

    rm -f "${SENTINEL}"
    mkdir -p "${OUTPUT_DIR}"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " FULL50: ${CLASS}"
    echo " Config: ${CONFIG}"
    echo " Output: ${OUTPUT_DIR}    Log: ${LOG}"
    echo "════════════════════════════════════════════════════════"

    wait_for_gpu "${CLASS}"

    echo "[${CLASS}] Starting at $(date)"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTHONUNBUFFERED=1

    "${VENV_PYTHON}" -u "${SCRIPT}" \
        --class_name "${CLASS}" \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        --seed 42 \
        --save_questions \
        2>&1 | tee "${LOG}"

    local EXIT_CODE=${PIPESTATUS[0]}
    echo "[${CLASS}] Pipeline exited with code ${EXIT_CODE} at $(date)"
    touch "${SENTINEL}"
    return ${EXIT_CODE}
}

# ── Sequential runs ──────────────────────────────────────────────────────────
run_one "juice_bottle"
run_one "pushpins"
run_one "screw_bag"
run_one "splicing_connectors"

echo ""
echo "════════════════════════════════════════════════════════"
echo " ALL FULL50 RUNS COMPLETE"
echo "════════════════════════════════════════════════════════"
echo "  results/full50_juice_bottle/         log: /tmp/full_juice_bottle.log"
echo "  results/full50_pushpins/             log: /tmp/full_pushpins.log"
echo "  results/full50_screw_bag/            log: /tmp/full_screw_bag.log"
echo "  results/full50_splicing_connectors/  log: /tmp/full_splicing_connectors.log"
