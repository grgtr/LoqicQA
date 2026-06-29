#!/bin/bash
# Run baseline GT pipeline for all 5 MVTec LOCO AD classes sequentially.
# Each class gets its own tmux session.

set -e

SCRIPT="scripts/run_pipeline.py"
CONFIG="config_baseline_gt.yaml"
DATA_DIR="/home/chikibriki/LoqicQA/dataset-ninja/"
GT_DIR="results"
OUT_BASE="results_baseline_model"

CLASSES=(
    breakfast_box
    screw_bag
    pushpins
    splicing_connectors
    juice_bottle
)

for CLASS in "${CLASSES[@]}"; do
    GT_FILE="${GT_DIR}/${CLASS}_questions_ground_truth.json"
    OUT_DIR="${OUT_BASE}/${CLASS}"

    if [ ! -f "$GT_FILE" ]; then
        echo "[SKIP] GT file not found: $GT_FILE"
        continue
    fi

    mkdir -p "$OUT_DIR"

    DONE_FLAG="${OUT_DIR}/.done"
    rm -f "$DONE_FLAG"

    echo "[RUN] Starting tmux session: $CLASS"
    tmux new-session -d -s "$CLASS" \
        "python $SCRIPT \
            --class_name $CLASS \
            --config $CONFIG \
            --questions_file $GT_FILE \
            --data_dir $DATA_DIR \
            --output_dir $OUT_DIR \
        && touch '$DONE_FLAG' && echo '[DONE] $CLASS finished' \
        || echo '[FAILED] $CLASS exited with error'; bash"

    echo "[WAIT] Waiting for $CLASS to finish..."
    while [ ! -f "$DONE_FLAG" ]; do
        # Exit early if the session died unexpectedly (crash before writing flag)
        if ! tmux has-session -t "$CLASS" 2>/dev/null; then
            echo "[WARN] Session $CLASS closed without writing done flag"
            break
        fi
        sleep 10
    done

    echo "[OK] $CLASS complete. Results in $OUT_DIR/"
    echo ""
done

echo "========================================"
echo "All classes finished."
echo "Results in: $OUT_BASE/"
echo "========================================"
