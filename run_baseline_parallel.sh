#!/bin/bash
# Run baseline GT pipeline for all 5 classes IN PARALLEL.
# Requires: single A100 80GB (5 x ~16GB InternVL2.5-8B instances fit).
# Each class gets its own tmux session; script waits until all are done.

set -e

SCRIPT="scripts/run_pipeline.py"
CONFIG="config_baseline_gt.yaml"
DATA_DIR="/home/chikibriki/LoqicQA/dataset-ninja/"
GT_DIR="results"
OUT_BASE="results_baseline_model"

CLASSES=(
    screw_bag
    pushpins
    splicing_connectors
    juice_bottle
)

# ------------------------------------------------------------------ #
# Launch all classes at once
# ------------------------------------------------------------------ #
for CLASS in "${CLASSES[@]}"; do
    GT_FILE="${GT_DIR}/${CLASS}_questions_ground_truth.json"
    OUT_DIR="${OUT_BASE}/${CLASS}"
    DONE_FLAG="${OUT_DIR}/.done"

    if [ ! -f "$GT_FILE" ]; then
        echo "[SKIP] GT file not found: $GT_FILE"
        continue
    fi

    mkdir -p "$OUT_DIR"
    rm -f "$DONE_FLAG"

    echo "[LAUNCH] $CLASS → $OUT_DIR"
    tmux new-session -d -s "$CLASS" \
        "python $SCRIPT \
            --class_name $CLASS \
            --config $CONFIG \
            --questions_file $GT_FILE \
            --data_dir $DATA_DIR \
            --output_dir $OUT_DIR \
        && touch '$DONE_FLAG' && echo '[DONE] $CLASS finished' \
        || echo '[FAILED] $CLASS exited with error'; bash"
done

echo ""
echo "All sessions launched. Waiting for all classes to finish..."
echo "Monitor with:  tmux ls"
echo "Attach to:     tmux attach -t <class_name>   (Ctrl+B D to detach)"
echo ""

# ------------------------------------------------------------------ #
# Wait until all done flags appear
# ------------------------------------------------------------------ #
while true; do
    ALL_DONE=true
    PENDING=()

    for CLASS in "${CLASSES[@]}"; do
        DONE_FLAG="${OUT_BASE}/${CLASS}/.done"
        GT_FILE="${GT_DIR}/${CLASS}_questions_ground_truth.json"
        [ ! -f "$GT_FILE" ] && continue   # was skipped at launch

        if [ ! -f "$DONE_FLAG" ]; then
            # Check if session crashed without writing the flag
            if ! tmux has-session -t "$CLASS" 2>/dev/null; then
                echo "[WARN] Session $CLASS closed without done flag — likely crashed"
            else
                ALL_DONE=false
                PENDING+=("$CLASS")
            fi
        fi
    done

    if $ALL_DONE; then
        break
    fi

    echo "[$(date '+%H:%M:%S')] Still running: ${PENDING[*]}"
    sleep 30
done

echo ""
echo "========================================"
echo "All classes finished."
echo "Results in: $OUT_BASE/"
for CLASS in "${CLASSES[@]}"; do
    RESULTS_FILE="${OUT_BASE}/${CLASS}/${CLASS}_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  OK  $CLASS → $RESULTS_FILE"
    else
        echo "  ERR $CLASS → results file missing"
    fi
done
echo "========================================"
