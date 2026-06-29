#!/bin/bash
# Pipeline 2 (full Stages 1-4) for all classes in parallel tmux sessions.
# Each session independently polls GPU free memory and acquires a launch lock
# so that at most one model loads at a time (prevents OOM during model init).
#
# Memory thresholds:
#   Without LangSAM (breakfast_box, screw_bag, juice_bottle): 17 GB
#   With    LangSAM (pushpins, splicing_connectors):          21 GB

set -e

CONFIG="config_baseline_full.yaml"
DATA_DIR="/home/chikibriki/LoqicQA/dataset-ninja/"
OUT_BASE="results_baseline_full"

# format: "class_name:threshold_MiB"
CLASSES=(
    "breakfast_box:25000"
    "screw_bag:25000"
    "juice_bottle:25000"
    "pushpins:27000"
    "splicing_connectors:27000"
)

# Shared lock file: held for LOCK_HOLD_SECONDS after launch
# so the next class waits until InternVL is fully loaded into VRAM.
LOCK_FILE="/tmp/logicqa_gpu_launch.lock"
LOCK_HOLD_SECONDS=123
CHECK_INTERVAL=77

mkdir -p "$OUT_BASE"
touch "$LOCK_FILE"

# ------------------------------------------------------------------ #
# Write per-class wait-and-run scripts to /tmp
# ------------------------------------------------------------------ #
for entry in "${CLASSES[@]}"; do
    CLASS="${entry%%:*}"
    THRESHOLD="${entry##*:}"
    OUT_DIR="${OUT_BASE}/${CLASS}"
    DONE_FLAG="${OUT_DIR}/.done"
    SCRIPT="/tmp/logicqa_run_${CLASS}.sh"

    mkdir -p "$OUT_DIR"
    rm -f "$DONE_FLAG"

    cat > "$SCRIPT" << SCRIPT_EOF
#!/bin/bash
CLASS="$CLASS"
THRESHOLD=$THRESHOLD
OUT_DIR="$OUT_DIR"
DONE_FLAG="$DONE_FLAG"
LOCK_FILE="$LOCK_FILE"
LOCK_HOLD_SECONDS=$LOCK_HOLD_SECONDS
CHECK_INTERVAL=$CHECK_INTERVAL

echo "[\$(date '+%H:%M:%S')] [\$CLASS] Waiting for \${THRESHOLD} MiB free on GPU..."

while true; do
    FREE_MIB=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "[\$(date '+%H:%M:%S')] [\$CLASS] GPU free: \${FREE_MIB} MiB / need: \${THRESHOLD} MiB"

    if [ "\${FREE_MIB}" -ge "\${THRESHOLD}" ]; then
        # Try to acquire launch lock (non-blocking)
        exec 9>"\$LOCK_FILE"
        if flock -n 9; then
            echo "[\$(date '+%H:%M:%S')] [\$CLASS] Got launch lock — launching!"
            # Release lock after LOCK_HOLD_SECONDS in background
            # (gives time for InternVL to load and claim VRAM)
            ( sleep \$LOCK_HOLD_SECONDS; flock -u 9 ) &
            break
        else
            echo "[\$(date '+%H:%M:%S')] [\$CLASS] Lock busy (another model loading), waiting..."
        fi
    fi

    sleep \$CHECK_INTERVAL
done

# ---- Run pipeline ----
.venv/bin/python3 scripts/run_pipeline.py \\
    --class_name "\$CLASS" \\
    --config "$CONFIG" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "\$OUT_DIR" \\
    --save_questions \\
    && touch "\$DONE_FLAG" \\
    && echo "[\$(date '+%H:%M:%S')] [DONE] \$CLASS finished" \\
    || echo "[\$(date '+%H:%M:%S')] [FAILED] \$CLASS exited with error"
SCRIPT_EOF

    chmod +x "$SCRIPT"
done

# ------------------------------------------------------------------ #
# Launch all tmux sessions
# ------------------------------------------------------------------ #
for entry in "${CLASSES[@]}"; do
    CLASS="${entry%%:*}"
    SCRIPT="/tmp/logicqa_run_${CLASS}.sh"

    # Kill stale session if exists
    tmux kill-session -t "full_${CLASS}" 2>/dev/null || true

    echo "[LAUNCH] tmux session: full_${CLASS}"
    tmux new-session -d -s "full_${CLASS}" "bash $SCRIPT; bash"
done

echo ""
echo "All sessions launched. Monitor:"
echo "  tmux ls"
echo "  tmux attach -t full_<class>   (Ctrl+B D to detach)"
echo ""

# ------------------------------------------------------------------ #
# Wait for all done flags
# ------------------------------------------------------------------ #
echo "Polling for completion every 60s..."
while true; do
    ALL_DONE=true
    PENDING=()

    for entry in "${CLASSES[@]}"; do
        CLASS="${entry%%:*}"
        DONE_FLAG="${OUT_BASE}/${CLASS}/.done"

        if [ ! -f "$DONE_FLAG" ]; then
            if ! tmux has-session -t "full_${CLASS}" 2>/dev/null; then
                echo "[WARN] Session full_${CLASS} closed without done flag"
            else
                ALL_DONE=false
                PENDING+=("$CLASS")
            fi
        fi
    done

    $ALL_DONE && break

    echo "[$(date '+%H:%M:%S')] Still running: ${PENDING[*]}"
    sleep 60
done

echo ""
echo "========================================"
echo "All classes finished. Results in: $OUT_BASE/"
for entry in "${CLASSES[@]}"; do
    CLASS="${entry%%:*}"
    Q_FILE="${OUT_BASE}/${CLASS}/${CLASS}_questions.json"
    R_FILE="${OUT_BASE}/${CLASS}/${CLASS}_results.json"
    q_status="MISSING"; r_status="MISSING"
    [ -f "$Q_FILE" ] && q_status="OK"
    [ -f "$R_FILE" ] && r_status="OK"
    echo "  $CLASS — questions: $q_status | results: $r_status"
done
echo "========================================"
