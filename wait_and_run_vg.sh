#!/usr/bin/env bash
# Wait until ≥30 GB GPU memory is free, then launch the visual grounding run
# in a persistent tmux session.
#
# Usage:
#   bash wait_and_run_vg.sh          # runs in foreground, Ctrl+C to cancel
#   nohup bash wait_and_run_vg.sh &  # runs in background

set -euo pipefail

SESSION="vg_bb50"
WINDOW="run8"
THRESHOLD_MB=30000   # 30 GB
POLL_INTERVAL=30     # seconds between checks
LOGFILE="/tmp/run8_vg_bb50.log"
WORKDIR="/home/chikibriki/LoqicQA"

CMD=".venv/bin/python3 scripts/run_pipeline.py \
    --class_name breakfast_box \
    --config config_visual_grounding_bb50.yaml \
    --output_dir results/visual_grounding_bb50 \
    --data_dir /home/chikibriki/LoqicQA/dataset-ninja/ \
    --seed 42 --save_questions \
    2>&1 | tee ${LOGFILE}"

echo "[wait_and_run] Waiting for ${THRESHOLD_MB} MB free GPU memory..."
echo "[wait_and_run] Polling every ${POLL_INTERVAL}s. Log → ${LOGFILE}"

while true; do
    # Sum free memory across all GPUs (MiB)
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
              | awk '{sum += $1} END {print sum}')

    echo "[$(date '+%H:%M:%S')] GPU free: ${FREE_MB} MB / needed: ${THRESHOLD_MB} MB"

    if [ "${FREE_MB}" -ge "${THRESHOLD_MB}" ]; then
        echo "[wait_and_run] Threshold met — launching tmux session '${SESSION}:${WINDOW}'"

        # Create session if it doesn't exist, otherwise add a new window
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
            tmux new-window -t "${SESSION}" -n "${WINDOW}"
        else
            tmux new-session -d -s "${SESSION}" -n "${WINDOW}"
        fi

        tmux send-keys -t "${SESSION}:${WINDOW}" "cd ${WORKDIR} && ${CMD}" Enter

        echo "[wait_and_run] Run started. Attach with: tmux attach -t ${SESSION}:${WINDOW}"
        echo "[wait_and_run] Live log: tail -f ${LOGFILE}"
        exit 0
    fi

    sleep "${POLL_INTERVAL}"
done
