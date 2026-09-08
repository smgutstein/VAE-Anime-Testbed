#!/bin/bash

LOG_FILE="run_experiments.txt"

# Start with a fresh log file
> "$LOG_FILE"

configs=(
#    config_greedy_beta.ini
    config_beta_1.ini
    config_beta_10.ini
    config_beta_100.ini
    config_beta_1000.ini
)

for config in "${configs[@]}"; do
    {
        echo "========================================"
        echo "Running: $config"
        echo "Started: $(date)"
        echo "========================================"
    } >> "$LOG_FILE"

    SECONDS=0

    # Output from the Python program goes normally to the terminal
    python VAE_Anime_Train.py --config "$config"
    status=$?

    elapsed=$SECONDS
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    seconds=$((elapsed % 60))

    {
        printf "Elapsed time: %02d:%02d:%02d\n" \
            "$hours" "$minutes" "$seconds"
        echo "Exit status: $status"
        echo "Finished: $(date)"
        echo
    } >> "$LOG_FILE"

    if [ "$status" -ne 0 ]; then
        echo "Run failed. Stopping experiment sequence." >> "$LOG_FILE"
        echo "Run failed. Stopping experiment sequence." 
        exit "$status"
    fi
done

echo "All runs completed successfully." >> "$LOG_FILE"

