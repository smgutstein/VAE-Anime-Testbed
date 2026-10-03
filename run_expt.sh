#!/bin/bash

LOG_FILE="run_experiments.txt"

# Start with a fresh log file
> "$LOG_FILE"

configs=(
    config_greedy_beta.ini
    config_beta_1.ini
    config_beta_10.ini
    config_beta_100.ini
    config_beta_1000.ini
)

#configs=(
#    config_beta_1_test.ini
#    config_beta_10_test.ini
#)

# Usage:
#   ./run_expt.sh                         # Run each config once as written.
#   ./run_expt.sh 1066 1067 1068          # Run the full config series for
#                                          # each of the three seeds.
#
# Seeds are command-line arguments rather than values stored here so that a new
# batch does not require editing this script or any of the tracked sample
# configs. With no arguments, the old behavior is preserved: every config is
# run once using the seed already recorded in that file.
seeds=("$@")

# Must match EXIT_TRAINING_DIVERGED in VAE_Anime_Train.py. A run that exits
# with this status diverged (trip-wire limit or non-finite loss) after saving
# its run summary and diagnostics, so the batch moves on to the next config.
EXIT_TRAINING_DIVERGED=3
diverged_runs=()

# Reject bad values before starting the first potentially long-running job.
for seed in "${seeds[@]}"; do
    # Check whether this seed contains anything other than digits 0 through 9.
    # If it does, reject it before starting any experiments.
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed '$seed': seeds must be non-negative integers." >&2
        exit 2
    fi
done

run_config() {
    local config="$1"
    local seed="${2:-}"
    local config_arg="$config"
    local temp_config=""

    # Check whether a seed was passed to this function. A non-empty seed means
    # this run needs a temporary config containing the requested override.
    if [ -n "$seed" ]; then
        local source_config="configs/$config"
        local seed_count

        # Check whether the named config exists as a regular file in configs/.
        if [ ! -f "$source_config" ]; then
            echo "Config file not found: $source_config" >&2
            exit 2
        fi

        # Replacing an ambiguous or missing setting could silently invalidate
        # the experiment metadata, so require exactly one seed assignment.
        seed_count=$(grep -Ec '^[[:space:]]*seed[[:space:]]*=' "$source_config")

        # Check whether the config contains exactly one seed assignment. Zero
        # means there is nothing to replace; more than one is ambiguous.
        if [ "$seed_count" -ne 1 ]; then
            echo "Expected exactly one seed setting in $source_config; found $seed_count." >&2
            exit 2
        fi

        # Override the seed in a temporary copy. This leaves the tracked
        # config untouched, while ExperimentRun.create() still copies the
        # effective config into expt_N/config.ini for reproducibility.
        temp_config=$(mktemp --suffix=.ini)
        sed -E \
            "s/^([[:space:]]*seed[[:space:]]*=[[:space:]]*).*/\\1$seed/" \
            "$source_config" > "$temp_config"
        config_arg="$temp_config"
    fi

    {
        echo "========================================"
        echo "Running: $config"

        # Check whether this run is using a command-line seed override. If so,
        # include that seed in the timing log.
        if [ -n "$seed" ]; then
            echo "Seed: $seed"
        fi

        echo "Started: $(date)"
        echo "========================================"
    } >> "$LOG_FILE"

    SECONDS=0

    # Output from the Python program goes normally to the terminal
    python VAE_Anime_Train.py --config_file "$config_arg"
    status=$?

    # Training has already copied the effective config into its experiment
    # directory, so the temporary input file is no longer needed.
    #
    # Check whether this run created a temporary config. Runs without a seed
    # argument use the original config directly and therefore have no temp file.
    if [ -n "$temp_config" ]; then
        rm -f "$temp_config"
    fi

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

    # Check whether training stopped because the model diverged. That is an
    # experimental outcome, not a program error: the run has already saved its
    # artifacts, so record it and let the loop continue with the next config.
    if [ "$status" -eq "$EXIT_TRAINING_DIVERGED" ]; then
        local label="$config"
        if [ -n "$seed" ]; then
            label="$config (seed $seed)"
        fi
        diverged_runs+=("$label")
        echo "Run diverged: $label. Continuing with next experiment." | tee -a "$LOG_FILE"
        echo >> "$LOG_FILE"
        return 0
    fi

    # Check whether the Python process returned any other nonzero exit status,
    # meaning training failed. Do not continue to later experiments after a failure.
    if [ "$status" -ne 0 ]; then
        echo "Run failed. Stopping experiment sequence." >> "$LOG_FILE"
        echo "Run failed. Stopping experiment sequence." 
        exit "$status"
    fi
}

# Check whether the script was called without seed arguments. With no seeds,
# run each config once as written; otherwise run the full series for every seed.
if [ "${#seeds[@]}" -eq 0 ]; then
    for config in "${configs[@]}"; do
        run_config "$config"
    done
else
    # Keep each seed's five experiments together: all configs for the first
    # seed, followed by all configs for the next seed, and so on.
    for seed in "${seeds[@]}"; do
        for config in "${configs[@]}"; do
            run_config "$config" "$seed"
        done
    done
fi

# Check whether any runs diverged. They did not stop the batch, but they should
# not be reported as successful either.
if [ "${#diverged_runs[@]}" -eq 0 ]; then
    echo "All runs completed successfully." >> "$LOG_FILE"
else
    {
        echo "All runs finished; ${#diverged_runs[@]} diverged:"
        printf "    %s\n" "${diverged_runs[@]}"
    } | tee -a "$LOG_FILE"
fi
