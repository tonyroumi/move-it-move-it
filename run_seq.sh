
#!/usr/bin/env bash

set -uo pipefail

ISAACLAB="$HOME/Desktop/IsaacLab/isaaclab.sh"

COMMANDS=(
    # "scripts/train.py --algo AMP --manifest configs/manifests/boxing_manifest.yaml --name box"
    "scripts/train.py --algo AMP --manifest configs/manifests/goal_jump_manifest.yaml --name goal_jump2"
    "scripts/train.py --algo AMP --manifest configs/manifests/dance_manifest.yaml --name dance2"
    # "scripts/train.py --algo AMP --manifest configs/manifests/dance_multiple_manifest.yaml --name dance_multiple"

)

TOTAL=${#COMMANDS[@]}

for i in "${!COMMANDS[@]}"; do
    echo "========================================"
    echo "Running experiment $((i + 1))/$TOTAL"
    echo "${COMMANDS[$i]}"
    echo "========================================"

    if "$ISAACLAB" -p ${COMMANDS[$i]}; then
        echo "Experiment $((i + 1)) completed."
    else
        echo "Experiment $((i + 1)) failed."
        exit 1
    fi
done

echo "All experiments completed."
