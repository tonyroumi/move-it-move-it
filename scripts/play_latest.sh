
#!/usr/bin/env bash
# Play the latest checkpoint under logs/.
#
# Usage:
#   scripts/play_latest.sh [--mode local|streaming] \
#       [--camera-type TYPE] \
#       [--joystick-mode none|velocity|point_goal] \
#       [extra play.py args...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/Desktop/IsaacLab}"

MODE="local"
CAMERA_TYPE="facing"
JOYSTICK_MODE="none"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --camera-type|--camera_type)
            CAMERA_TYPE="${2:?--camera-type requires a value}"
            shift 2
            ;;
        --camera-type=*|--camera_type=*)
            CAMERA_TYPE="${1#*=}"
            shift
            ;;
        --joystick-mode)
            JOYSTICK_MODE="${2:?--joystick-mode requires a value}"
            shift 2
            ;;
        --joystick-mode=*)
            JOYSTICK_MODE="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Find the newest numbered checkpoint by modification time.
CHECKPOINT="$(
    find "$REPO_ROOT/logs" \
        -path '*/checkpoints/*.pt' \
        -regex '.*/[0-9]+\.pt' \
        -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-
)"

if [[ -z "$CHECKPOINT" ]]; then
    echo "[ERROR] No checkpoints found under $REPO_ROOT/logs" >&2
    exit 1
fi

# Configure visualization.
case "$MODE" in
    local)
        export LIVESTREAM=0
        export HEADLESS=0
        VIZ_ARGS=(--viz kit)
        ;;
    streaming)
        export LIVESTREAM=2
        export HEADLESS=0
        VIZ_ARGS=()

        if [[ -z "${PUBLIC_IP:-}" ]]; then
            if ! command -v tailscale >/dev/null 2>&1; then
                echo "[ERROR] Tailscale not found. Set PUBLIC_IP manually." >&2
                exit 1
            fi
            PUBLIC_IP="$(tailscale ip -4)"
        fi
        export PUBLIC_IP
        ;;
    *)
        echo "[ERROR] Invalid mode: $MODE (expected local or streaming)" >&2
        exit 1
        ;;
esac

echo "[INFO] Playing latest checkpoint: $CHECKPOINT"
echo "[INFO] Mode: $MODE | Camera: $CAMERA_TYPE"

cd "$REPO_ROOT"
exec "$ISAACLAB_PATH/isaaclab.sh" -p scripts/play.py \
    --checkpoint "$CHECKPOINT" \
    --joystick "$JOYSTICK_MODE" \
    --camera "$CAMERA_TYPE" \
    "${VIZ_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
