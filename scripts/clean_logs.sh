#!/bin/bash
# ------------------------------------------------------------
# clean_logs.sh
# Removes generated visualization logs, images, meshes, and videos.
# Run with:  bash clean_logs.sh
# ------------------------------------------------------------

# Default output directory (same as used in AmassVisualizer)
OUT_DIR="amass_vis_out"

echo "[INFO] Cleaning visualization output..."

# Safety: confirm before deleting unless '-y' is passed
if [[ "$1" != "-y" ]]; then
    read -p "This will delete all files in '$OUT_DIR'. Continue? [y/N]: " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "[CANCELLED] No files deleted."
        exit 0
    fi
fi

# Remove all generated subfolders and files
if [ -d "$OUT_DIR" ]; then
    rm -rf "$OUT_DIR"/*
    echo "[OK] Cleared contents of '$OUT_DIR/'."
else
    echo "[INFO] Output directory '$OUT_DIR' does not exist."
fi

# Optionally remove Python __pycache__ and log files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.log" -delete 2>/dev/null

echo "[DONE] Cleanup complete."