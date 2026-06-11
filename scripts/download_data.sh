#!/usr/bin/env bash
# Download and extract the GroceryVision MPR dataset (CC-BY-NC 4.0).
#
# Usage: bash scripts/download_data.sh [DEST_DIR]   # default DEST_DIR: data
#
# Disk space: ~8 GB for the archive plus ~24 GB extracted (~32 GB free).
# The archive can be deleted after extraction.
set -euo pipefail

URL="https://d35668us06lihg.cloudfront.net/mpr_challenge.tar.gz"
EXPECTED_SIZE=8521632328
DEST="${1:-data}"
ARCHIVE="$DEST/mpr_challenge.tar.gz"

if [ -d "$DEST/mpr_challenge/appearance_based" ]; then
    echo "Dataset already extracted at $DEST/mpr_challenge — nothing to do."
    exit 0
fi

mkdir -p "$DEST"

if [ ! -f "$ARCHIVE" ] || [ "$(wc -c < "$ARCHIVE")" -ne "$EXPECTED_SIZE" ]; then
    echo "Downloading $URL (~8 GB)..."
    curl -L --fail --retry 3 -C - -o "$ARCHIVE" "$URL"
fi

ACTUAL_SIZE="$(wc -c < "$ARCHIVE")"
if [ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]; then
    echo "ERROR: archive size $ACTUAL_SIZE does not match expected $EXPECTED_SIZE bytes." >&2
    echo "The download may be incomplete or the upstream file may have changed." >&2
    exit 1
fi

echo "Extracting to $DEST/ (~24 GB)..."
tar -xzf "$ARCHIVE" -C "$DEST"

echo "Done. Dataset root: $DEST/mpr_challenge/appearance_based"
echo "You can delete $ARCHIVE to reclaim ~8 GB."
