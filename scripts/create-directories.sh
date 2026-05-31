#!/usr/bin/env bash
# Create the autoboat data store directory structure.
# Run on the boat Pi (or anywhere you want a data root).
# Override the location with the AUTOBOAT_DATA environment variable.

set -euo pipefail

DATA_ROOT="${AUTOBOAT_DATA:-$HOME/autoboat-data}"

mkdir -p "$DATA_ROOT/captures"   # calibration still images
mkdir -p "$DATA_ROOT/sessions"   # operational run recordings

echo "Data store ready at: $DATA_ROOT"
echo "  captures/   calibration still images (curated JPEGs)"
echo "  sessions/   operational run recordings (video + CSV streams)"
