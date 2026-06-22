#!/usr/bin/env bash
# Download the SAM 2.1 (Hiera-Tiny) backbone checkpoint used by SCISSR.
# SCISSR keeps this backbone frozen and only trains the lightweight
# Scribble Encoder / SGF / LoRA modules (shipped under checkpoints/scissr/).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

echo "Downloading SAM 2.1 Hiera-Tiny checkpoint..."
wget -c -O sam2.1_hiera_tiny.pt "${BASE_URL}/sam2.1_hiera_tiny.pt"

echo "Done. Saved to ${SCRIPT_DIR}/sam2.1_hiera_tiny.pt"
