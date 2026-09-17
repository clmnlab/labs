#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm -v "${BIDS_DIR:-/mnt/ext4/KBSI/preproc/bids}:/data:ro" bids/validator:3.0.1 /data
