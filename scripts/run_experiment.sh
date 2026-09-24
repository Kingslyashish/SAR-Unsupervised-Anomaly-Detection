#!/usr/bin/env bash
set -euo pipefail

: "${SAR_BASE:?Set SAR_BASE to the local project-data directory}"
export SAR_BASE
export SAR_OUT="${SAR_OUT:-./results/checkpoints}"
python src/unified_sar_experiment.py
