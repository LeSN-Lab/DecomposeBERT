#!/bin/bash

. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate DecomposeTransformer

# Default parameters
NAME="OSDG"
DEVICE="cuda:0"
CHECKPOINT=None
BATCH_SIZE=32
NUM_WORKERS=16
NUM_SAMPLES=64
MAGNITUDE_SPARSITY_RATIO=0.1
CI_SPARSITY_RATIO=0.6
TI_RECOVERY_RATIO=0.1
INCLUDE_LAYERS="attention intermediate output"
EXCLUDE_LAYERS=None

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) NAME="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --magnitude_sparsity_ratio) MAGNITUDE_SPARSITY_RATIO="$2"; shift ;;
        --ci_sparsity_ratio) CI_SPARSITY_RATIO="$2"; shift ;;
        --ti_recovery_ratio) TI_RECOVERY_RATIO="$2"; shift ;;
        --include_layers) INCLUDE_LAYERS="$2"; shift ;;
        --exclude_layers) EXCLUDE_LAYERS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
cd ../Getting_Started

echo "Running Python script"

python3 ./CITI.py \
    --name "$NAME" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --num_samples "$NUM_SAMPLES" \
    --magnitude_sparsity_ratio "$MAGNITUDE_SPARSITY_RATIO" \
    --ci_sparsity_ratio "$CI_SPARSITY_RATIO" \
    --ti_recovery_ratio "$TI_RECOVERY_RATIO" \
    --include_layers $INCLUDE_LAYERS \
    --exclude_layers $EXCLUDE_LAYERS \
echo "Python script finished"
