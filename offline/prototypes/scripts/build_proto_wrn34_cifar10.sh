#!/bin/bash
# Build consciousness prototypes for WideResNet-34-10 on CIFAR-10.
# This is required only for Conscious IG-Defense (not for basic IG-Defense).
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="wrn34_10"
DATASET="cifar10"
LAYER_NAME="block3"
LAYER_DIM=640
CHECKPOINT="TRADES-AWP_cifar10_linf_wrn34-10.pt"
OUT_DIR="saved_prototypes"
MAX_SAMPLES=5000
BATCH_SIZE=128

echo "========================================"
echo "Building Consciousness Prototypes"
echo "========================================"
echo "  Model:       WideResNet-34-10"
echo "  Dataset:     CIFAR-10"
echo "  Layer:       ${LAYER_NAME} (dim=${LAYER_DIM})"
echo "  Checkpoint:  checkpoints/${CHECKPOINT}"
echo "  Max samples: ${MAX_SAMPLES}"
echo "  Output:      ${OUT_DIR}/${ARCH}/${DATASET}/${LAYER_NAME}/"
echo "========================================"

python offline/prototypes/build_prototypes.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --layer-names ${LAYER_NAME} \
    --layer-dim ${LAYER_DIM} \
    --output-dir ${OUT_DIR} \
    --max-samples ${MAX_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --use-train

echo ""
echo "Done! Prototypes saved to: ${OUT_DIR}/${ARCH}/${DATASET}/${LAYER_NAME}/"
