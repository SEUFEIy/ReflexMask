#!/bin/bash
# Build consciousness prototypes for ResNet-18 on CIFAR-10.
# This is required only for Conscious IG-Defense (not for basic IG-Defense).
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18_val"
DATASET="cifar10"
LAYER_NAME="layer4"
LAYER_DIM=512
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
OUT_DIR="saved_prototypes"
MAX_SAMPLES=5000
BATCH_SIZE=128

echo "========================================"
echo "Building Consciousness Prototypes"
echo "========================================"
echo "  Model:       ResNet-18"
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
