#!/bin/bash
# Generate LOIR (Leave-One-Out Importance Ranking) for ResNet-18 on CIFAR-10.
# This is a one-time offline step. Results are cached after first run.
set -e

# Navigate to project root: offline/rankings/scripts -> ReflexMask/
cd "$(dirname "$0")/../../.."
# Ensure project root is on PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18_val"
DATASET="cifar10"
LAYER_NAME="layer4"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
OUT_DIR="saved_rankings"
BATCH_SIZE=128

echo "========================================"
echo "Generating LOIR Rankings"
echo "========================================"
echo "  Model:       ResNet-18"
echo "  Dataset:     CIFAR-10"
echo "  Layer:       ${LAYER_NAME}"
echo "  Checkpoint:  checkpoints/${CHECKPOINT}"
echo "  Output:      ${OUT_DIR}/${LAYER_NAME}/"
echo "========================================"

python offline/rankings/loir_rankings.py \
    --arch ${ARCH} \
    --load-model ${CHECKPOINT} \
    --dataset ${DATASET} \
    --layer-name ${LAYER_NAME} \
    --out-dir ${OUT_DIR} \
    --batch-size ${BATCH_SIZE}

echo ""
echo "Done! Rankings saved to: ${OUT_DIR}/${LAYER_NAME}/"
