#!/bin/bash
# Generate LOIR (Leave-One-Out Importance Ranking) for WideResNet-34-10 on CIFAR-10.
# This is a one-time offline step. Results are cached after first run.
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="wrn34_10"
DATASET="cifar10"
LAYER_NAME="block3"
CHECKPOINT="TRADES-AWP_cifar10_linf_wrn34-10.pt"
OUT_DIR="saved_rankings"
BATCH_SIZE=128

echo "========================================"
echo "Generating LOIR Rankings"
echo "========================================"
echo "  Model:       WideResNet-34-10"
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
