#!/bin/bash
# Generate LOIR (Leave-One-Out Importance Ranking) for ResNet-50 on ImageNet.
# This is a one-time offline step. ImageNet has 2048 neurons in layer4,
# which takes longer. For parallel processing, use --start-dim and --end-dim flags.
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn50"
DATASET="imagenet"
LAYER_NAME="layer4"
CHECKPOINT="imagenet_model_weights_4px.pth.tar"
OUT_DIR="saved_rankings"
BATCH_SIZE=100

echo "========================================"
echo "Generating LOIR Rankings"
echo "========================================"
echo "  Model:       ResNet-50"
echo "  Dataset:     ImageNet"
echo "  Layer:       ${LAYER_NAME} (2048 neurons)"
echo "  Checkpoint:  checkpoints/${CHECKPOINT}"
echo "  Output:      ${OUT_DIR}/${LAYER_NAME}/"
echo ""
echo "NOTE: ImageNet has 2048 neurons. This may take several hours."
echo "For parallel processing, split by --start-dim and --end-dim."
echo "See REPRODUCE.md for details."
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
