#!/bin/bash
# Evaluate clean (non-adversarial) accuracy of defended model.
# Compares base model vs IG-Defense.
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18"
DATASET="cifar10"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
MASK_WHICH="loir"
LAYER_NAME="layer4"
IMPORTANT_DIM=50
NUM_SAMPLES=1000

echo "========================================"
echo "Clean Accuracy Evaluation (IG-Defense)"
echo "========================================"
echo "  Model:          ${ARCH}"
echo "  Dataset:        ${DATASET}"
echo "  Mask method:   ${MASK_WHICH}"
echo "  Layer:          ${LAYER_NAME}"
echo "  k (neurons):   ${IMPORTANT_DIM}"
echo "  Samples:       ${NUM_SAMPLES}"
echo "========================================"
echo ""
echo "[1/2] Base model..."
python eval/autoattack_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which none \
    --n-ex ${NUM_SAMPLES}

echo ""
echo "[2/2] IG-Defense model..."
python eval/autoattack_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which ${MASK_WHICH} \
    --layer-name ${LAYER_NAME} \
    --important-dim ${IMPORTANT_DIM} \
    --n-ex ${NUM_SAMPLES}

echo ""
echo "Done! Results saved in results/"
