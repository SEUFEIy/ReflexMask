#!/bin/bash
# Evaluate IG-Defense with AutoAttack (L-inf, epsilon=8/255).
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18"
DATASET="cifar10"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
MASK_WHICH="loir"
LAYER_NAME="layer4"
IMPORTANT_DIM=50
N_EX=1000
VERSION="standard"

echo "========================================"
echo "AutoAttack Evaluation (IG-Defense)"
echo "========================================"
echo "  Model:          ${ARCH}"
echo "  Dataset:        ${DATASET}"
echo "  Epsilon:        8/255"
echo "  Mask method:   ${MASK_WHICH}"
echo "  Layer:          ${LAYER_NAME}"
echo "  k (neurons):   ${IMPORTANT_DIM}"
echo "  Samples:       ${N_EX}"
echo "========================================"
echo ""

echo "[1/3] Base model (no defense)..."
python eval/autoattack_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which none \
    --n-ex ${N_EX} \
    --version ${VERSION}

echo ""
echo "[2/3] IG-Defense (CDIR)..."
python eval/autoattack_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which cdir \
    --layer-name ${LAYER_NAME} \
    --important-dim ${IMPORTANT_DIM} \
    --rs --rs-sigma 4 --rs-nsmooth 1 \
    --n-ex ${N_EX} \
    --version ${VERSION}

echo ""
echo "[3/3] IG-Defense (LOIR)..."
python eval/autoattack_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which ${MASK_WHICH} \
    --layer-name ${LAYER_NAME} \
    --important-dim ${IMPORTANT_DIM} \
    --rs --rs-sigma 4 --rs-nsmooth 1 \
    --n-ex ${N_EX} \
    --version ${VERSION}

echo ""
echo "Done! Results saved in results/"
