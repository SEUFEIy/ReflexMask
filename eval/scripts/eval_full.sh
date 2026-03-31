#!/bin/bash
# Full evaluation: clean accuracy + FGSM + PGD-10/20/50.
# Produces a comparison table of base vs defended model.
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18"
DATASET="cifar10"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
LAYER_NAME="layer4"
MASK_WHICH="loir"
IMPORTANT_DIM=50
NUM_SAMPLES_CLEAN=5000
NUM_SAMPLES_ADV=1000

echo "========================================"
echo "Full Evaluation Pipeline"
echo "========================================"
echo "  Model:          ${ARCH}"
echo "  Dataset:        ${DATASET}"
echo "  Defense:        ${MASK_WHICH} (k=${IMPORTANT_DIM})"
echo "  Clean samples:  ${NUM_SAMPLES_CLEAN}"
echo "  Adv samples:    ${NUM_SAMPLES_ADV}"
echo "  Attacks:        FGSM, PGD-10, PGD-20, PGD-50"
echo "========================================"

python eval/full_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --mask-which ${MASK_WHICH} \
    --layer-name ${LAYER_NAME} \
    --important-dim ${IMPORTANT_DIM} \
    --num-samples-clean ${NUM_SAMPLES_CLEAN} \
    --num-samples-adv ${NUM_SAMPLES_ADV}

echo ""
echo "Done! Results saved in results/"
