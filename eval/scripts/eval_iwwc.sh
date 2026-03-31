#!/bin/bash
# Evaluate model robustness with Image-Wise Worst-Case (IW-WC) metric.
# A sample is robust only if it resists ALL attacks (FGSM, PGD-10, PGD-20, PGD-50, MI-FGSM).
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18_val"
DATASET="cifar10"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
NUM_SAMPLES=1000
EPS=8

echo "========================================"
echo "IW-WC (Image-Wise Worst-Case) Evaluation"
echo "========================================"
echo "  Model:       ${ARCH}"
echo "  Dataset:     ${DATASET}"
echo "  Epsilon:     ${EPS}/255"
echo "  Samples:     ${NUM_SAMPLES}"
echo "  Attacks:     FGSM, PGD-10, PGD-20, PGD-50, MI-FGSM"
echo "========================================"

python eval/iwwc_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --num-samples ${NUM_SAMPLES} \
    --epsilon ${EPS} \
    --attacks fgsm pgd-10 pgd-20 pgd-50 mifgsm

echo ""
echo "Done! Results saved in results/"
