#!/bin/bash
# Quick IW-WC evaluation with fewer samples for testing.
# Use this to verify the setup before running the full evaluation.
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

ARCH="rn18_val"
DATASET="cifar10"
CHECKPOINT="Addepalli2022Efficient_RN18.pt"
NUM_SAMPLES=100
EPS=8

echo "========================================"
echo "IW-WC Quick Test (${NUM_SAMPLES} samples)"
echo "========================================"
echo "  Model:       ${ARCH}"
echo "  Dataset:     ${DATASET}"
echo "  Epsilon:     ${EPS}/255"
echo "  Samples:     ${NUM_SAMPLES}"
echo "========================================"

python eval/iwwc_eval.py \
    --arch ${ARCH} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --num-samples ${NUM_SAMPLES} \
    --epsilon ${EPS} \
    --attacks fgsm pgd-10

echo ""
echo "Quick test done! Results saved in results/"
echo ""
echo "For full evaluation, run:"
echo "  bash eval/scripts/eval_iwwc.sh"
