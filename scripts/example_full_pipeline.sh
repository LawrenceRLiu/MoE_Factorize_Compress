#!/bin/bash
# Example: Full MoE Compression Pipeline
# This script demonstrates the complete workflow for compressing an MoE model

set -e  # Exit on error

echo "========================================="
echo "MoE Compression - Full Pipeline Example"
echo "========================================="

# Activate conda environment
echo "Activating conda environment..."
conda activate MoE_Compress

# Configuration
MODEL_NAME="Qwen/Qwen2.5-MoE-A14B-Chat"  # Adjust to actual model
EXPERIMENT_NAME="qwen_compression_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Experiment: $EXPERIMENT_NAME"

# Phase 1: Zero-Shot Compression
echo ""
echo "========================================="
echo "Phase 1: Zero-Shot Compression"
echo "========================================="
echo "This will use all 8 GPUs to compress the model in parallel..."

python scripts/run_compression.py \
    experiment_name=$EXPERIMENT_NAME \
    model.name=$MODEL_NAME

echo ""
echo "✓ Compression complete!"
echo "Check compressed weights in: models/$MODEL_NAME/compressed/"

# Phase 2a: Start Async Evaluation (in background)
echo ""
echo "========================================="
echo "Phase 2a: Starting Async Evaluation"
echo "========================================="
echo "This will run in the background on GPUs 6-7..."

python scripts/run_async_eval.py \
    experiment_name=$EXPERIMENT_NAME \
    model.name=$MODEL_NAME \
    evaluation.evaluate_baseline=true &

EVAL_PID=$!
echo "Evaluation started with PID: $EVAL_PID"

# Give evaluation time to start and evaluate baseline
sleep 10

# Phase 2b: Knowledge Distillation
echo ""
echo "========================================="
echo "Phase 2b: Knowledge Distillation"
echo "========================================="
echo "This will run on GPUs 0-5 while evaluation runs on 6-7..."

python scripts/run_distillation.py \
    experiment_name=$EXPERIMENT_NAME \
    model.name=$MODEL_NAME

echo ""
echo "✓ Distillation complete!"

# Stop async evaluation
echo ""
echo "Stopping async evaluation..."
kill $EVAL_PID 2>/dev/null || true

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  Compressed model: models/$MODEL_NAME/compressed/"
echo "  Distilled model: models/$MODEL_NAME/distilled/"
echo "  Evaluation results: models/$MODEL_NAME/evaluation/"
echo ""
echo "View results in WandB: https://wandb.ai (project: moe-compression)"
