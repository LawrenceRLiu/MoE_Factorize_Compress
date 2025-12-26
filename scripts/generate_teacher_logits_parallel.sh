#!/bin/bash
# Multi-GPU Parallel Teacher Logits Generation
#
# This script launches multiple teacher logit generation workers in parallel,
# one per GPU. Each worker processes a shard of the dataset.
#
# Usage:
#   ./scripts/generate_teacher_logits_parallel.sh [num_gpus]
#
# Example:
#   ./scripts/generate_teacher_logits_parallel.sh 4

set -e

# Number of GPUs to use
NUM_GPUS=${1:-4}

echo "=================================================="
echo "Multi-GPU Teacher Logits Generation"
echo "=================================================="
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Launch workers
PIDS=()
for ((worker_id=0; worker_id<NUM_GPUS; worker_id++)); do
    echo "Launching worker $worker_id on cuda:$worker_id..."

    CUDA_VISIBLE_DEVICES=$worker_id python scripts/generate_teacher_logits.py \
        distillation.teacher.generation.worker_id=$worker_id \
        distillation.teacher.generation.total_workers=$NUM_GPUS \
        distillation.teacher.generation.device="cuda" \
        &

    PIDS+=($!)
done

echo ""
echo "All workers launched. Waiting for completion..."
echo ""

# Wait for all workers to complete
failed=0
for ((i=0; i<NUM_GPUS; i++)); do
    if wait ${PIDS[$i]}; then
        echo "Worker $i completed successfully"
    else
        echo "Worker $i failed!"
        failed=1
    fi
done

if [ $failed -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "All workers completed successfully!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Merge the worker cache files (if needed)"
    echo "2. Run distillation training"
else
    echo ""
    echo "=================================================="
    echo "Some workers failed! Check logs above."
    echo "=================================================="
    exit 1
fi
