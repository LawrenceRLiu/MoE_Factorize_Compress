#!/bin/bash
# Multi-GPU Parallel Teacher Logits Generation
#
# This script launches multiple teacher logit generation workers in parallel.
# Each worker can use one or more GPUs (controlled by GPUS_PER_MODEL).
#
# Usage:
#   ./scripts/generate_teacher_logits_parallel.sh [gpus_per_model] [available_gpus]
#
# Arguments:
#   gpus_per_model: Number of GPUs each model copy should use (default: 1)
#   available_gpus: Comma-separated list of GPU IDs (default: auto-detect)
#
# Examples:
#   # 8 GPUs, 1 GPU per model -> 8 workers
#   ./scripts/generate_teacher_logits_parallel.sh 1 "0,1,2,3,4,5,6,7"
#
#   # 8 GPUs, 2 GPUs per model -> 4 workers
#   ./scripts/generate_teacher_logits_parallel.sh 2 "0,1,2,3,4,5,6,7"
#
#   # 8 GPUs, 4 GPUs per model -> 2 workers
#   ./scripts/generate_teacher_logits_parallel.sh 4 "0,1,2,3,4,5,6,7"
#
#   # Auto-detect GPUs, 1 per model
#   ./scripts/generate_teacher_logits_parallel.sh 1
#
#   # Use only specific GPUs
#   ./scripts/generate_teacher_logits_parallel.sh 1 "0,2,4,6"

set -e

# Parse arguments
GPUS_PER_MODEL=${1:-1}
AVAILABLE_GPUS=${2:-""}

# Auto-detect GPUs if not specified
if [ -z "$AVAILABLE_GPUS" ]; then
    # Count available GPUs using nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        NUM_TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
        AVAILABLE_GPUS=$(seq -s, 0 $((NUM_TOTAL_GPUS-1)))
        echo "Auto-detected $NUM_TOTAL_GPUS GPUs: $AVAILABLE_GPUS"
    else
        echo "Error: nvidia-smi not found and no GPUs specified"
        exit 1
    fi
fi

# Convert comma-separated string to array
IFS=',' read -ra GPU_ARRAY <<< "$AVAILABLE_GPUS"
NUM_TOTAL_GPUS=${#GPU_ARRAY[@]}

# Calculate number of workers
NUM_WORKERS=$((NUM_TOTAL_GPUS / GPUS_PER_MODEL))

if [ $NUM_WORKERS -eq 0 ]; then
    echo "Error: Not enough GPUs!"
    echo "  Available GPUs: $NUM_TOTAL_GPUS"
    echo "  GPUs per model: $GPUS_PER_MODEL"
    echo "  Need at least $GPUS_PER_MODEL GPUs to run 1 worker"
    exit 1
fi

echo "=================================================="
echo "Multi-GPU Teacher Logits Generation"
echo "=================================================="
echo "Configuration:"
echo "  Total GPUs available: $NUM_TOTAL_GPUS (${AVAILABLE_GPUS})"
echo "  GPUs per model: $GPUS_PER_MODEL"
echo "  Number of workers: $NUM_WORKERS"
echo ""

# Show worker GPU assignments
echo "Worker GPU assignments:"
for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
    start_idx=$((worker_id * GPUS_PER_MODEL))
    end_idx=$((start_idx + GPUS_PER_MODEL - 1))

    # Build CUDA_VISIBLE_DEVICES string for this worker
    worker_gpus=""
    for ((i=start_idx; i<=end_idx; i++)); do
        if [ -n "$worker_gpus" ]; then
            worker_gpus="${worker_gpus},${GPU_ARRAY[$i]}"
        else
            worker_gpus="${GPU_ARRAY[$i]}"
        fi
    done

    echo "  Worker $worker_id: GPUs $worker_gpus"
done
echo ""

echo "Logging:"
echo "  Worker 0: Console output (stdout)"
if [ $NUM_WORKERS -gt 1 ]; then
    echo "  Workers 1-$((NUM_WORKERS-1)): Log files in logs/"
fi
echo ""

# Create logs directory
mkdir -p logs

# Launch workers
echo "Launching workers..."
echo ""
PIDS=()

for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
    # Calculate GPU assignment for this worker
    start_idx=$((worker_id * GPUS_PER_MODEL))
    end_idx=$((start_idx + GPUS_PER_MODEL - 1))

    # Build CUDA_VISIBLE_DEVICES string
    worker_gpus=""
    for ((i=start_idx; i<=end_idx; i++)); do
        if [ -n "$worker_gpus" ]; then
            worker_gpus="${worker_gpus},${GPU_ARRAY[$i]}"
        else
            worker_gpus="${GPU_ARRAY[$i]}"
        fi
    done

    echo "Launching worker $worker_id (GPUs: $worker_gpus)..."

    # Launch worker with its assigned GPUs
    # device_map="auto" will automatically shard across visible GPUs
    CUDA_VISIBLE_DEVICES=$worker_gpus python scripts/generate_teacher_logits.py \
        distillation.teacher.generation.worker_id=$worker_id \
        distillation.teacher.generation.total_workers=$NUM_WORKERS \
        &

    PIDS+=($!)
done

echo ""
echo "All workers launched. Waiting for completion..."
echo ""

# Wait for all workers to complete
failed=0
for ((i=0; i<NUM_WORKERS; i++)); do
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

    # Show worker logs
    if [ $NUM_WORKERS -gt 1 ]; then
        echo "Worker logs:"
        for ((i=1; i<NUM_WORKERS; i++)); do
            logfile="logs/teacher_logits_worker_$i.log"
            if [ -f "$logfile" ]; then
                echo "  Worker $i: $logfile"
            fi
        done
        echo ""
    fi

    echo "Output files:"
    for ((i=0; i<NUM_WORKERS; i++)); do
        echo "  Worker $i: output/teacher_logits/teacher_logits_worker${i}.h5"
    done
    echo ""

    echo "Next steps:"
    if [ $NUM_WORKERS -gt 1 ]; then
        echo "1. Merge the worker cache files:"
        echo "   python scripts/merge_teacher_logits.py --num-workers $NUM_WORKERS"
        echo "2. Run distillation training"
    else
        echo "1. Run distillation training"
    fi
else
    echo ""
    echo "=================================================="
    echo "Some workers failed! Check logs:"
    echo "=================================================="
    echo ""

    # Show error logs
    for ((i=1; i<NUM_WORKERS; i++)); do
        logfile="logs/teacher_logits_worker_$i.log"
        if [ -f "$logfile" ]; then
            echo "Worker $i log: $logfile"
            echo "Last 10 lines:"
            tail -n 10 "$logfile" | sed 's/^/  /'
            echo ""
        fi
    done

    exit 1
fi
