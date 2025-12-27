# Parallel GPU Generation - Examples

## How It Works

The parallel script now supports **flexible GPU assignment**:

```bash
./scripts/generate_teacher_logits_parallel.sh [gpus_per_model] [gpu_list]
```

- **gpus_per_model**: How many GPUs each model copy uses
- **gpu_list**: Comma-separated list of available GPUs
- **Result**: `total_gpus / gpus_per_model` workers launched

Each worker:
1. Gets assigned specific GPUs via `CUDA_VISIBLE_DEVICES`
2. Loads model with `device_map="auto"`
3. HuggingFace automatically handles single vs multi-GPU

## Example Scenarios

### Scenario 1: Small Model (7B), 8 GPUs

**Goal**: Maximum parallelism (8 workers)

```bash
./scripts/generate_teacher_logits_parallel.sh 1 "0,1,2,3,4,5,6,7"
```

**Result**:
- Worker 0: GPU 0
- Worker 1: GPU 1
- Worker 2: GPU 2
- ...
- Worker 7: GPU 7

Each worker processes 1/8th of the dataset in parallel.

**Output**:
```
==================================================
Multi-GPU Teacher Logits Generation
==================================================
Configuration:
  Total GPUs available: 8 (0,1,2,3,4,5,6,7)
  GPUs per model: 1
  Number of workers: 8

Worker GPU assignments:
  Worker 0: GPUs 0
  Worker 1: GPUs 1
  Worker 2: GPUs 2
  Worker 3: GPUs 3
  Worker 4: GPUs 4
  Worker 5: GPUs 5
  Worker 6: GPUs 6
  Worker 7: GPUs 7
```

### Scenario 2: Medium Model (34B, needs 2 GPUs), 8 GPUs

**Goal**: 4 model copies, each using 2 GPUs

```bash
./scripts/generate_teacher_logits_parallel.sh 2 "0,1,2,3,4,5,6,7"
```

**Result**:
- Worker 0: GPUs 0,1 (model sharded across both)
- Worker 1: GPUs 2,3 (model sharded across both)
- Worker 2: GPUs 4,5 (model sharded across both)
- Worker 3: GPUs 6,7 (model sharded across both)

Each worker processes 1/4th of the dataset.

**Output**:
```
Configuration:
  Total GPUs available: 8 (0,1,2,3,4,5,6,7)
  GPUs per model: 2
  Number of workers: 4

Worker GPU assignments:
  Worker 0: GPUs 0,1
  Worker 1: GPUs 2,3
  Worker 2: GPUs 4,5
  Worker 3: GPUs 6,7
```

### Scenario 3: Large Model (70B, needs 4 GPUs), 8 GPUs

**Goal**: 2 model copies, each using 4 GPUs

```bash
./scripts/generate_teacher_logits_parallel.sh 4 "0,1,2,3,4,5,6,7"
```

**Result**:
- Worker 0: GPUs 0,1,2,3 (model sharded)
- Worker 1: GPUs 4,5,6,7 (model sharded)

Each worker processes half the dataset.

### Scenario 4: Very Large Model (405B, needs all 8 GPUs), 8 GPUs

**Goal**: 1 model copy using all 8 GPUs

```bash
# Just run single instance, no parallel script needed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/generate_teacher_logits.py
```

**Or equivalently**:
```bash
./scripts/generate_teacher_logits_parallel.sh 8 "0,1,2,3,4,5,6,7"
```

**Result**:
- Worker 0: GPUs 0,1,2,3,4,5,6,7 (entire model sharded)

Processes entire dataset on one model copy.

### Scenario 5: Using Only Specific GPUs

**Goal**: Use only GPUs 1, 3, 5, 7 (e.g., others are busy)

```bash
./scripts/generate_teacher_logits_parallel.sh 1 "1,3,5,7"
```

**Result**:
- Worker 0: GPU 1
- Worker 1: GPU 3
- Worker 2: GPU 5
- Worker 3: GPU 7

4 workers, each on 1 GPU.

### Scenario 6: Auto-Detect GPUs

**Goal**: Use all available GPUs, 1 per model

```bash
./scripts/generate_teacher_logits_parallel.sh 1
```

Script automatically detects available GPUs using `nvidia-smi`.

## How device_map="auto" Works

For each worker, HuggingFace sees only the GPUs in `CUDA_VISIBLE_DEVICES`:

**Worker 0 with CUDA_VISIBLE_DEVICES=0,1**:
```python
# From worker's perspective, it has 2 GPUs (cuda:0 and cuda:1)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"  # HuggingFace shards across cuda:0 and cuda:1
)
```

HuggingFace automatically:
1. Analyzes model size and layer dependencies
2. Calculates memory required per layer
3. Distributes layers across available GPUs
4. Handles cross-GPU communication

## Performance Comparison

| Setup | Workers | GPUs/Worker | Speedup | Use Case |
|-------|---------|-------------|---------|----------|
| 1 GPU | 1 | 1 | 1x | Baseline |
| 8 GPUs, 1/worker | 8 | 1 | ~8x | Small model |
| 8 GPUs, 2/worker | 4 | 2 | ~4x | Medium model |
| 8 GPUs, 4/worker | 2 | 4 | ~2x | Large model |
| 8 GPUs, 8/worker | 1 | 8 | 1x | Very large model |

**Speedup Formula**: `num_workers` (assuming perfect parallel efficiency)

## Choosing GPUs Per Model

### Rule of Thumb

1. **Check model memory requirement**:
   ```bash
   # Rough estimate: model_size * 2 (for activations)
   # 7B model @ fp16: ~14GB -> fits on 1x A100 (80GB)
   # 34B model @ fp16: ~68GB -> needs 2x A100
   # 70B model @ fp16: ~140GB -> needs 4x A100
   ```

2. **Test with single GPU first**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/generate_teacher_logits.py \
       distillation.dataset.max_samples=10
   ```

3. **If OOM, increase GPUs**:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python scripts/generate_teacher_logits.py \
       distillation.dataset.max_samples=10
   ```

4. **Once you know GPUs needed, use parallel script**:
   ```bash
   ./scripts/generate_teacher_logits_parallel.sh [gpus_needed]
   ```

## Common Patterns

### Pattern 1: Maximize Throughput (Small Model)
```bash
# Use as many workers as possible
./scripts/generate_teacher_logits_parallel.sh 1
```

### Pattern 2: Large Model Fits on N GPUs
```bash
# Multiple copies for data parallelism
./scripts/generate_teacher_logits_parallel.sh [N]
```

### Pattern 3: Model Barely Fits
```bash
# Single copy with all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/generate_teacher_logits.py
```

## Debugging

### Check GPU Assignment
```bash
# In worker log, you'll see:
# Worker 2/4
# CUDA_VISIBLE_DEVICES: 4,5
```

### Monitor GPU Usage
```bash
# While script is running:
watch -n 1 nvidia-smi
```

### Test Small Batch First
```bash
# Before full run, test with 10 samples:
./scripts/generate_teacher_logits_parallel.sh 2 "0,1,2,3" \
    distillation.dataset.max_samples=10
```

## Advanced: Uneven GPU Distribution

If you have heterogeneous GPUs (e.g., 4x A100 + 4x V100):

```bash
# Run two separate instances:

# Instance 1: Use A100s (GPUs 0-3) with 1 GPU each
./scripts/generate_teacher_logits_parallel.sh 1 "0,1,2,3"

# Instance 2: Use V100s (GPUs 4-7) with 2 GPUs each (slower GPUs)
./scripts/generate_teacher_logits_parallel.sh 2 "4,5,6,7"
```

Or manually:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_teacher_logits.py worker_id=0 total_workers=6 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher_logits.py worker_id=1 total_workers=6 &
# etc...
```

## Summary

**Simple formula**:
```
num_workers = total_gpus / gpus_per_model
```

**To use**:
```bash
./scripts/generate_teacher_logits_parallel.sh [gpus_per_model] [gpu_list]
```

**Key insight**: `device_map="auto"` handles everything automatically once you set `CUDA_VISIBLE_DEVICES`!
