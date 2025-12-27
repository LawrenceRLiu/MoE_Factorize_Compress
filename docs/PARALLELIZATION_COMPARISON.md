# Parallelization Approach Comparison

## Current Approach: Custom Bash Script

### What We Have
```bash
#!/bin/bash
# scripts/generate_teacher_logits_parallel.sh
for ((worker_id=0; worker_id<NUM_GPUS; worker_id++)); do
    CUDA_VISIBLE_DEVICES=$worker_id python scripts/generate_teacher_logits.py \
        distillation.teacher.generation.worker_id=$worker_id \
        distillation.teacher.generation.total_workers=$NUM_GPUS &
done
wait
```

### Pros
- **Simple**: Easy to understand, ~40 lines of bash
- **Transparent**: Clear what's happening
- **Flexible**: Easy to customize worker assignment
- **No dependencies**: Just bash + Python
- **File-based**: Each worker writes separate cache file

### Cons
- **Logging chaos**: Multiple processes write to stdout simultaneously
- **No coordination**: Workers don't communicate
- **Error handling**: Must manually check each worker
- **Platform-specific**: Bash script won't work on Windows

## Alternative: Accelerate

### What It Would Look Like

```python
# scripts/generate_teacher_logits_accelerate.py
from accelerate import Accelerator

accelerator = Accelerator()

# Accelerate automatically handles:
# - Process spawning
# - Device assignment
# - Data sharding
# - Logging coordination

model = AutoModelForCausalLM.from_pretrained(...)
model = accelerator.prepare(model)

# Accelerate shards dataset automatically
for batch in accelerator.prepare(dataloader):
    with torch.no_grad():
        logits = model(batch)
    # Save logits...
```

### Pros
- **Unified logging**: Accelerate handles multi-process logging
- **Better coordination**: Built-in synchronization primitives
- **Cross-platform**: Works on Windows
- **Standard tool**: Well-maintained, widely used
- **Easier scaling**: Can extend to multi-node easily

### Cons
- **More complex**: More abstraction, harder to debug
- **Less control**: Harder to customize sharding strategy
- **File handling**: Would need to coordinate HDF5 writes
- **Learning curve**: Team needs to understand Accelerate
- **Overkill**: Might be overengineered for this task

## Alternative: Torch Distributed

### What It Would Look Like

```python
# scripts/generate_teacher_logits_distributed.py
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Manual data sharding
dataset_shard = dataset.shard(num_shards=world_size, index=rank)

# Process shard...
```

### Pros
- **Lower-level control**: Fine-grained control over distribution
- **PyTorch native**: No extra dependencies
- **Efficient communication**: NCCL backend for GPU communication

### Cons
- **Most complex**: Requires understanding distributed training
- **Boilerplate**: Lots of manual setup
- **Still has logging issues**: Would need manual coordination

## Recommendation: Hybrid Approach

### Best Solution: Custom Script + Better Logging

Keep the simple bash script BUT:
1. **Fix logging**: Use per-worker log files
2. **Use accelerate for model loading**: Get automatic model parallelism
3. **Keep data sharding simple**: Manual sharding in our code

### Implementation

```python
# src/distillation_utils.py
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def generate_teacher_logits(...):
    # Option 1: Use accelerate for model parallelism only
    if device == "auto":
        with init_empty_weights():
            teacher_model = AutoModelForCausalLM.from_config(config)
        teacher_model = load_checkpoint_and_dispatch(
            teacher_model,
            teacher_model_path,
            device_map="auto"
        )
    else:
        # Option 2: Traditional loading for data parallelism
        teacher_model = AutoModelForCausalLM.from_pretrained(...)

    # Keep our simple data sharding logic
    # Keep our simple HDF5 caching logic
```

## Why Not Full Accelerate?

### The Core Issue: HDF5 Writing

Our caching approach requires:
```python
with h5py.File(cache_file, 'w') as h5f:
    for batch in dataset:
        logits = model(batch)
        h5f[sample_idx] = logits  # Sequential writing
```

**Problem**: HDF5 doesn't support parallel writes from multiple processes to same file!

### Workarounds

1. **Separate files (current)**: Each worker writes own file
   - Pro: Simple, no conflicts
   - Con: Need to merge later

2. **Synchronized writing**: Lock-based coordination
   - Pro: Single output file
   - Con: Serializes writes, slow

3. **Leader-worker pattern**: One process writes, others send data
   - Pro: Single file, coordinated
   - Con: Communication overhead, complex

4. **Use different format**: Zarr instead of HDF5
   - Pro: Supports parallel writes
   - Con: Different ecosystem, less standard

## Final Recommendation

**Stick with custom bash script** with these improvements:

### For Issue #1 (Large Models)
```bash
# Model parallelism: Single worker, model across all GPUs
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.device=auto
```

### For Issue #2 (Better Logging)
```python
# In generate_teacher_logits.py
import logging
from pathlib import Path

# Per-worker log files
if worker_id > 0:
    log_file = Path("logs") / f"worker_{worker_id}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - Worker %(worker_id)s - %(message)s'
    )
else:
    # Worker 0 logs to stdout
    logging.basicConfig(...)
```

### For Issue #3 (Better Coordination)
```python
# Optional: Simple progress tracking via files
progress_file = cache_dir / f"worker_{worker_id}_progress.json"
json.dump({"samples_processed": count}, open(progress_file, 'w'))

# Monitor script can aggregate progress
# scripts/monitor_parallel_generation.py
total_progress = sum(
    json.load(open(f))["samples_processed"]
    for f in glob("worker_*_progress.json")
)
```

## Summary Table

| Approach | Complexity | Logging | Model Parallel | Data Parallel | Flexibility |
|----------|-----------|---------|----------------|---------------|-------------|
| Custom Bash | ⭐ Low | ❌ Poor | ✅ Yes (auto) | ✅ Yes | ⭐⭐⭐ High |
| Accelerate Full | ⭐⭐⭐ High | ✅ Good | ✅ Yes | ✅ Yes | ⭐ Low |
| Hybrid (Recommended) | ⭐⭐ Medium | ✅ Good | ✅ Yes | ✅ Yes | ⭐⭐ Medium |

## Decision

**Use hybrid approach**:
1. Keep bash script for data parallelism (simple, works)
2. Add `device="auto"` support for model parallelism (already done!)
3. Fix logging (next step)
4. Add optional progress monitoring

This gives us:
- ✅ Support for large models (model parallelism)
- ✅ Support for multiple model copies (data parallelism)
- ✅ Simple, understandable code
- ✅ Easy to debug and customize
- ⚠️ Need to fix logging (next)
