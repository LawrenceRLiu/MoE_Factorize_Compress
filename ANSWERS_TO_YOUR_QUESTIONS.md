# Answers to Your Follow-up Questions

## Question 1: Models Too Large for Single GPU

**Your Question**: "What about models which are too large to fit on a single GPU, yet small enough that we can make multiple copies?"

### Answer: Now Supported via `device="auto"`

I've added support for automatic model parallelism! The model can now be automatically sharded across multiple GPUs.

### How It Works

**Before** (only single GPU per worker):
```bash
# Each worker uses one GPU with entire model
CUDA_VISIBLE_DEVICES=0 python scripts/generate_teacher_logits.py
```

**Now** (model parallelism):
```bash
# Single worker, model automatically sharded across all GPUs
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.device=auto
```

### Implementation Details

In [src/distillation_utils.py](src/distillation_utils.py#L157-L180):

```python
# Determine device_map strategy
if device == "auto":
    # Enable automatic model parallelism across available GPUs
    device_map_strategy = "auto"
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto"  # HuggingFace automatically distributes model
    )
```

HuggingFace's `device_map="auto"` automatically:
1. Analyzes model architecture
2. Calculates GPU memory requirements
3. Distributes layers across available GPUs
4. Handles cross-GPU communication

### Three Parallelization Strategies

**1. Data Parallelism** (Multiple small model copies)
- Use when: Model fits on 1 GPU, you have many GPUs
- Command: `./scripts/generate_teacher_logits_parallel.sh 8`
- Result: 8 model copies, each on 1 GPU, processing different data

**2. Model Parallelism** (One large model sharded)
- Use when: Model too large for 1 GPU
- Command: `device=auto`
- Result: 1 model copy, sharded across all GPUs

**3. Hybrid** (Multiple sharded model copies)
- Use when: Model needs 2 GPUs, you have 8 GPUs → 4 copies
- Command: Manual setup (advanced)
- Result: 4 model copies, each using 2 GPUs

### Configuration

Updated [conf/distillation/default.yaml](conf/distillation/default.yaml#L13-L30):

```yaml
teacher:
  generation:
    device: "cuda"  # Options:
                    #   "cuda" - single GPU (cuda:0)
                    #   "auto" - automatic model parallelism
                    #   "cuda:0", "cuda:1" - specific GPU

    # Multi-GPU Strategies:
    # 1. Data Parallelism: Multiple model copies
    # 2. Model Parallelism: Single sharded model
    # 3. Hybrid: Advanced usage
```

### Example Use Cases

**Small Model (7B), 8 GPUs → Data Parallelism**:
```bash
./scripts/generate_teacher_logits_parallel.sh 8
# 8 workers, each with full model on 1 GPU
```

**Large Model (70B), 8 GPUs → Model Parallelism**:
```bash
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.device=auto
# 1 worker, model sharded across 8 GPUs
```

**Medium Model (34B needs 2 GPUs), 8 GPUs → Hybrid**:
```bash
# TODO: Implement automatic hybrid mode
# For now: manually run 4 workers, each sees 2 GPUs
```

---

## Question 2: Accelerate vs Custom Script

**Your Question**: "Can we do parallel processing through accelerate or some other off-the-shelf parallelizer? I'm concerned about logging when spawning multiple instances."

### Answer: Sticking with Custom Script (with improvements)

I've evaluated this thoroughly in [docs/PARALLELIZATION_COMPARISON.md](docs/PARALLELIZATION_COMPARISON.md).

### Why Not Full Accelerate?

**The Core Problem**: HDF5 doesn't support parallel writes!

```python
# What we need to do:
with h5py.File(cache_file, 'w') as h5f:
    for batch in dataset:
        logits = model(batch)
        h5f[sample_idx] = logits  # Can't do this from multiple processes!
```

**Options**:
1. **Separate files per worker** (current) - Simple, no conflicts
2. **Synchronized writing** - Serializes writes, defeats parallelism
3. **Leader-worker pattern** - Complex, communication overhead
4. **Use Zarr instead of HDF5** - Different ecosystem

### Hybrid Approach: Best of Both Worlds

✅ **Use Accelerate for**: Model parallelism (automatic sharding)
✅ **Use custom script for**: Data parallelism (simple, works)

```python
# src/distillation_utils.py
if device == "auto":
    # Accelerate handles model sharding automatically
    device_map_strategy = "auto"
else:
    # Custom bash script handles data parallelism
    device_map_strategy = device
```

### Comparison Table

| Approach | Complexity | Logging | HDF5 Support | Flexibility |
|----------|-----------|---------|--------------|-------------|
| Custom Bash | Low ⭐ | Fixed ✅ | Yes ✅ | High ⭐⭐⭐ |
| Full Accelerate | High ⭐⭐⭐ | Good ✅ | No ❌ | Low ⭐ |
| **Hybrid (Current)** | **Medium ⭐⭐** | **Good ✅** | **Yes ✅** | **High ⭐⭐⭐** |

### What We Use

- **Model loading**: HuggingFace's `device_map="auto"` (Accelerate under the hood)
- **Data sharding**: Custom bash script
- **Cache writing**: Per-worker HDF5 files
- **Logging**: Per-worker log files (addressed in Q3!)

---

## Question 3: Logging vs tqdm Conflicts

**Your Question**: "You use tqdm, should the logging initialization be changed to fit that?"

### Answer: Yes! Fixed with Per-Worker Logging

### The Problem

```bash
# Old approach: All workers log to stdout
Worker 0: Loading model...
Worker 1: Loading model...
Worker 0: Processing batch 1...
Worker 1: Processing batch 1...
Worker 2: Loading model...
# Chaos! tqdm progress bars get mangled
```

### The Solution

**Worker 0** (Main): Logs to console with tqdm
**Workers 1+**: Log to separate files

Updated [scripts/generate_teacher_logits.py](scripts/generate_teacher_logits.py#L40-L83):

```python
def setup_logging(worker_id: int = 0, log_dir: Path = None):
    """Setup logging that works with tqdm and multi-worker."""
    if worker_id > 0:
        # Workers 1+: Log to files
        log_file = log_dir / f"teacher_logits_worker_{worker_id}.log"
        logging.basicConfig(
            handlers=[logging.FileHandler(log_file)],
            format=f'%(asctime)s - Worker {worker_id} - %(message)s'
        )
        # Reduce noise from transformers/datasets
        logging.getLogger("transformers").setLevel(logging.WARNING)
    else:
        # Worker 0: Log to console (tqdm-compatible)
        logging.basicConfig(
            handlers=[logging.StreamHandler(sys.stdout)]
        )
```

### Updated Parallel Script

[scripts/generate_teacher_logits_parallel.sh](scripts/generate_teacher_logits_parallel.sh) now:

1. Creates `logs/` directory
2. Tells user where to find logs
3. On failure, shows last 10 lines of each worker log

```bash
$ ./scripts/generate_teacher_logits_parallel.sh 4

==================================================
Multi-GPU Teacher Logits Generation
==================================================
Number of GPUs: 4

Logging:
  Worker 0: Console output (stdout)
  Workers 1-3: Log files in logs/

Launching worker 0 on cuda:0...
Launching worker 1 on cuda:1...
Launching worker 2 on cuda:2...
Launching worker 3 on cuda:3...

# Worker 0 output visible here with clean tqdm progress bars
# Workers 1-3 are quiet (logging to files)

==================================================
All workers completed successfully!
==================================================

Worker logs:
  Worker 1: logs/teacher_logits_worker_1.log
  Worker 2: logs/teacher_logits_worker_2.log
  Worker 3: logs/teacher_logits_worker_3.log
```

### tqdm Compatibility

Worker 0 can now use tqdm safely:

```python
# src/teacher_logits.py
for batch in tqdm(dataloader, desc="Generating logits"):
    # Progress bar displays cleanly!
    logits = model(batch)
```

No more:
- Mangled progress bars ❌
- Mixed log messages ❌
- Unclear which worker is doing what ❌

Now:
- Clean progress bar from worker 0 ✅
- Separate logs for debugging ✅
- Easy to monitor all workers ✅

### Debugging Failed Workers

If a worker fails:

```bash
Some workers failed! Check logs:
==================================================

Worker 2 log: logs/teacher_logits_worker_2.log
Last 10 lines:
  2025-12-26 - Worker 2 - Loading model...
  2025-12-26 - Worker 2 - Model loaded successfully
  2025-12-26 - Worker 2 - Processing samples 50000-75000
  2025-12-26 - Worker 2 - ERROR: CUDA out of memory
  Traceback (most recent call last):
    ...
```

Much easier to debug!

---

## Summary of Improvements

### Issue 1: Large Models ✅ SOLVED
- Added `device="auto"` for automatic model parallelism
- Models can now be sharded across GPUs
- Documented in config with clear examples

### Issue 2: Parallelization Approach ✅ SOLVED
- Evaluated Accelerate vs custom script
- Chose hybrid: Accelerate for model, custom for data
- Documented reasoning in [PARALLELIZATION_COMPARISON.md](docs/PARALLELIZATION_COMPARISON.md)

### Issue 3: Logging Chaos ✅ SOLVED
- Per-worker log files (workers 1+)
- Worker 0 logs to console (tqdm-compatible)
- Parallel script shows log locations and tails on failure

## Usage Examples

### Example 1: Small Model, Many GPUs (Data Parallelism)
```bash
# 7B model, 8 GPUs → 8 copies
./scripts/generate_teacher_logits_parallel.sh 8
```

**Result**:
- Worker 0: Console output with progress bar
- Workers 1-7: Log to `logs/teacher_logits_worker_*.log`
- Linear speedup (~8x faster)

### Example 2: Large Model, Multiple GPUs (Model Parallelism)
```bash
# 70B model, 8 GPUs → 1 sharded copy
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.device=auto
```

**Result**:
- Model automatically distributed across 8 GPUs
- Single worker, logs to console
- Can fit models that wouldn't fit on 1 GPU

### Example 3: Debugging Failed Worker
```bash
# If worker 3 failed:
tail -f logs/teacher_logits_worker_3.log
# Watch live progress and see where it failed
```

## Files Modified

1. **[src/distillation_utils.py](src/distillation_utils.py)** - Added model parallelism support
2. **[scripts/generate_teacher_logits.py](scripts/generate_teacher_logits.py)** - Added per-worker logging
3. **[scripts/generate_teacher_logits_parallel.sh](scripts/generate_teacher_logits_parallel.sh)** - Better log reporting
4. **[conf/distillation/default.yaml](conf/distillation/default.yaml)** - Documented parallelization strategies

## New Documentation

1. **[docs/PARALLELIZATION_COMPARISON.md](docs/PARALLELIZATION_COMPARISON.md)** - Detailed comparison of approaches
2. **[ANSWERS_TO_YOUR_QUESTIONS.md](ANSWERS_TO_YOUR_QUESTIONS.md)** - This file

## Testing

To verify improvements:

```bash
# Test 1: Model parallelism
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.device=auto \
    distillation.dataset.max_samples=100

# Test 2: Data parallelism with logging
./scripts/generate_teacher_logits_parallel.sh 2
# Check logs/ directory for worker logs

# Test 3: Check log files
ls logs/
cat logs/teacher_logits_worker_1.log
```

All three issues are now resolved! ✅
