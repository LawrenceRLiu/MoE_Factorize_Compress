# Advanced Teacher Logits Generation

This document describes advanced features for teacher logits generation, including streaming dataset support and multi-GPU parallelization.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Streaming Dataset Support](#streaming-dataset-support)
- [Multi-GPU Parallel Generation](#multi-gpu-parallel-generation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

Teacher logits generation has been enhanced to support:

1. **Unified Configuration**: No more duplicated configs! All settings are now in `conf/distillation/default.yaml`
2. **Streaming Datasets**: Process massive datasets like FineWeb, SlimPajama without loading everything into memory
3. **Multi-GPU Parallelization**: Distribute generation across multiple GPUs for faster processing

## Configuration

All teacher logit generation settings are now in `conf/distillation/default.yaml`:

```yaml
teacher:
  model_path: null
  use_cached_logits: true
  cache_dir: "${output.temp_dir}/teacher_logits"

  generation:
    top_k: 64              # Number of top logits to cache
    batch_size: 4          # Batch size for inference
    device: "cuda"         # Device to use
    force_regenerate: false

    # Multi-GPU settings
    num_workers: 1         # Number of parallel GPU workers
    devices: null          # Explicit device list or null for auto
    worker_id: 0          # For manual multi-worker runs
    total_workers: 1      # Total number of workers

dataset:
  name: null
  config_name: null
  split: "train"
  streaming: false       # Set to true for streaming datasets
  max_samples: null
  text_column: "text"
  max_length: 2048

  # Streaming-specific settings
  streaming_buffer_size: 10000   # Buffer size for streaming
  streaming_chunk_size: 1000     # Process in chunks
  num_proc: 4                    # Tokenization processes (non-streaming)
```

## Streaming Dataset Support

⚠️ **CRITICAL**: When using streaming datasets, you **MUST** specify `max_samples`! See [STREAMING_DATASETS_GUIDE.md](STREAMING_DATASETS_GUIDE.md) for detailed guidance.

### Why Streaming?

For large pretraining datasets (e.g., FineWeb-10B, SlimPajama), you cannot load the entire dataset into memory. Streaming allows you to:

- Process datasets larger than available RAM/disk
- Start processing immediately without downloading everything
- Handle datasets with billions of samples

### IMPORTANT: The FineWeb Trap

FineWeb contains **15 trillion tokens** across ~44 billion documents. If you try to process the entire dataset:
- It would take months on a single GPU
- Generate terabytes of cache files
- Likely run out of disk space
- Never actually complete

**Solution**: Always specify `max_samples` to limit how many samples to process.

### How to Use

#### 1. Enable Streaming in Config

```yaml
# conf/distillation/default.yaml
dataset:
  name: "HuggingFaceFW/fineweb"
  config_name: "sample-10BT"
  split: "train"
  streaming: true           # Enable streaming!
  max_samples: 1000000      # REQUIRED! Limit to 1M samples (2B tokens)
  text_column: "text"
  max_length: 2048
  streaming_chunk_size: 1000  # Process 1000 samples at a time
```

**Note**: If you set `streaming=true` without `max_samples`, the code will raise an error to prevent accidental infinite iteration.

#### 2. Run Generation

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=100000
```

### How It Works

1. **Progressive Processing**: Samples are processed in chunks and written to HDF5 incrementally
2. **Memory Efficient**: Only keeps one chunk in memory at a time
3. **Resizable HDF5**: Cache file grows dynamically as samples are processed
4. **Multi-Worker Support**: Each worker processes every N-th sample (interleaved sharding)

### Limitations

- Cannot know total dataset size upfront
- Random access not available during generation
- For multi-worker streaming, workers use interleaved sharding (worker 0: samples 0, N, 2N...; worker 1: samples 1, N+1, 2N+1...)

## Multi-GPU Parallel Generation

### Why Parallelization?

Teacher logit generation can be the bottleneck. If you have multiple GPUs:

- **4 GPUs**: 4x faster generation
- **8 GPUs**: 8x faster generation
- Fully utilize your hardware

### Method 1: Automatic Parallel Script

The easiest way is to use the provided script:

```bash
# Use 4 GPUs
./scripts/generate_teacher_logits_parallel.sh 4

# Use 8 GPUs
./scripts/generate_teacher_logits_parallel.sh 8
```

This automatically:
- Launches one worker per GPU
- Shards the dataset across workers
- Waits for all workers to complete
- Creates separate cache files per worker

### Method 2: Manual Multi-Worker

For more control, launch workers manually:

```bash
# Terminal 1 - Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.worker_id=0 \
    distillation.teacher.generation.total_workers=4 \
    distillation.teacher.generation.device="cuda"

# Terminal 2 - Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.worker_id=1 \
    distillation.teacher.generation.total_workers=4 \
    distillation.teacher.generation.device="cuda"

# ... and so on for workers 2 and 3
```

### Output Files

Each worker creates its own cache:

```
teacher_logits/
├── teacher_logits_worker0.h5
├── metadata_worker0.json
├── teacher_logits_worker1.h5
├── metadata_worker1.json
├── teacher_logits_worker2.h5
├── metadata_worker2.json
├── teacher_logits_worker3.h5
└── metadata_worker3.json
```

### Using Multi-Worker Caches in Training

**Option 1: Merge cache files** (TODO: implement merge utility)

```bash
python scripts/merge_teacher_logits.py \
    --input-dir output/teacher_logits \
    --num-workers 4
```

**Option 2: Use worker caches directly** (requires custom dataset loader)

Currently, the distillation trainer expects a single cache file. You can:
- Manually concatenate the HDF5 files
- Or modify `CachedLogitsDataset` to load from multiple files

## Examples

### Example 1: Small Dataset (WikiText)

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=wikitext \
    distillation.dataset.config_name=wikitext-2-raw-v1 \
    distillation.dataset.streaming=false
```

### Example 2: Large Dataset with Streaming (FineWeb)

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.config_name=sample-10BT \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=1000000 \
    distillation.dataset.streaming_chunk_size=1000
```

### Example 3: Multi-GPU on 8 GPUs

```bash
./scripts/generate_teacher_logits_parallel.sh 8
```

### Example 4: Streaming + Multi-GPU

```bash
# Set streaming in config first
# Then run parallel script
./scripts/generate_teacher_logits_parallel.sh 4
```

Each worker will process every 4th sample from the stream.

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1**: Reduce batch size
```yaml
teacher:
  generation:
    batch_size: 2  # Default is 4
```

**Solution 2**: Enable streaming
```yaml
dataset:
  streaming: true
```

### Slow Generation

**Solution**: Use multi-GPU parallelization
```bash
./scripts/generate_teacher_logits_parallel.sh 4
```

### Streaming Dataset Hangs

**Check**: Network connection and HuggingFace credentials
```bash
huggingface-cli login
```

### Cache Files Too Large

**Solution 1**: Reduce top_k
```yaml
teacher:
  generation:
    top_k: 32  # Default is 64
```

**Solution 2**: Reduce max_length
```yaml
dataset:
  max_length: 1024  # Default is 2048
```

### Worker Failures in Multi-GPU

**Check**:
1. All GPUs are available: `nvidia-smi`
2. No other processes using GPUs
3. Sufficient disk space for all cache files
4. Check logs for specific worker errors

### Config Not Found Error

**Old error**: `conf/teacher_logits/default.yaml not found`

**Solution**: The old config is deprecated. All settings are now in:
```
conf/distillation/default.yaml
```

Update your scripts to use the unified config.

## Performance Tips

1. **Batch Size**: Increase until you hit OOM, then back off slightly
   - Small models: Try 8-16
   - Large models: Try 2-4

2. **Top-K**: Balance between quality and storage
   - Full vocabulary: Don't use top-k (set very high)
   - Good compression: 64 (default)
   - Aggressive compression: 32

3. **Multi-GPU**: Use all available GPUs
   - Each GPU processes independently
   - Linear speedup (4 GPUs = 4x faster)

4. **Streaming Chunk Size**:
   - Larger chunks = fewer writes, more memory
   - Smaller chunks = more writes, less memory
   - Recommended: 1000

## Migration Guide

### From Old Config to New Config

**Before** (using `conf/teacher_logits/default.yaml`):
```yaml
# teacher_logits/default.yaml
teacher_model: null
dataset:
  name: "wikitext"
generation:
  top_k: 64
  batch_size: 4
```

**After** (using `conf/distillation/default.yaml`):
```yaml
# distillation/default.yaml
teacher:
  generation:
    top_k: 64
    batch_size: 4
dataset:
  name: "wikitext"
```

**Script changes**:

Before:
```bash
python scripts/generate_teacher_logits.py \
    teacher_logits.dataset.name=c4
```

After:
```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=c4
```

## Advanced: Custom Sharding Strategy

For very large clusters or custom setups, you can implement custom sharding:

```python
# Custom worker assignment
def custom_shard(sample_idx, worker_id, total_workers):
    # Example: first half to worker 0, second half to worker 1
    if worker_id == 0:
        return sample_idx < total_samples // 2
    else:
        return sample_idx >= total_samples // 2
```

Modify `TeacherLogitsGenerator._generate_streaming()` to use your custom logic.

## Contributing

Found a bug or have a feature request? Please open an issue!

Common improvements needed:
- [ ] Merge utility for multi-worker cache files
- [ ] Automatic cache file merging in distillation trainer
- [ ] Support for distributed caching across nodes
- [ ] Resume capability for interrupted generation
