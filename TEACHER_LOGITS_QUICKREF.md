# Teacher Logits Generation - Quick Reference

## Common Commands

### Basic Usage (Small Dataset)
```bash
python scripts/generate_teacher_logits.py
```

### Streaming Dataset (FineWeb, SlimPajama, etc.)
⚠️ **CRITICAL**: You MUST specify `max_samples` for streaming datasets!

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=1000000  # REQUIRED!
```

**Why?** FineWeb has 15T tokens. Without max_samples, it would run for months!

### Multi-GPU Parallel

**Syntax**: `./scripts/generate_teacher_logits_parallel.sh [gpus_per_model] [gpu_list]`

```bash
# 8 GPUs, 1 GPU per model -> 8 workers (small model, max parallelism)
./scripts/generate_teacher_logits_parallel.sh 1 "0,1,2,3,4,5,6,7"

# 8 GPUs, 2 GPUs per model -> 4 workers (medium model ~34B)
./scripts/generate_teacher_logits_parallel.sh 2 "0,1,2,3,4,5,6,7"

# 8 GPUs, 4 GPUs per model -> 2 workers (large model ~70B)
./scripts/generate_teacher_logits_parallel.sh 4 "0,1,2,3,4,5,6,7"

# Auto-detect all GPUs, 1 per model
./scripts/generate_teacher_logits_parallel.sh 1
```

See [docs/PARALLEL_GPU_EXAMPLES.md](docs/PARALLEL_GPU_EXAMPLES.md) for more examples.

## Configuration Quick Reference

### Key Config Parameters

```yaml
# conf/distillation/default.yaml

# Teacher settings
teacher:
  model_path: null  # Use model.name if null
  cache_dir: "${output.temp_dir}/teacher_logits"

  generation:
    top_k: 64           # Number of top logits to cache
    batch_size: 4       # Batch size (increase for faster, decrease for OOM)
    device: "cuda"      # Device to use
    force_regenerate: false  # Set true to regenerate

    # Multi-GPU (usually auto-configured by parallel script)
    num_workers: 1
    worker_id: 0
    total_workers: 1

# Dataset settings
dataset:
  name: "wikitext"
  config_name: "wikitext-2-raw-v1"
  split: "train"
  streaming: false    # Set true for large datasets
  max_samples: null   # Limit samples (null = all)
  max_length: 2048

  # Streaming-specific
  streaming_chunk_size: 1000
  streaming_buffer_size: 10000
```

## Command-line Override Examples

### Change Dataset
```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=c4 \
    distillation.dataset.config_name=en
```

### Reduce Batch Size (for OOM)
```bash
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.batch_size=2
```

### Change Top-K (reduce cache size)
```bash
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.top_k=32
```

### Force Regenerate
```bash
python scripts/generate_teacher_logits.py \
    distillation.teacher.generation.force_regenerate=true
```

### Limit Samples
```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.max_samples=10000
```

## Troubleshooting Quick Fixes

### Out of Memory
```yaml
# Reduce batch size
distillation.teacher.generation.batch_size=2
```

### Cache Too Large
```yaml
# Reduce top_k
distillation.teacher.generation.top_k=32

# Or reduce max_length
distillation.dataset.max_length=1024
```

### Slow Generation
```bash
# Use multi-GPU
./scripts/generate_teacher_logits_parallel.sh 4
```

### Dataset Not Found
```bash
# Check dataset name and login to HuggingFace
huggingface-cli login
```

## Output Files

### Single GPU
```
output/teacher_logits/
├── teacher_logits.h5      # HDF5 cache file
└── metadata.json          # Metadata
```

### Multi-GPU (4 workers)
```
output/teacher_logits/
├── teacher_logits_worker0.h5
├── metadata_worker0.json
├── teacher_logits_worker1.h5
├── metadata_worker1.json
├── teacher_logits_worker2.h5
├── metadata_worker2.json
├── teacher_logits_worker3.h5
└── metadata_worker3.json
```

## Performance Tips

| Scenario | Recommended Settings |
|----------|---------------------|
| Small dataset (<1GB) | `streaming=false`, `batch_size=8` |
| Medium dataset (1-10GB) | `streaming=false`, `batch_size=4` |
| Large dataset (>10GB) | `streaming=true`, `batch_size=4` |
| Very large dataset (>100GB) | `streaming=true`, multi-GPU |
| Fast GPU (A100) | `batch_size=8-16` |
| Slow GPU (V100) | `batch_size=2-4` |
| Multiple GPUs available | Use parallel script |

## Common Dataset Configs

### WikiText (Small)
```yaml
dataset:
  name: "wikitext"
  config_name: "wikitext-2-raw-v1"
  streaming: false
```

### C4 (Large)
```yaml
dataset:
  name: "c4"
  config_name: "en"
  streaming: true
  max_samples: 1000000
```

### FineWeb (Very Large)
```yaml
dataset:
  name: "HuggingFaceFW/fineweb"
  config_name: "sample-10BT"
  streaming: true
  max_samples: 10000000
```

### SlimPajama (Very Large)
```yaml
dataset:
  name: "cerebras/SlimPajama-627B"
  config_name: null
  streaming: true
  max_samples: 10000000
```

## Next Steps

After generating teacher logits:

1. **Run Distillation**:
   ```bash
   python scripts/run_distillation.py
   ```

2. **Or Distributed Distillation**:
   ```bash
   torchrun --nproc_per_node=8 scripts/run_distillation.py
   ```

## Documentation Links

- [Advanced Guide](docs/TEACHER_LOGITS_ADVANCED.md) - Streaming and multi-GPU details
- [Distillation Guide](docs/DISTILLATION.md) - Full distillation documentation
- [Improvements Summary](IMPROVEMENTS_TEACHER_LOGITS.md) - What's new

## Getting Help

If you encounter issues:
1. Check the logs for error messages
2. Try reducing batch_size
3. Verify dataset name and config
4. Check available GPU memory: `nvidia-smi`
5. See troubleshooting in [docs/TEACHER_LOGITS_ADVANCED.md](docs/TEACHER_LOGITS_ADVANCED.md)
