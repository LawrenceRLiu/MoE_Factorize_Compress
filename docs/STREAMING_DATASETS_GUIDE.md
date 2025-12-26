# Streaming Datasets Guide: Avoiding the FineWeb Trap

## The Problem

Datasets like FineWeb, SlimPajama, and others contain **billions** of samples and **trillions** of tokens:

- **FineWeb**: 15 trillion tokens across ~44 billion documents
- **SlimPajama**: 627 billion tokens
- **C4**: 364 billion tokens

**Critical Issue**: If you try to generate teacher logits for the entire dataset, you will:
- Run for weeks or months
- Generate terabytes of cache files
- Likely run out of disk space
- Never actually complete

## The Solution: Always Use max_samples

### Required Configuration

When using streaming datasets, you **MUST** specify `max_samples`:

```yaml
# conf/distillation/default.yaml
dataset:
  streaming: true
  max_samples: 1000000  # REQUIRED! Pick appropriate size
```

### The Safety Check

The code now **enforces** this requirement:

```python
# src/distillation_utils.py
if streaming and max_samples is None:
    raise ValueError("max_samples MUST be specified for streaming datasets!")
```

If you try to run without `max_samples`, you'll get a helpful error message.

## How to Choose max_samples

### Guideline by Use Case

| Use Case | Recommended max_samples | Approx. Tokens (2K seq) | Generation Time (1 GPU) |
|----------|------------------------|------------------------|------------------------|
| Quick Test | 1,000 - 10,000 | 2M - 20M | Minutes |
| Development | 10,000 - 100,000 | 20M - 200M | Hours |
| Small-scale Training | 100,000 - 1,000,000 | 200M - 2B | Hours - 1 day |
| Medium-scale Training | 1M - 10M | 2B - 20B | 1-10 days |
| Large-scale Training | 10M - 100M | 20B - 200B | 10-100 days |

### Calculation

For a dataset with `max_length=2048`:
- **Tokens per sample**: ~2048
- **Total tokens**: `max_samples * 2048`
- **FineWeb full (15T tokens)**: Would need ~7.3 billion samples!

### Realistic Recommendations

**For Research/Experimentation**:
```yaml
max_samples: 100000  # 200M tokens, manageable size
```

**For Serious Training**:
```yaml
max_samples: 1000000  # 2B tokens, good balance
```

**For Large-scale Production**:
```yaml
max_samples: 10000000  # 20B tokens, substantial dataset
```

**Never do this**:
```yaml
max_samples: null  # ERROR! Will try to process entire FineWeb
```

## Understanding Streaming Datasets

### How `.take()` Works

When you specify `max_samples` for streaming:

```python
dataset = dataset.take(max_samples)  # Limits iteration
```

This tells the dataset iterator to stop after `max_samples` samples. The dataset doesn't download everything—it only downloads what you need.

### Memory Usage

**Non-streaming**:
- Downloads entire dataset to disk
- Loads indices into memory
- Memory usage: O(dataset_size)

**Streaming with max_samples**:
- Downloads samples on-the-fly
- Only keeps current batch in memory
- Memory usage: O(batch_size) - constant!

## Practical Examples

### Example 1: Testing with FineWeb (Fast)

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.config_name=sample-10BT \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=1000
```

**Result**: Processes 1000 samples (~2M tokens) in minutes

### Example 2: Small Training Run (Moderate)

```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=100000
```

**Result**: Processes 100K samples (~200M tokens) in hours

### Example 3: Production Run with Multi-GPU (Large)

```bash
# First set config
# conf/distillation/default.yaml:
#   dataset:
#     streaming: true
#     max_samples: 10000000

# Then run with 8 GPUs
./scripts/generate_teacher_logits_parallel.sh 8
```

**Result**:
- Each GPU processes 1.25M samples
- Total: 10M samples (~20B tokens)
- Time: ~10 days → ~1.25 days with 8 GPUs

## Alternative Approach: On-the-Fly Generation

For very large datasets, consider **NOT pre-caching** teacher logits at all!

### Why?

1. **Disk Space**: 10M samples with top-k=64 can be ~100GB+ compressed
2. **Flexibility**: Can't easily change datasets mid-training
3. **Iteration**: For multi-epoch training, regenerating logits on each epoch might be better

### On-the-Fly Generation (Future Feature)

Instead of pre-caching:

```python
# During training, generate teacher logits in real-time
for batch in dataloader:
    with torch.no_grad():
        teacher_logits = teacher_model(batch)
    student_logits = student_model(batch)
    loss = distillation_loss(student_logits, teacher_logits, batch)
```

**Pros**:
- No disk space needed
- No pre-generation time
- Can train on unlimited data

**Cons**:
- Requires more GPU memory (teacher + student both loaded)
- Slower per-step training (teacher inference overhead)
- Harder to scale with FSDP

This is **TODO** for future implementation.

## Storage Requirements

### Cache File Size Estimation

For top-k logits caching:

```
Size per sample = max_length * top_k * (2 bytes float16 + 4 bytes int32)
                ≈ max_length * top_k * 6 bytes

Examples (max_length=2048, top_k=64):
- 1K samples: ~786 KB
- 100K samples: ~76 MB (compressed: ~20-40 MB)
- 1M samples: ~768 MB (compressed: ~200-400 MB)
- 10M samples: ~7.6 GB (compressed: ~2-4 GB)
- 100M samples: ~76 GB (compressed: ~20-40 GB)
```

### Disk Space Planning

Add 50% buffer for safety:

| max_samples | Estimated Disk Space (compressed) |
|-------------|----------------------------------|
| 100K | ~50-100 MB |
| 1M | ~300-600 MB |
| 10M | ~3-6 GB |
| 100M | ~30-60 GB |

## Multi-GPU Considerations

When using multi-GPU parallel generation:

```bash
./scripts/generate_teacher_logits_parallel.sh 4
```

Each worker processes a shard:
- Worker 0: samples 0, 4, 8, 12... (for streaming)
- Worker 1: samples 1, 5, 9, 13...
- Worker 2: samples 2, 6, 10, 14...
- Worker 3: samples 3, 7, 11, 15...

**Important**: Each worker will still process `max_samples / num_workers` samples (approximately).

## Error Messages You Might See

### "max_samples MUST be specified for streaming datasets!"

**Cause**: You set `streaming=true` but didn't set `max_samples`

**Fix**:
```bash
python scripts/generate_teacher_logits.py \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=100000  # Add this!
```

### Disk Space Errors

**Cause**: Cache files grew too large

**Fix**: Reduce `max_samples` or increase `top_k` compression:
```yaml
teacher:
  generation:
    top_k: 32  # Reduce from 64
dataset:
  max_samples: 100000  # Reduce from 1M
```

## Best Practices Summary

1. **Always set max_samples for streaming datasets**
2. **Start small** (1K-10K samples) to test your pipeline
3. **Estimate disk space** before large runs
4. **Use multi-GPU** for large max_samples (>1M)
5. **Monitor disk space** during generation
6. **Consider on-the-fly generation** for very large datasets (future)

## Real-World Example: Training on FineWeb

Let's say you want to do distillation training on FineWeb:

### Option 1: Pre-cache Logits (Recommended for <10M samples)

```bash
# 1. Generate teacher logits for 1M samples
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=1000000

# 2. Train student with cached logits
python scripts/run_distillation.py
```

### Option 2: Multi-Epoch Streaming (For very large datasets)

```bash
# Generate logits for 100K samples
python scripts/generate_teacher_logits.py \
    distillation.dataset.max_samples=100000 \
    distillation.dataset.streaming=true

# Train for multiple epochs on this subset
python scripts/run_distillation.py \
    distillation.training.num_train_epochs=10
```

This gives you 1M effective samples (100K * 10 epochs) without needing to cache 1M samples.

### Option 3: Continuous Streaming (Future)

```bash
# TODO: Implement on-the-fly generation
python scripts/run_distillation.py \
    distillation.teacher.use_cached_logits=false \
    distillation.dataset.streaming=true \
    distillation.dataset.max_samples=10000000  # Process 10M samples once
```

## Conclusion

**Golden Rule**: For streaming datasets, always specify `max_samples`.

Start with small values for testing (1K-10K), then scale up based on your:
- Available disk space
- Training time budget
- GPU resources

Remember: You don't need to train on the entire FineWeb dataset. Even 1-10M samples (2-20B tokens) can be sufficient for effective distillation!
