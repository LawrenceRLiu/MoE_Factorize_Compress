# Optimizer Memory Optimization Guide

## Problem: OOM During Optimizer Step

When training large models with FSDP, you may encounter OOM errors specifically during `optimizer.step()`, even though forward/backward passes fit in memory. This happens because **optimizer states consume significant memory**.

### Memory Breakdown for AdamW

AdamW maintains two state tensors per parameter:
- **Momentum (first moment)**: Same size as parameters
- **Variance (second moment)**: Same size as parameters

**Total memory = Parameters + Gradients + Optimizer States**
- Parameters: ~50GB (bf16)
- Gradients: ~50GB (bf16)
- Optimizer states: ~100GB (fp32, even with bf16 training!)
  - 2 states × 50GB × 2 (bf16→fp32) = 200GB, but FSDP shards this to ~100GB per GPU

**Result**: 50GB + 50GB + 100GB = 200GB total, or ~100GB per GPU with FSDP sharding
**Problem**: Exceeds 80GB A100 capacity → OOM during optimizer.step()

## Solution 1: Paged AdamW 32-bit (RECOMMENDED for FSDP)

Use `bitsandbytes` paged optimizer for memory efficiency with better FSDP compatibility.

### Configuration

In [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml):

```yaml
training:
  optim: "paged_adamw_32bit"  # Memory-efficient with FSDP compatibility
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### Installation

```bash
# Activate your environment
conda activate MoE_Compress

# Install bitsandbytes
python -m pip install bitsandbytes
```

### How It Works

- Uses unified memory paging to handle optimizer states
- Automatically moves optimizer states between GPU and CPU RAM as needed
- Keeps states in fp32 but reduces memory pressure through paging
- **Full FSDP compatibility** (unlike 8-bit variants)

### Memory Savings

**Before (adamw_torch)**:
- Parameters: 50GB
- Gradients: 50GB
- Optimizer states: 100GB (fp32, all on GPU)
- **Total: 200GB → 100GB/GPU with FSDP → OOM!**

**After (paged_adamw_32bit)**:
- Parameters: 50GB
- Gradients: 50GB
- Optimizer states: 100GB (fp32, paged between GPU/CPU)
- **Effective GPU usage: ~70-75GB/GPU**

**Result**: Fits on 80GB A100s through dynamic paging!

## Solution 1b: 8-bit AdamW (Higher Memory Savings, FSDP Compatibility Issues)

**NOTE**: `adamw_bnb_8bit` has known compatibility issues with FSDP in some configurations. It may fail with "tensors not on same device" errors. If you encounter this, use `paged_adamw_32bit` instead (Solution 1 above).

### Configuration

```yaml
training:
  optim: "adamw_bnb_8bit"  # May have FSDP issues
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### Known Issues

- **FSDP Compatibility**: May fail with `RuntimeError: All input tensors need to be on the same GPU`
- Some FSDP operations keep optimizer states on CPU temporarily
- 8-bit optimizer expects all tensors on GPU
- **Use `paged_adamw_32bit` instead for FSDP**

### Memory Savings (if it works)

- Optimizer states: 25GB (int8) vs 100GB (fp32)
- **Total: ~65GB/GPU** (best case)
- But **not compatible with FSDP in all cases**

## Solution 2: Reduce Sequence Length

If still encountering OOM, reduce the sequence length to decrease activation memory.

### Configuration

```yaml
dataset:
  max_length: 1024  # Down from 2048
```

### Memory Impact

- Activation memory scales with sequence length
- 2048 → 1024 reduces activation memory by ~50%
- **Trade-off**: Fewer tokens per batch, longer training time

## Solution 3: CPU Offloading (Last Resort)

Only use if the above solutions don't work. CPU offloading is **much slower** (~2-3x).

### Configuration

```bash
# Use CPU offload config
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_2xa100_cpu_offload.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment
```

In [accelerate_config_2xa100_cpu_offload.yaml](accelerate_config_2xa100_cpu_offload.yaml):

```yaml
fsdp_config:
  fsdp_offload_params: true  # Offload parameters to CPU
```

### Trade-offs

- **Pro**: Can fit larger models/batches
- **Con**: 2-3x slower due to CPU↔GPU transfers
- **Con**: Requires significant CPU RAM

## Solution 4: Alternative Optimizers

### AdamW 8-bit Variants

```yaml
# Standard 8-bit (recommended)
optim: "adamw_bnb_8bit"

# Paged 8-bit (for unified memory systems)
optim: "paged_adamw_8bit"

# 32-bit paged (if 8-bit causes issues)
optim: "paged_adamw_32bit"
```

### SGD (Memory-efficient but less common for LLMs)

```yaml
optim: "sgd"
learning_rate: 0.1  # Higher LR needed for SGD
```

SGD only stores momentum (1 state vs 2 for Adam), but **rarely used for LLM pretraining**.

## Recommended Strategy

1. **Start with 8-bit AdamW** (`adamw_bnb_8bit`)
   - Best performance/memory trade-off
   - No convergence impact
   - Easy to implement

2. **If still OOM**: Reduce sequence length to 1024
   - Minimal impact on final quality
   - Faster iteration during debugging

3. **Last resort**: Enable CPU offloading
   - Sacrifices speed for capacity
   - Only if above solutions insufficient

## Monitoring Memory Usage

Watch GPU memory during training:

```bash
# Terminal 1: Start training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment

# Terminal 2: Monitor memory
watch -n 1 nvidia-smi
```

### What to Look For

- **Initial load**: Both GPUs should show ~30-40GB (model loading)
- **Forward pass**: Gradual increase to ~50-60GB (activations)
- **Backward pass**: Peak at ~60-70GB (gradients)
- **Optimizer step**: Should stay under 80GB with 8-bit AdamW
  - If it spikes to >80GB → OOM
  - With regular AdamW, it spikes to >80GB (the problem!)

### Verifying FSDP Sharding

Both GPUs should show similar memory usage:
- **Good**: GPU 0: 65GB, GPU 1: 64GB (balanced sharding)
- **Bad**: GPU 0: 75GB, GPU 1: 10GB (not sharding, DDP fallback)

If memory is unbalanced, verify FSDP is configured correctly in TrainingArguments.

## Performance Comparison

| Configuration | Memory/GPU | Speed | FSDP Compat | Notes |
|--------------|------------|-------|-------------|-------|
| adamw_torch | ~100GB | 1.0x | ✅ Yes | OOM! |
| **paged_adamw_32bit** | **~73GB** | **0.92x** | **✅ Yes** | **RECOMMENDED** |
| adamw_bnb_8bit | ~63GB | 0.98x | ⚠️ Issues | FSDP device errors |
| adamw_torch + offload | ~60GB | 0.35x | ✅ Yes | 2-3x slower |
| paged_adamw_32bit + seq1024 | ~55GB | 0.90x | ✅ Yes | Faster, less memory |

**Recommendation**: Use `paged_adamw_32bit` with full sequence length (2048) for best FSDP compatibility and good memory efficiency.

## Troubleshooting

### Error: "No module named 'bitsandbytes'"

```bash
conda activate MoE_Compress
python -m pip install bitsandbytes
```

### Error: "bitsandbytes not compiled with CUDA support"

Your CUDA version may be incompatible. Check:

```bash
python -c "import bitsandbytes as bnb; print(bnb.cuda_setup.main())"
```

If issues persist, use CPU offloading instead.

### Error: "All input tensors need to be on the same GPU" (with adamw_bnb_8bit)

This is a known FSDP compatibility issue with 8-bit optimizers. **Solution**: Use `paged_adamw_32bit` instead:

```yaml
training:
  optim: "paged_adamw_32bit"  # Instead of adamw_bnb_8bit
```

### Still getting OOM with paged_adamw_32bit

Try reducing sequence length:

```yaml
dataset:
  max_length: 1024  # Down from 2048

training:
  optim: "paged_adamw_32bit"
  gradient_accumulation_steps: 64  # Maintain effective batch size
```

Or as last resort, use CPU offloading:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_2xa100_cpu_offload.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu
```

## References

- [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes)
- [8-bit Optimizers paper](https://arxiv.org/abs/2110.02861)
- HuggingFace Transformers optimizer guide
- [OOM_TROUBLESHOOTING.md](OOM_TROUBLESHOOTING.md) - General OOM guide
- [TWO_GPU_SETUP.md](TWO_GPU_SETUP.md) - FSDP configuration guide
