# 2-GPU Training Setup Guide

## Your Question: FSDP vs Model Parallelism

You asked whether to use FSDP when you can fit 2 copies of Qwen-3-30B-A3B on your server. Here's the answer:

**Yes, use FSDP for recovery training, even if you can fit 2 copies!**

### Why Use FSDP?

1. **Memory Efficiency**: FSDP shards the model weights, gradients, AND optimizer states across GPUs
   - Without FSDP: Each GPU needs full model + gradients + optimizer states
   - With FSDP (full_shard): Model + gradients + optimizer states are split across GPUs
   - Optimizer states (Adam) are typically 2x the model size!

2. **Example for Qwen-3-30B-A3B**:
   - Compressed model: ~20-25GB (assuming 20-30% compression)
   - Gradients: ~20-25GB (same size as model)
   - Optimizer states (Adam): ~40-50GB (2x model size)
   - **Total per GPU without FSDP**: ~80-100GB ❌ Won't fit!
   - **Total per GPU with FSDP**: ~40-50GB ✅ Fits comfortably!

3. **Better Throughput**: FSDP can be faster than data parallelism for large models

### FSDP vs Other Options

| Approach | Memory per GPU | Speed | Use Case |
|----------|---------------|-------|----------|
| **FSDP (full_shard)** | Lowest | Good | Large models, limited VRAM (RECOMMENDED) |
| Data Parallel (DDP) | Highest | Best | Small models, lots of VRAM |
| Model Parallel | Medium | Slower | Very large models |

## Configuration for 2x A100 GPUs (cuda:0, cuda:1)

### Option 1: Using the Pre-configured Setup (Recommended)

```bash
# Use the two_gpu.yaml configuration
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=recovery_2gpu
```

### Option 2: Override Default Config

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    experiment_name=recovery_2gpu \
    recovery.fsdp.enabled=true \
    recovery.fsdp.fsdp_sharding_strategy=full_shard \
    recovery.model.device_map=null \
    recovery.training.gradient_checkpointing=true
```

## Important Settings Explained

### 1. CUDA_VISIBLE_DEVICES=0,1
This restricts PyTorch to only see GPUs 0 and 1 (your 2x A100s), ignoring the 6x A6000s.

### 2. torchrun --nproc_per_node=2
Launches 2 processes (one per GPU). This is the standard way to run distributed training.

### 3. FSDP Sharding Strategy
- **full_shard** (ZeRO-3): Shards parameters, gradients, and optimizer states
  - Most memory efficient
  - Slight communication overhead
  - **RECOMMENDED for your setup**

- **shard_grad_op** (ZeRO-2): Only shards gradients and optimizer states
  - Less memory efficient than full_shard
  - Less communication overhead
  - Use if you have enough memory

- **no_shard**: No sharding (equivalent to DDP)
  - Least memory efficient
  - Fastest (no sharding overhead)
  - Only use if model fits entirely on each GPU with optimizer states

### 4. device_map=null
**CRITICAL**: When using FSDP, you MUST set `device_map=null`.
- FSDP manages device placement automatically
- Using `device_map="auto"` will conflict with FSDP

### 5. gradient_checkpointing=true
Trades computation for memory by recomputing activations during backward pass.
- Reduces memory by ~30-40%
- Increases training time by ~20-30%
- **Highly recommended for large models**

## Memory Optimization Checklist

If you run into OOM (Out of Memory) errors:

1. ✅ **Use FSDP with full_shard** (already configured)
2. ✅ **Enable gradient checkpointing** (already configured)
3. ⚙️ **Reduce batch size**: `recovery.training.per_device_train_batch_size=1`
4. ⚙️ **Increase gradient accumulation**: `recovery.training.gradient_accumulation_steps=32`
5. ⚙️ **Reduce sequence length**: `recovery.dataset.max_length=1024`
6. 🔧 **Enable CPU offloading** (slower): `recovery.fsdp.fsdp_offload_params=true`
7. 🔧 **Use activation checkpointing**: Already enabled via gradient_checkpointing

## Example Commands

### Quick Test Run (100 steps)
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=test_2gpu \
    recovery.training.max_steps=100 \
    recovery.training.save_steps=50 \
    recovery.dataset.num_samples=1000
```

### Full Training Run
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=qwen3_recovery \
    recovery.training.num_train_epochs=1 \
    recovery.training.save_steps=500
```

### Resume from Checkpoint
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=qwen3_recovery \
    recovery.model.compressed_checkpoint=/path/to/checkpoint-1000
```

## Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

You should see:
- Both GPUs (0 and 1) at high utilization (~90-100%)
- Memory usage: ~40-60GB per GPU (with FSDP full_shard)
- 2 Python processes (one per GPU)

### WandB Logging
Monitor training progress at: https://wandb.ai/your-username/moe-compression

## Troubleshooting

### Problem: "CUDA out of memory"
**Solutions**:
1. Reduce batch size to 1: `recovery.training.per_device_train_batch_size=1`
2. Enable CPU offloading: `recovery.fsdp.fsdp_offload_params=true`
3. Reduce sequence length: `recovery.dataset.max_length=1024`

### Problem: "NCCL timeout" or communication errors
**Solutions**:
1. Check network connectivity between GPUs
2. Increase timeout: `export NCCL_TIMEOUT=1800`
3. Check NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`

### Problem: Only 1 GPU being used
**Solutions**:
1. Make sure you're using `torchrun --nproc_per_node=2`
2. Verify `CUDA_VISIBLE_DEVICES=0,1` is set
3. Check `nvidia-smi` shows both GPUs

### Problem: "device_map conflicts with FSDP"
**Solution**: Set `recovery.model.device_map=null`

## Performance Expectations

With 2x A100 80GB and FSDP:
- **Memory per GPU**: ~40-60GB (leaving headroom)
- **Training speed**: ~0.5-1.0 seconds per step (batch size 1, seq len 2048)
- **Effective batch size**: `per_device_batch_size * num_gpus * gradient_accumulation_steps`
  - Example: 1 * 2 * 16 = 32

## Key Takeaways

1. **Always use FSDP** for large model training on multiple GPUs
2. **Set device_map=null** when using FSDP
3. **Use CUDA_VISIBLE_DEVICES** to restrict to A100s only
4. **Use torchrun** to launch distributed training
5. **Enable gradient checkpointing** for memory savings
6. **Start with the two_gpu.yaml config** - it's pre-optimized!

## Alternative: If You Want to Use All 8 GPUs

If you want to use all 8 GPUs (2x A100 + 6x A6000) for even faster training:

```bash
# Use all 8 GPUs
torchrun --nproc_per_node=8 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=recovery_8gpu \
    recovery.training.per_device_train_batch_size=1 \
    recovery.training.gradient_accumulation_steps=4
```

**Note**: FSDP will handle the heterogeneous memory (80GB + 48GB) automatically, but you may need to tune batch sizes if the A6000s run out of memory.
