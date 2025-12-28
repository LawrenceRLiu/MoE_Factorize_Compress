# OOM (Out of Memory) Troubleshooting Guide

If you're getting CUDA OOM errors during recovery training, follow this guide.

## Understanding the Error

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 1 has a total capacity of 79.15 GiB of which 2.19 MiB is free.
```

This error typically means the model, gradients, and optimizer states don't fit in GPU memory.

## Expected Memory Usage

For an 8B parameter model with ~3B active parameters:
- **Model parameters (bf16)**: ~6 GB (3B × 2 bytes)
- **Gradients (bf16)**: ~6 GB
- **Optimizer states (AdamW)**: ~24 GB (8 bytes per param × 3B × 2 states)
- **Activations**: ~10-20 GB (depends on sequence length and batch size)
- **Total per GPU WITHOUT FSDP**: ~46-56 GB

With **FSDP FULL_SHARD** on 2 GPUs:
- Everything divided by 2: ~23-28 GB per GPU
- **Should easily fit on 80GB A100s!**

If you're hitting OOM, FSDP is likely **not working correctly**.

## Root Causes & Solutions

### 1. ✅ FSDP Not Actually Running (MOST LIKELY)

**Problem**: When using Accelerate, FSDP config in TrainingArguments conflicts with Accelerate's FSDP config.

**Solution**: The script now automatically detects Accelerate and skips TrainingArguments FSDP config. You should see this log:

```
================================================================================
ACCELERATE DETECTED:
  FSDP will be configured by Accelerate, not TrainingArguments
  Skipping TrainingArguments FSDP configuration to avoid conflicts
================================================================================
```

If you DON'T see this message, Accelerate might not be detected. Check:

```bash
# Verify Accelerate config is set
echo $ACCELERATE_CONFIG_FILE

# Should output: accelerate_config_2xa100.yaml
```

### 2. ✅ Model Loading Before FSDP Wrapping

**Problem**: Model loads fully into GPU memory BEFORE FSDP can shard it.

**Check**: Look for warnings about model device placement during loading.

**Solution**: The config already sets `device_map: null` which is correct. FSDP will handle device placement.

### 3. ✅ Try CPU Offloading

If FSDP is working but you still hit OOM, use CPU offloading:

```bash
# Use the CPU offload config
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100_cpu_offload.yaml
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery
```

**Trade-off**: ~20-30% slower, but should never OOM.

### 4. ✅ Reduce Sequence Length

If still hitting OOM, reduce sequence length:

```bash
recovery.dataset.max_length=1024  # Down from 2048
```

This halves activation memory.

### 5. ✅ Disable Gradient Checkpointing (Counter-intuitive!)

Sometimes gradient checkpointing can cause issues with FSDP. Try disabling it:

```bash
recovery.training.gradient_checkpointing=false
```

## Diagnostic Commands

### Check if FSDP is Working

Run training and look for these logs:

```bash
# 1. Check for Accelerate detection
grep "ACCELERATE DETECTED" training.log

# 2. Check FSDP initialization
grep "FSDP" training.log

# 3. Monitor GPU memory during training
watch -n 1 nvidia-smi
```

With FSDP working correctly, you should see:
- **Similar memory usage on both GPUs** (~25-30 GB each)
- **NOT 76 GB on one GPU** (that means FSDP isn't sharding)

### Check Accelerate Config

```bash
# Verify Accelerate sees your config
accelerate env

# Test Accelerate with FSDP
CUDA_VISIBLE_DEVICES=0,1 accelerate test --config_file accelerate_config_2xa100.yaml
```

## Step-by-Step Debugging

### Step 1: Verify Environment

```bash
# Check GPU visibility
CUDA_VISIBLE_DEVICES=0,1 python -c "import torch; print(f'Visible GPUs: {torch.cuda.device_count()}')"
# Should output: Visible GPUs: 2

# Check Accelerate config
echo $ACCELERATE_CONFIG_FILE
# Should output: accelerate_config_2xa100.yaml (or full path)

# Verify config file exists
ls -lh accelerate_config_2xa100.yaml
```

### Step 2: Try Minimal Test

```bash
export CUDA_VISIBLE_DEVICES=0,1
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml

# Very short training run to test FSDP
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=oom_test \
    recovery.dataset.num_samples=100 \
    recovery.training.max_steps=2 \
    recovery.training.per_device_train_batch_size=1
```

Monitor `nvidia-smi` in another terminal:
```bash
watch -n 0.5 nvidia-smi
```

### Step 3: Check FSDP Sharding

If the test runs successfully, check the memory pattern:

✅ **GOOD** (FSDP working):
```
GPU 0: 28 GB / 80 GB
GPU 1: 28 GB / 80 GB
```

❌ **BAD** (FSDP not working):
```
GPU 0: 76 GB / 80 GB  <- Model on one GPU!
GPU 1: 4 GB / 80 GB
```

or

❌ **BAD** (No sharding):
```
GPU 0: 76 GB / 80 GB
GPU 1: 76 GB / 80 GB  <- Duplicated!
```

### Step 4: If Still OOM, Try CPU Offload

```bash
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100_cpu_offload.yaml
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=cpu_offload_test \
    recovery.dataset.num_samples=100 \
    recovery.training.max_steps=2
```

With CPU offload, memory should be much lower:
```
GPU 0: 15-20 GB / 80 GB
GPU 1: 15-20 GB / 80 GB
```

## Configuration Matrix

| Config | Memory/GPU | Speed | Use When |
|--------|-----------|-------|----------|
| `accelerate_config_2xa100.yaml` | ~28 GB | Fast | Default (should work) |
| `accelerate_config_2xa100_cpu_offload.yaml` | ~18 GB | 70% speed | Hitting OOM |
| `max_length=1024` | ~18 GB | Fast | Need more memory |
| `gradient_checkpointing=false` | ~35 GB | Faster | FSDP issues |

## Still Having Issues?

### Gather Debug Info

```bash
# 1. Save full training log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=debug \
    recovery.dataset.num_samples=10 \
    recovery.training.max_steps=1 \
    2>&1 | tee training_debug.log

# 2. Check Accelerate version
pip show accelerate

# 3. Check transformers version
pip show transformers

# 4. Check torch version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Known Version Issues

- **transformers < 4.44**: FSDP support may be incomplete
- **accelerate < 0.27**: FSDP config format changed
- **torch < 2.1**: FSDP features may be limited

## Emergency Fallback: No FSDP

If FSDP absolutely won't work, you can try running on a single GPU with aggressive memory optimization:

```bash
# Single GPU, heavy optimization
CUDA_VISIBLE_DEVICES=0 python scripts/run_recovery_training.py \
    --config-name config \
    experiment_name=single_gpu \
    recovery.training.per_device_train_batch_size=1 \
    recovery.training.gradient_accumulation_steps=64 \
    recovery.training.gradient_checkpointing=true \
    recovery.dataset.max_length=1024 \
    recovery.fsdp.enabled=false
```

⚠️ **Warning**: This will be MUCH slower and might still OOM for large models.

## Summary Checklist

- [ ] `CUDA_VISIBLE_DEVICES=0,1` is set
- [ ] `ACCELERATE_CONFIG_FILE` is set to accelerate config
- [ ] Training logs show "ACCELERATE DETECTED"
- [ ] `nvidia-smi` shows similar memory on both GPUs
- [ ] `per_device_train_batch_size=1` (already at minimum)
- [ ] `gradient_checkpointing=true` in config
- [ ] Model `device_map=null` in config
- [ ] If OOM persists, try `accelerate_config_2xa100_cpu_offload.yaml`

## Quick Fixes Summary

```bash
# Fix 1: Make sure Accelerate is detected
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1

# Fix 2: If still OOM, use CPU offload
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100_cpu_offload.yaml

# Fix 3: If still OOM, reduce sequence length
recovery.dataset.max_length=1024

# Fix 4: Check that only 2 GPUs are visible
nvidia-smi --list-gpus  # Should show exactly 2 GPUs
```
