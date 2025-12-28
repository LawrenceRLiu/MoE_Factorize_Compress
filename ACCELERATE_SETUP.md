# Accelerate Setup Guide

This guide explains how to use Accelerate for distributed recovery training on your 2x 80GB A100 GPUs.

## Quick Start

```bash
# Set environment variables (recommended)
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1  # IMPORTANT: Restrict to 2x A100s only

# Launch recovery training
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery
```

**Important:** Always use `CUDA_VISIBLE_DEVICES=0,1` to ensure training only uses your 2x 80GB A100 GPUs (cuda:0 and cuda:1), not all 8 GPUs on your system.

## Installation

If you haven't installed Accelerate yet:

```bash
pip install accelerate
```

## Configuration Files

### accelerate_config_2xa100.yaml

Pre-configured for your dual A100 setup with:
- **2 GPUs**: cuda:0 and cuda:1
- **FSDP**: Full sharding (ZeRO-3) for maximum memory efficiency
- **Mixed Precision**: BF16 (optimal for A100s)
- **No CPU Offload**: Not needed with 80GB GPUs
- **Gradient Checkpointing**: Supported via training config

### conf/recovery/two_gpu.yaml

Hydra config optimized for 2x A100s with:
- **Batch size**: 1 per device
- **Gradient accumulation**: 16 steps (effective batch size = 32)
- **Gradient checkpointing**: Enabled to save memory
- **Full FSDP sharding**: Maximum memory efficiency

### FSDP vs DeepSpeed

This setup uses **FSDP** (Fully Sharded Data Parallel), not DeepSpeed:
- **FSDP** is native to PyTorch (no extra dependencies needed)
- Better integrated with HuggingFace Trainer
- Equivalent to DeepSpeed ZeRO-3 for 2-GPU setups
- Simpler configuration and debugging

**You do NOT need DeepSpeed** for your 2x A100 setup. FSDP provides the same memory efficiency and is already configured.

## Usage Patterns

### Basic Launch

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    experiment_name=my_experiment
```

### With Two-GPU Config (Recommended)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment
```

### With Custom Settings

```bash
# Specify training budget in tokens (recommended)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment \
    recovery.training.max_tokens=5_000_000_000 \
    recovery.training.learning_rate=3e-5 \
    recovery.training.save_steps=1000
```

### Using Environment Variables (Recommended)

```bash
# Set once per session
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1  # Restrict to 2x A100s

# Then use shorter commands
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment
```

## Testing Your Setup

Before running full training, test your Accelerate configuration:

```bash
# Test multi-GPU communication
CUDA_VISIBLE_DEVICES=0,1 accelerate test --config_file accelerate_config_2xa100.yaml

# Quick training test with limited data
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=test_run \
    recovery.dataset.num_samples=1000 \
    recovery.training.max_steps=10
```

## Monitoring

### Check GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- Both GPUs (cuda:0 and cuda:1) active
- Similar memory usage on both GPUs (due to FSDP sharding)
- High GPU utilization during training

### WandB Logging

Metrics logged automatically:
- Training loss
- Learning rate
- GPU memory usage
- Training throughput

## Troubleshooting

### Issue: "No CUDA devices found"

```bash
# Check GPU visibility
nvidia-smi

# Verify PyTorch sees GPUs
python -c "import torch; print(torch.cuda.device_count())"
```

### Issue: "NCCL error" or communication timeout

```bash
# Set longer timeout
export NCCL_TIMEOUT=1800

# Enable NCCL debugging
export NCCL_DEBUG=INFO
```

### Issue: Out of memory

1. Reduce batch size in `conf/recovery/two_gpu.yaml`:
   ```yaml
   training:
     per_device_train_batch_size: 1  # Already at minimum
     gradient_accumulation_steps: 32  # Increase this instead
   ```

2. Enable CPU offload in `accelerate_config_2xa100.yaml`:
   ```yaml
   fsdp_config:
     fsdp_offload_params: true
   ```

### Issue: Accelerate not using config file

```bash
# Verify config is valid
accelerate test --config_file accelerate_config_2xa100.yaml

# Make sure you're passing the flag correctly
accelerate launch --config_file accelerate_config_2xa100.yaml scripts/...

# Or set as environment variable
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
```

## Advanced: Creating Custom Configs

To create a new Accelerate config interactively:

```bash
accelerate config

# Or specify output file
accelerate config --config_file my_custom_config.yaml
```

Follow the prompts to customize:
- Number of GPUs
- FSDP sharding strategy
- Mixed precision settings
- CPU offload options

## Performance Tips

1. **Use BF16**: Already enabled in config, optimal for A100s
2. **Monitor memory**: Keep an eye on `nvidia-smi` to ensure you're not underutilizing memory
3. **Gradient accumulation**: Increase if you can fit larger effective batch sizes
4. **Batch size**: Try increasing `per_device_train_batch_size` to 2 if memory allows
5. **Data loading**: Increase `dataloader_num_workers` if CPU is underutilized

## Comparison: Accelerate vs torchrun

| Feature | Accelerate | torchrun |
|---------|-----------|----------|
| Ease of use | ✅ Easier | ⚠️ More complex |
| HF Integration | ✅ Native | ⚠️ Manual |
| Config management | ✅ YAML files | ⚠️ CLI flags |
| Error messages | ✅ Better | ⚠️ Cryptic |
| Flexibility | ✅ More options | ✅ Lower-level control |
| **Recommendation** | **Use for this project** | Use if you need low-level control |

## Full Example Workflow

```bash
# 1. Set up environment (once per session)
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1  # Restrict to 2x A100s

# 2. Test configuration
accelerate test

# 3. Quick test run
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=test \
    recovery.dataset.num_samples=1000 \
    recovery.training.max_steps=10

# 4. Full training run
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=production_run \
    recovery.training.max_steps=10000 \
    recovery.training.save_steps=500

# 5. Monitor in WandB or watch nvidia-smi
watch -n 1 nvidia-smi
```

## Next Steps

- See [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) for full recovery training documentation
- Check [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml) for hyperparameter tuning options
- Review [accelerate_config_2xa100.yaml](accelerate_config_2xa100.yaml) for FSDP settings
