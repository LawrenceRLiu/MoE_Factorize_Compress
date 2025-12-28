# Recovery Training Documentation

## Overview

Recovery training is Phase 2 of the MoE compression pipeline. After zero-shot initialization (Phase 1), we perform recovery pretraining on the compressed model to restore performance.

## Key Features

- **Minimal HuggingFace Trainer Subclass**: Uses `RecoveryTrainer`, a minimal extension of the HuggingFace `Trainer` class
- **FSDP Support**: Full support for FSDP (Fully Sharded Data Parallel) for efficient distributed training
- **Hydra Configuration**: Easy configuration management through YAML files
- **Checkpoint Format**: Saves checkpoints as `checkpoint-{step}` for seamless async evaluation
- **WandB Integration**: Built-in experiment tracking with Weights & Biases

## Architecture

### Files

- **Configuration**: `conf/recovery/default.yaml` - All recovery training settings
- **Trainer**: `src/recovery_trainer.py` - RecoveryTrainer class and utilities
- **Script**: `scripts/run_recovery_training.py` - Main training script with Hydra

### Components

1. **RecoveryTrainer**: Minimal subclass of `transformers.Trainer`
   - Standard language modeling loss (no distillation)
   - Custom checkpoint callback for compressed models
   - FSDP-aware state dict saving

2. **CompressedModelCheckpointCallback**: Handles saving compressed models
   - Saves to `checkpoint-{step}` format
   - FSDP-compatible checkpoint gathering
   - Saves training metadata for async evaluation

## Usage

### Basic Usage

```bash
python scripts/run_recovery_training.py \
    experiment_name=my_recovery_experiment
```

This will:
1. Load the compressed model from `checkpoint-0` (default from zero-shot init)
2. Use settings from `conf/recovery/default.yaml`
3. Save checkpoints to `models/{model_name}/{experiment_name}/checkpoints/`

### Custom Checkpoint

To start from a specific checkpoint:

```bash
python scripts/run_recovery_training.py \
    recovery.model.compressed_checkpoint=/path/to/checkpoint-0 \
    experiment_name=my_experiment
```

### Override Training Settings

```bash
# Specify training budget in tokens (recommended)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment \
    recovery.training.max_tokens=2_000_000_000 \
    recovery.training.learning_rate=1e-5 \
    recovery.training.save_steps=1000

# Or override with explicit max_steps
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment \
    recovery.training.max_steps=20000 \
    recovery.training.learning_rate=1e-5
```

### FSDP Configuration

FSDP is enabled by default. To customize:

```bash
python scripts/run_recovery_training.py \
    recovery.fsdp.fsdp_sharding_strategy=shard_grad_op \
    recovery.fsdp.fsdp_offload_params=true
```

FSDP sharding strategies:
- `full_shard`: Full ZeRO-3 style sharding (default, most memory efficient)
- `shard_grad_op`: ZeRO-2 style (shard gradients and optimizer states)
- `no_shard`: No sharding (DDP-style)

### Dataset Configuration

```bash
python scripts/run_recovery_training.py \
    recovery.dataset.name=HuggingFaceFW/fineweb-edu \
    recovery.dataset.max_length=2048 \
    recovery.dataset.streaming=true
```

### Testing with Limited Data

For quick testing:

```bash
python scripts/run_recovery_training.py \
    recovery.dataset.num_samples=1000 \
    recovery.training.max_steps=100 \
    recovery.training.save_steps=50
```

### Distributed Training

**Recommended:** Using `accelerate` for multi-GPU training:

```bash
# Using the provided config for 2x 80GB A100s
accelerate launch --config_file accelerate_config_2xa100.yaml scripts/run_recovery_training.py \
    experiment_name=distributed_recovery

# Or set as default and use simplified command
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
accelerate launch scripts/run_recovery_training.py \
    experiment_name=distributed_recovery
```

**Alternative:** Using `torchrun` (if you prefer direct PyTorch control):

```bash
torchrun --nproc_per_node=2 scripts/run_recovery_training.py \
    experiment_name=distributed_recovery
```

**For 2x 80GB A100s specifically (cuda:0 and cuda:1 only):**

```bash
# Recommended command for dual A100 setup
# IMPORTANT: Use CUDA_VISIBLE_DEVICES to restrict to only the 2x A100s
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery_2gpu
```

### Why Accelerate is Recommended

Accelerate offers several advantages over torchrun for this project:

1. **Better HuggingFace Integration**: Seamless compatibility with `transformers.Trainer`
2. **Flexible Configuration**: Version-controlled config files (`accelerate_config_2xa100.yaml`)
3. **Easier Scaling**: Same command works for 1-GPU, 2-GPU, or multi-node setups
4. **Better Debugging**: More informative error messages and better handling of FSDP edge cases
5. **Advanced Features**: Built-in support for DeepSpeed, mixed precision, and gradient accumulation
6. **Unified Interface**: No need to change launch commands when scaling up/down

The provided `accelerate_config_2xa100.yaml` is optimized for 2x 80GB A100s with:
- Full FSDP sharding (ZeRO-3) for maximum memory efficiency
- BF16 mixed precision for A100 performance
- No CPU offload (not needed with 80GB GPUs)
- Gradient checkpointing support via training config

**Note on DeepSpeed:** This setup uses **FSDP** (Fully Sharded Data Parallel), not DeepSpeed. FSDP is:
- Native to PyTorch (no extra dependencies)
- Better integrated with HuggingFace Trainer
- Sufficient for 2-GPU setups
- Similar to DeepSpeed ZeRO-3 in functionality

DeepSpeed is only needed for:
- Very large multi-node clusters
- Specific optimizations like ZeRO-Offload to NVMe
- DeepSpeed-specific features

For your 2x A100 setup, **FSDP is the right choice** and is already configured.

## Configuration Reference

### Key Configuration Options

See `conf/recovery/default.yaml` for all options. Key sections:

#### Dataset
- `dataset.name`: HuggingFace dataset name
- `dataset.max_length`: Maximum sequence length
- `dataset.streaming`: Stream large datasets
- `dataset.num_samples`: Limit samples (for testing)

#### Training
- `training.max_tokens`: Total training budget in tokens (recommended, e.g., `1_000_000_000` for 1B tokens)
- `training.max_steps`: Maximum training steps (auto-calculated from `max_tokens`, or set explicitly to override)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `training.per_device_train_batch_size`: Batch size per device
- `training.gradient_accumulation_steps`: Gradient accumulation
- `training.save_steps`: Checkpoint saving frequency
- `training.gradient_checkpointing`: Enable gradient checkpointing
- `training.save_total_limit`: Set to `null` to save all checkpoints (recommended for async eval)

**Training Duration:** Specify your training budget in tokens using `max_tokens` (e.g., `1_000_000_000` for 1B tokens). The script will automatically calculate `max_steps` based on:
- Sequence length (`dataset.max_length`)
- Batch size (`per_device_train_batch_size`)
- Gradient accumulation steps
- Number of GPUs

You can also override by setting `max_steps` explicitly, which takes precedence over `max_tokens`.

#### FSDP
- `fsdp.enabled`: Enable/disable FSDP
- `fsdp.fsdp_sharding_strategy`: Sharding strategy
- `fsdp.fsdp_offload_params`: Offload parameters to CPU
- `fsdp.fsdp_state_dict_type`: How to save checkpoints

#### WandB
- `wandb.enabled`: Enable WandB logging
- `wandb.run_name`: Custom run name (auto-generated if null)
- `wandb.tags`: Tags for the run

## Checkpoint Format

Checkpoints are saved in the following structure:

```
checkpoints/
├── checkpoint-0/           # Zero-shot initialization
├── checkpoint-500/         # Recovery checkpoint at step 500
│   ├── pytorch_model.bin   # Model weights
│   ├── config.json         # Model configuration
│   ├── tokenizer.json      # Tokenizer files
│   └── training_metadata.json  # Training info
├── checkpoint-1000/
└── ...
```

Each checkpoint includes:
- `pytorch_model.bin`: Full model state dict
- `config.json`: Model configuration
- Tokenizer files
- `training_metadata.json`: Step, epoch, and model info

**Note:** By design, the trainer saves **all checkpoints** (no `save_total_limit`). The async evaluation script is responsible for monitoring and managing checkpoints, including cleanup if needed.

## Integration with Async Evaluation

The checkpoint format (`checkpoint-{step}`) is designed to work seamlessly with async evaluation:

1. Recovery training saves checkpoints every N steps
2. Async evaluation monitors the checkpoints directory
3. New checkpoints are automatically evaluated
4. Results are logged to WandB

To run async evaluation alongside recovery training:

```bash
# Terminal 1: Start recovery training
python scripts/run_recovery_training.py experiment_name=my_experiment

# Terminal 2: Start async evaluation
python scripts/run_async_eval.py experiment_name=my_experiment
```

## Memory Optimization

For large models, use these strategies:

1. **Enable FSDP**: Full sharding reduces memory per GPU
2. **Gradient Checkpointing**: Trade compute for memory
3. **Smaller Batch Size**: Reduce per-device batch size
4. **Gradient Accumulation**: Increase steps to maintain effective batch size
5. **CPU Offloading**: Enable `fsdp.fsdp_offload_params=true`

Example for 48GB GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    recovery.training.per_device_train_batch_size=1 \
    recovery.training.gradient_accumulation_steps=16 \
    recovery.training.gradient_checkpointing=true \
    recovery.fsdp.fsdp_sharding_strategy=full_shard
```

**Disk Space Management**: Since all checkpoints are saved (no limit), ensure you have sufficient disk space. Each checkpoint for a 30B model with MoE can be 10-30GB. The async evaluation script can be configured to clean up old checkpoints after evaluation.

## Monitoring

### WandB Metrics

The trainer logs:
- `loss`: Training loss
- `learning_rate`: Current learning rate
- `epoch`: Current epoch
- `step`: Global step

### Local Logs

Check training logs in the output directory:
- `{output_dir}/recovery_config.yaml`: Saved configuration
- `{checkpoints_dir}/trainer_state/`: Trainer state and logs

## Troubleshooting

### Out of Memory

**If you encounter OOM errors, see [OOM_TROUBLESHOOTING.md](OOM_TROUBLESHOOTING.md) for a comprehensive guide.**

Quick fixes:
1. **Use CPU offload config**: `export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100_cpu_offload.yaml`
2. Reduce sequence length: `recovery.dataset.max_length=1024`
3. Verify FSDP is working: Check that both GPUs show similar memory usage in `nvidia-smi`

**Note**: Batch size is already at minimum (1). The issue is likely that FSDP isn't sharding properly.

### FSDP Issues

1. Ensure all processes can communicate (check network/firewall)
2. **Recommended**: Use `accelerate launch --config_file accelerate_config_2xa100.yaml` for multi-GPU training
3. Verify CUDA and NCCL versions are compatible
4. Check GPU visibility: `nvidia-smi` should show both GPUs
5. For Accelerate-specific debugging: `accelerate test --config_file accelerate_config_2xa100.yaml`

### Dataset Loading

1. For large datasets, use `streaming=true`
2. If text column is different, set `recovery.dataset.text_column=your_column`
3. For quick testing, limit samples: `recovery.dataset.num_samples=1000`

### Specifying Training Duration

**Recommended approach:** Use `max_tokens` to specify your training budget in tokens:

```bash
# Train on 1 billion tokens
recovery.training.max_tokens=1_000_000_000

# Train on 10 billion tokens
recovery.training.max_tokens=10_000_000_000
```

The script will automatically calculate the required `max_steps` based on:
- Your sequence length (default: 2048 tokens)
- Batch size and gradient accumulation
- Number of GPUs

**Alternative:** Set `max_steps` explicitly to override automatic calculation:
```bash
recovery.training.max_steps=10000
```

**Note:** The `max_tokens` approach is recommended as it makes it easier to compare training budgets across different batch size and GPU configurations.

## Next Steps

After recovery training:

1. **Monitor checkpoints**: Use async evaluation to track performance
2. **Analyze results**: Compare metrics to original model baseline
3. **Iterate**: Adjust hyperparameters based on results
4. **Evaluate final model**: Run comprehensive evaluation on best checkpoint

## Example Workflow

Complete workflow from zero-shot init to recovery training:

```bash
# Step 1: Run zero-shot initialization (Phase 1)
python scripts/run_compression.py experiment_name=my_experiment

# Step 2: Start async evaluation
python scripts/run_async_eval.py experiment_name=my_experiment &

# Step 3: Start recovery training (Phase 2) - RECOMMENDED
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment

# Monitor progress on WandB
```

### Example for 2x 80GB A100s (Full Command)

```bash
# Export config to avoid repeating flags
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1  # Restrict to 2x A100s only

# Start recovery training with optimized settings
# Train on 2 billion tokens
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_2gpu_recovery \
    recovery.training.max_tokens=2_000_000_000 \
    recovery.training.per_device_train_batch_size=1 \
    recovery.training.gradient_accumulation_steps=16 \
    recovery.training.save_steps=500
```

The script will print the calculated `max_steps` at startup:
```
================================================================================
Calculated max_steps from max_tokens:
  max_tokens: 2,000,000,000
  max_length (tokens/sequence): 2048
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  world_size (num GPUs): 2
  tokens_per_batch: 65,536
  => max_steps: 30,517
================================================================================
```
