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
python scripts/run_recovery_training.py \
    experiment_name=my_experiment \
    recovery.training.learning_rate=1e-5 \
    recovery.training.per_device_train_batch_size=2 \
    recovery.training.save_steps=1000 \
    recovery.training.max_steps=10000
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

Using `torchrun` for multi-GPU training:

```bash
torchrun --nproc_per_node=8 scripts/run_recovery_training.py \
    experiment_name=distributed_recovery
```

Or with `accelerate`:

```bash
accelerate launch scripts/run_recovery_training.py \
    experiment_name=distributed_recovery
```

## Configuration Reference

### Key Configuration Options

See `conf/recovery/default.yaml` for all options. Key sections:

#### Dataset
- `dataset.name`: HuggingFace dataset name
- `dataset.max_length`: Maximum sequence length
- `dataset.streaming`: Stream large datasets
- `dataset.num_samples`: Limit samples (for testing)

#### Training
- `training.learning_rate`: Learning rate (default: 2e-5)
- `training.per_device_train_batch_size`: Batch size per device
- `training.gradient_accumulation_steps`: Gradient accumulation
- `training.save_steps`: Checkpoint saving frequency
- `training.max_steps`: Maximum training steps (-1 for full epochs)
- `training.gradient_checkpointing`: Enable gradient checkpointing

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
python scripts/run_recovery_training.py \
    recovery.training.per_device_train_batch_size=1 \
    recovery.training.gradient_accumulation_steps=16 \
    recovery.training.gradient_checkpointing=true \
    recovery.fsdp.fsdp_sharding_strategy=full_shard
```

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

1. Reduce batch size: `recovery.training.per_device_train_batch_size=1`
2. Enable gradient checkpointing: `recovery.training.gradient_checkpointing=true`
3. Use full FSDP sharding: `recovery.fsdp.fsdp_sharding_strategy=full_shard`
4. Enable CPU offloading: `recovery.fsdp.fsdp_offload_params=true`

### FSDP Issues

1. Ensure all processes can communicate (check network/firewall)
2. Use `torchrun` or `accelerate` for multi-GPU training
3. Verify CUDA and NCCL versions are compatible

### Dataset Loading

1. For large datasets, use `streaming=true`
2. If text column is different, set `recovery.dataset.text_column=your_column`
3. For quick testing, limit samples: `recovery.dataset.num_samples=1000`

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

# Step 3: Start recovery training (Phase 2)
python scripts/run_recovery_training.py experiment_name=my_experiment

# Monitor progress on WandB
```
