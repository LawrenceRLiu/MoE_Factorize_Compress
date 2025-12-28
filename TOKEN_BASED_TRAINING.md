# Token-Based Training Duration

This document explains how to specify training duration in tokens instead of steps.

## Why Token-Based?

Specifying training budget in **tokens** is more intuitive than **steps** because:

1. **Standardized Metric**: Tokens are the universal currency in LLM training
2. **Easier Comparison**: Compare budgets across different setups (e.g., "I trained on 10B tokens")
3. **Hardware Agnostic**: The same token budget works regardless of GPU count or batch size
4. **Research Standard**: Papers report training budgets in tokens (e.g., "trained on 300B tokens")

## How It Works

### Configuration

In your config files ([conf/recovery/default.yaml](conf/recovery/default.yaml) or [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml)):

```yaml
training:
  max_tokens: 1_000_000_000  # 1 billion tokens
  max_steps: null  # Will be auto-calculated
```

### Automatic Calculation

The script automatically calculates `max_steps` using:

```python
tokens_per_batch = max_length × per_device_batch_size × gradient_accumulation × num_gpus
max_steps = max_tokens / tokens_per_batch
```

### Example Calculation

With the default 2x A100 configuration:
- `max_length`: 2048 tokens/sequence
- `per_device_batch_size`: 1
- `gradient_accumulation_steps`: 16
- `num_gpus`: 2

**Tokens per batch:**
```
2048 × 1 × 16 × 2 = 65,536 tokens/batch
```

**For 1B tokens:**
```
1,000,000,000 ÷ 65,536 = 15,258 steps
```

The script logs this calculation at startup:
```
================================================================================
Calculated max_steps from max_tokens:
  max_tokens: 1,000,000,000
  max_length (tokens/sequence): 2048
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  world_size (num GPUs): 2
  tokens_per_batch: 65,536
  => max_steps: 15,258
================================================================================
```

## Usage Examples

### Basic Usage

```bash
# Train on 1 billion tokens (default)
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=1B_tokens
```

### Custom Token Budget

```bash
# Train on 5 billion tokens
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=5B_tokens \
    recovery.training.max_tokens=5_000_000_000
```

### Common Token Budgets

```bash
# Small experiment (100M tokens)
recovery.training.max_tokens=100_000_000

# Medium experiment (1B tokens)
recovery.training.max_tokens=1_000_000_000

# Large experiment (10B tokens)
recovery.training.max_tokens=10_000_000_000

# Very large experiment (100B tokens)
recovery.training.max_tokens=100_000_000_000
```

### Override with Explicit Steps

If you need to set steps explicitly (overrides token calculation):

```bash
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=explicit_steps \
    recovery.training.max_steps=50000
```

## Token Budget Estimator

Use this table to estimate training time:

| Token Budget | Steps (2x A100) | Approx Time* |
|--------------|-----------------|--------------|
| 100M         | 1,526           | ~1 hour      |
| 500M         | 7,629           | ~5 hours     |
| 1B           | 15,258          | ~10 hours    |
| 5B           | 76,293          | ~50 hours    |
| 10B          | 152,587         | ~100 hours   |
| 50B          | 762,939         | ~500 hours   |
| 100B         | 1,525,878       | ~1000 hours  |

*Assuming ~20 tokens/sec throughput on 2x A100s with the default configuration

## Adjusting Configuration

If you change batch size or GPU count, the token budget stays the same:

### Example: Doubling Batch Size

```yaml
# Original: 1B tokens over 15,258 steps
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  max_tokens: 1_000_000_000
# => 15,258 steps

# Double batch size: still 1B tokens, but half the steps
training:
  per_device_train_batch_size: 2  # doubled
  gradient_accumulation_steps: 16
  max_tokens: 1_000_000_000
# => 7,629 steps (automatic!)
```

### Example: Adding More GPUs

```yaml
# Original: 2 GPUs, 1B tokens, 15,258 steps
# New: 4 GPUs, 1B tokens
# => Automatically calculates: 7,629 steps
```

The token budget remains constant regardless of hardware configuration!

## Best Practices

1. **Start Small**: Begin with 100M-1B tokens for initial experiments
2. **Scale Up**: Increase to 10B+ tokens for serious recovery training
3. **Monitor Loss**: Check if loss plateaus before max_tokens is reached
4. **Save Frequently**: Set `save_steps` to save checkpoints regularly
5. **Use WandB**: Track tokens/sec and total tokens processed

## Troubleshooting

### Issue: "max_steps is 0 or negative"

This means your token budget is too small. Minimum tokens needed:
```
min_tokens = max_length × per_device_batch_size × gradient_accumulation × num_gpus
```

For default config: min = 2048 × 1 × 16 × 2 = 65,536 tokens

### Issue: Training takes too long

Reduce `max_tokens`:
```bash
recovery.training.max_tokens=500_000_000  # 500M instead of 1B
```

Or increase batch size to process more tokens per step:
```bash
recovery.training.per_device_train_batch_size=2
```

## Implementation Details

The calculation happens in [scripts/run_recovery_training.py:257-286](scripts/run_recovery_training.py#L257-286):

```python
if max_steps is None or max_steps <= 0:
    max_tokens = training_config.max_tokens
    max_length = cfg.recovery.dataset.max_length
    per_device_batch_size = training_config.per_device_train_batch_size
    gradient_accumulation_steps = training_config.gradient_accumulation_steps

    # Get world size (number of GPUs)
    world_size = dist.get_world_size() if dist.is_initialized() else torch.cuda.device_count()

    # Calculate
    tokens_per_batch = max_length * per_device_batch_size * gradient_accumulation_steps * world_size
    max_steps = int(max_tokens / tokens_per_batch)
```

## See Also

- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) - Full recovery training documentation
- [ACCELERATE_SETUP.md](ACCELERATE_SETUP.md) - Accelerate configuration guide
- [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml) - 2-GPU configuration example
