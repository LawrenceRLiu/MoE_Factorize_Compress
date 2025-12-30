# Staged Learning Rate Training

This guide explains how to use the staged learning rate scheduling feature for recovery training.

**Implementation:** Uses a custom LR scheduler ([src/custom_lr_scheduler.py](src/custom_lr_scheduler.py)) that wraps HuggingFace's standard schedulers.

## Overview

The staged LR scheduler allows you to divide training into multiple stages, each with different learning rates for different parameter groups. This is useful for:

1. **Progressive unfreezing**: Start by training only compressed layers (cores + wrappers), then fine-tune the entire model
2. **Differential learning rates**: Use higher LRs for compression artifacts and lower LRs for pretrained components
3. **Curriculum learning**: Adapt training focus as the model recovers

## Key Features

- **Zero-LR freezing without gradient disabling**: Parameters with 0.0 learning rate are effectively frozen but maintain Adam states
- **Regex-based parameter matching**: Flexible pattern matching for parameter groups
- **Automatic stage transitions**: No manual intervention needed during training
- **WandB logging**: Track which parameters are active at each stage

## Configuration

### Basic Structure

In your recovery config (e.g., `conf/recovery/two_gpu.yaml`), add an `lr_schedule` section:

```yaml
lr_schedule:
  - fraction: 0.5  # Fraction of total training steps for this stage
    base_lr_multiplier: 0.0  # Default multiplier for all parameters
    param_patterns:  # Exceptions to the default
      - pattern: "regex_pattern_here"
        lr_multiplier: 1.0

  - fraction: 0.5  # Next stage
    base_lr_multiplier: 0.1
    param_patterns:
      - pattern: "regex_pattern_here"
        lr_multiplier: 1.0
```

### Parameter Name Patterns

The regex patterns match against the **full parameter name** in the model. Here are the key patterns for compressed MoE models:

#### Shared Cores
```yaml
pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.core$"
```
Matches: `model.layers.0.mlp.experts.gate_shared_core.core`, etc.

#### Low-Rank Wrappers (U and V matrices)
```yaml
pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
```
Matches: `model.layers.0.mlp.experts.gate_shared_core.experts.0.input_wrapper.U`, etc.

#### MoE Gate Weights
```yaml
pattern: ".*\\.mlp\\.gate\\.weight$"
```
Matches: `model.layers.0.mlp.gate.weight`

#### Attention Layers
```yaml
pattern: ".*\\.(q_proj|k_proj|v_proj|o_proj)\\.weight$"
```
Matches attention projection weights

#### Embeddings and LM Head
```yaml
pattern: "(embed_tokens\\.weight|lm_head\\.weight)"
```

## Example Configurations

### Example 1: Two-Stage Training (Default in two_gpu.yaml)

**Stage 1 (50% of steps)**: Train only compressed layers
**Stage 2 (50% of steps)**: Fine-tune entire model with lower LR

```yaml
lr_schedule:
  - fraction: 0.5
    base_lr_multiplier: 0.0  # Freeze everything
    param_patterns:
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.core$"
        lr_multiplier: 1.0  # Train cores
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
        lr_multiplier: 1.0  # Train wrappers

  - fraction: 0.5
    base_lr_multiplier: 0.01  # 1% LR for frozen layers
    param_patterns:
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.core$"
        lr_multiplier: 1.0  # Full LR for cores
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
        lr_multiplier: 1.0  # Full LR for wrappers
```

**Effective learning rates** (assuming `base_lr=2e-5`):
- Stage 1:
  - Cores/wrappers: 2e-5 (full LR)
  - Everything else: 0 (frozen)
- Stage 2:
  - Cores/wrappers: 2e-5 (full LR)
  - Attention/embeddings: 2e-7 (1% of base LR)

### Example 2: Three-Stage Progressive Unfreezing

```yaml
lr_schedule:
  # Stage 1: Wrappers only
  - fraction: 0.33
    base_lr_multiplier: 0.0
    param_patterns:
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
        lr_multiplier: 1.0

  # Stage 2: Wrappers + Cores
  - fraction: 0.33
    base_lr_multiplier: 0.0
    param_patterns:
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.(core|experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V))$"
        lr_multiplier: 1.0

  # Stage 3: Full model with differential LRs
  - fraction: 0.34
    base_lr_multiplier: 0.01
    param_patterns:
      - pattern: ".*\\.experts\\.(gate_shared_core|up_shared_core|down_shared_core)\\.(core|experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V))$"
        lr_multiplier: 1.0
```

### Example 3: Disable Staged Training

To use standard single learning rate for all parameters:

```yaml
lr_schedule: null
```

Or simply omit the `lr_schedule` section entirely.

## Usage

### Running with Staged LR

```bash
# Use the two_gpu config (has staged LR enabled by default)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment \
    recovery.training.max_tokens=2_000_000_000
```

### Customizing on the Command Line

```bash
# Override to disable staged training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    recovery.lr_schedule=null \
    experiment_name=my_experiment
```

### Creating Custom Stage Configs

Create a new config file `conf/recovery/my_custom.yaml`:

```yaml
# @package _global_

# Inherit from two_gpu
defaults:
  - two_gpu

# Override lr_schedule
lr_schedule:
  - fraction: 0.7
    base_lr_multiplier: 0.0
    param_patterns:
      - pattern: ".*\\.experts\\.*"
        lr_multiplier: 1.0

  - fraction: 0.3
    base_lr_multiplier: 0.05
    param_patterns:
      - pattern: ".*\\.experts\\.*"
        lr_multiplier: 1.0
```

Then use it:
```bash
recovery=my_custom
```

## Monitoring

### Logs

The scheduler logs stage transitions and parameter group summaries:

```
================================================================================
TRANSITIONING TO STAGE 1 at step 2500
  Stage runs from step 2500 to 5000
  Base LR multiplier: 0.01
  Parameters by LR multiplier: {0.01: 1234, 1.0: 567}
================================================================================
```

### WandB Metrics

The following metrics are logged to WandB:

- `param_schedule/stage`: Current stage index (0, 1, 2, ...)
- `param_schedule/lr_mult_X.XXX_count`: Number of parameters with each LR multiplier

You can create custom dashboards to visualize:
- Stage transitions over time
- Parameter group sizes
- Correlation between stage and loss/perplexity

## Implementation Details

### How It Works

1. **Scheduler Creation**: `RecoveryTrainer.create_scheduler()` wraps the base HF scheduler with `ParameterGroupLRScheduler`
2. **Initialization**: Builds a cache of parameter names from the model
3. **Each Step**:
   - Base scheduler updates all param groups to base LR (e.g., cosine with warmup)
   - Custom scheduler immediately applies parameter-specific multipliers
   - `get_last_lr()` returns the multiplied LRs for logging
4. **Stage Transitions**: Automatically detected based on step count
5. **Pattern Matching**: First matching pattern wins; falls back to `base_lr_multiplier`

### Zero-LR vs Gradient Disabling

We use **zero learning rate** rather than disabling gradients because:

- ✅ **Simpler**: No optimizer reinitialization needed
- ✅ **Preserves Adam state**: Momentum/variance useful when unfreezing
- ✅ **No warmup needed**: Transition between stages is seamless
- ✅ **FSDP compatible**: No param group changes during training
- ⚠️ **Memory overhead**: Adam states still stored for frozen params (~2x param memory)

For most use cases, this is a good tradeoff. The memory overhead is minimal compared to model size.

### Computational Overhead

The scheduler has negligible overhead:
- Regex matching: ~1ms per step (only checks current stage)
- LR updates: Direct assignment to optimizer param groups

## Debugging

### Check Parameter Names

To see actual parameter names in your model:

```bash
python -c "
from src.model_utils import load_compressed_model
import torch

model = load_compressed_model('path/to/checkpoint-0', 'Qwen/Qwen3-30B-A3B')
for name, param in model.named_parameters():
    print(name)
" | head -50
```

### Verify Pattern Matching

Test your regex patterns:

```python
import re

pattern = re.compile(r".*\.experts\..*\.core$")
test_names = [
    "model.layers.0.mlp.experts.gate_shared_core.core",  # Should match
    "model.layers.0.mlp.experts.gate_shared_core.experts.0.input_wrapper.U",  # Should not match
]

for name in test_names:
    print(f"{name}: {bool(pattern.search(name))}")
```

### Stage Fraction Warnings

If your stage fractions don't sum to ~1.0, you'll see a warning:

```
WARNING: Stage fractions sum to 0.75, expected ~1.0. This may cause unexpected behavior.
```

Make sure your fractions sum to 1.0 (or very close).

## FAQ

### Q: Can I change learning rates mid-training?

A: Yes! The scheduler recalculates LR at every step, so you can even modify configs during training (though this requires restarting).

### Q: What if I want to freeze parameters completely?

A: Set `lr_multiplier: 0.0` for those parameters. They'll receive no gradient updates.

### Q: Can I use this with other optimizers?

A: Yes, the scheduler works with any PyTorch optimizer, including:
- Standard PyTorch optimizers (Adam, AdamW, SGD, etc.)
- **torchao's AdamW8bit** (fully compatible - automatically handles tensor-based LRs)
- Other custom optimizers

The scheduler automatically detects whether the optimizer uses scalar or tensor learning rates and handles them appropriately.

### Q: How does this interact with warmup?

A: Warmup applies to the `base_lr` before multipliers. So a parameter with `lr_multiplier=0.5` will have 0.5x the warmed-up learning rate at each step.

### Q: Can I restart training from a checkpoint?

A: Yes, but the scheduler resets stage tracking based on `global_step`. Make sure to resume from the same stage configuration.

## Troubleshooting

### Issue: Parameters not being trained

**Symptom**: Loss not decreasing, params not changing

**Fix**: Check that your patterns actually match parameter names. Use the debugging commands above.

### Issue: Stage transitions happening at wrong time

**Symptom**: Stage 1 is too short/long

**Fix**: Verify that `max_steps` is calculated correctly. Stage boundaries are based on `total_steps`, which should match `max_steps` in TrainingArguments.

### Issue: OOM after unfreezing

**Symptom**: Works in Stage 1, crashes in Stage 2

**Fix**: When you unfreeze parameters, gradients are computed for more params. Try:
- Reducing `gradient_accumulation_steps` for Stage 2
- Using CPU offload config
- Reducing sequence length

## See Also

- [CLAUDE.md](CLAUDE.md) - Main project documentation
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) - Recovery training guide
- [OPTIMIZER_MEMORY_GUIDE.md](OPTIMIZER_MEMORY_GUIDE.md) - Memory optimization strategies
