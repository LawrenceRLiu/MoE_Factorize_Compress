# Recovery Training Fixes - Summary

This document summarizes the fixes applied to resolve issues with Accelerate-based recovery training.

## Issues Addressed

### 1. ✅ TrainingArguments Parameter Error
**Issue:** `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

**Root Cause:** The parameter was renamed from `evaluation_strategy` to `eval_strategy` in newer versions of transformers.

**Fix:** Updated [scripts/run_recovery_training.py:295](scripts/run_recovery_training.py#L295)
```python
# Changed from:
"evaluation_strategy": training_config.evaluation_strategy,

# To:
"eval_strategy": training_config.evaluation_strategy,
```

### 1b. ✅ compute_loss() Parameter Error
**Issue:** `TypeError: RecoveryTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'`

**Root Cause:** Newer versions of transformers (4.46+) added the `num_items_in_batch` parameter to `compute_loss()` for better gradient accumulation handling.

**Fix:** Updated [src/recovery_trainer.py:166](src/recovery_trainer.py#L166)
```python
# Changed from:
def compute_loss(self, model, inputs, return_outputs=False):

# To:
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
```

### 2. ✅ Running on All GPUs Instead of Just 2x A100s
**Issue:** Training was using all 8 GPUs instead of just the 2x 80GB A100s (cuda:0 and cuda:1)

**Root Cause:** While the Accelerate config specifies `gpu_ids: "0,1"`, it doesn't prevent other GPUs from being visible to the process.

**Fix:** Use `CUDA_VISIBLE_DEVICES` environment variable to restrict GPU visibility:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml ...
```

**Updated Files:**
- [accelerate_config_2xa100.yaml](accelerate_config_2xa100.yaml) - Added usage notes
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) - All examples now include `CUDA_VISIBLE_DEVICES=0,1`
- [ACCELERATE_SETUP.md](ACCELERATE_SETUP.md) - All examples now include `CUDA_VISIBLE_DEVICES=0,1`

### 3. ✅ DeepSpeed Confusion
**Issue:** Unclear whether DeepSpeed needs to be configured separately

**Answer:** **NO, you do NOT need DeepSpeed.** This setup uses FSDP (Fully Sharded Data Parallel), which is:
- Native to PyTorch (no extra dependencies)
- Better integrated with HuggingFace Trainer
- Equivalent to DeepSpeed ZeRO-3 for your use case
- Already fully configured

**Documentation Added:**
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md#L156-167) - "Note on DeepSpeed" section
- [ACCELERATE_SETUP.md](ACCELERATE_SETUP.md#L48-56) - "FSDP vs DeepSpeed" section

### 4. ✅ Checkpoint Save Limit Removed
**Issue:** `save_total_limit: 5` was limiting checkpoints, but async eval needs all checkpoints

**Solution:** Set `save_total_limit: null` to save all checkpoints. The async evaluation script is responsible for monitoring and managing checkpoint cleanup, not the trainer.

**Files Updated:**
- [conf/recovery/default.yaml](conf/recovery/default.yaml#L39) - Set to `null`
- [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml#L54) - Set to `null`
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md#L222) - Added checkpoint management note

**Note:** Ensure sufficient disk space. Each checkpoint for a 30B MoE model can be 10-30GB.

### 5. ✅ Streaming Dataset Requires max_steps
**Issue:** `ValueError: The train_dataset does not implement __len__, max_steps has to be specified`

**Root Cause:** Streaming datasets (default with `streaming=true`) don't have a fixed length, so the Trainer can't calculate the number of steps from `num_train_epochs`. The learning rate scheduler needs to know the total steps in advance.

**Solution:** Set explicit `max_steps` in config instead of using `-1` (which means "use epochs"):

**Files Updated:**
- [conf/recovery/default.yaml](conf/recovery/default.yaml#L26) - Set `max_steps: 10000`
- [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml#L41) - Set `max_steps: 10000`
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md#L313-322) - Added troubleshooting section

**Note:** Adjust `max_steps` based on your training budget. For reference:
- With batch_size=1, gradient_accumulation=16, 2 GPUs: effective_batch_size = 32
- 10,000 steps × 32 = 320,000 sequences
- At 2048 tokens/sequence = ~655M tokens

### 6. ✅ Token-Based Training Duration (Enhancement)
**Enhancement:** Added ability to specify training budget in tokens instead of steps

**Implementation:**
- Added `max_tokens` parameter to config files (e.g., `max_tokens: 1_000_000_000` for 1B tokens)
- Script automatically calculates `max_steps` from `max_tokens` based on:
  - Sequence length (`max_length`)
  - Batch size (`per_device_train_batch_size`)
  - Gradient accumulation (`gradient_accumulation_steps`)
  - Number of GPUs (world size)
- Setting `max_steps` explicitly overrides automatic calculation

**Benefits:**
- More intuitive to specify training budget (e.g., "train on 10B tokens")
- Easier to compare budgets across different batch size/GPU configurations
- Automatic calculation logged at startup for transparency

**Files Updated:**
- [conf/recovery/default.yaml](conf/recovery/default.yaml#L25-27) - Added `max_tokens: 1_000_000_000`
- [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml#L40-42) - Added `max_tokens: 1_000_000_000`
- [scripts/run_recovery_training.py](scripts/run_recovery_training.py#L257-286) - Added calculation logic
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md#L329-351) - Added documentation

**Example:**
```bash
# Train on 2 billion tokens (much clearer than "train for X steps")
recovery.training.max_tokens=2_000_000_000
```

## Corrected Usage

### Quick Start (Recommended)

```bash
# Set environment variables once per session
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml
export CUDA_VISIBLE_DEVICES=0,1  # Restrict to 2x A100s only

# Launch recovery training (will train on 1B tokens by default)
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery

# Or specify custom token budget
accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery \
    recovery.training.max_tokens=5_000_000_000  # 5B tokens
```

### Full Command (One-liner)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_recovery
```

## What's Configured

### FSDP (Not DeepSpeed)
- **Sharding Strategy**: FULL_SHARD (equivalent to ZeRO-3)
- **Mixed Precision**: BF16 (optimal for A100s)
- **CPU Offload**: Disabled (not needed with 80GB GPUs)
- **Gradient Checkpointing**: Enabled via training config

### Training Settings (2x A100s)
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 1 × 2 GPUs × 16 = 32
- **Sequence Length**: 2048 tokens
- **Learning Rate**: 2e-5 with cosine schedule

## Verification Checklist

Before running full training:

1. ✅ Set `CUDA_VISIBLE_DEVICES=0,1`
2. ✅ Use `--config-name config recovery=two_gpu`
3. ✅ Check that only 2 GPUs show activity in `nvidia-smi`
4. ✅ Verify similar memory usage on both GPUs (FSDP sharding)
5. ✅ Confirm no DeepSpeed installation is needed

## Testing Your Setup

```bash
# 1. Check GPU visibility
CUDA_VISIBLE_DEVICES=0,1 python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Expected output: GPUs: 2

# 2. Test Accelerate configuration
CUDA_VISIBLE_DEVICES=0,1 accelerate test --config_file accelerate_config_2xa100.yaml

# 3. Quick training test
export CUDA_VISIBLE_DEVICES=0,1
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100.yaml

accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=test_run \
    recovery.dataset.num_samples=1000 \
    recovery.training.max_steps=10
```

## Common Pitfalls to Avoid

1. ❌ **Forgetting CUDA_VISIBLE_DEVICES**
   - Results in training on all 8 GPUs instead of 2

2. ❌ **Missing `--config-name config`**
   - Hydra won't find the config files properly

3. ❌ **Missing `recovery=two_gpu`**
   - Will use default recovery config instead of 2-GPU optimized settings

4. ❌ **Trying to install DeepSpeed**
   - Not needed! FSDP is already configured and working

5. ❌ **Using max_steps: -1 with streaming datasets**
   - Results in ValueError about __len__
   - Must use explicit max_steps value

## Files Modified

1. [scripts/run_recovery_training.py](scripts/run_recovery_training.py#L295) - Fixed `eval_strategy` parameter + token-based calculation
2. [src/recovery_trainer.py](src/recovery_trainer.py#L166) - Fixed `compute_loss()` signature for newer transformers
3. [accelerate_config_2xa100.yaml](accelerate_config_2xa100.yaml) - Added CUDA_VISIBLE_DEVICES notes
4. [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) - All examples updated + checkpoint management + token-based training
5. [ACCELERATE_SETUP.md](ACCELERATE_SETUP.md) - All examples updated + DeepSpeed clarification
6. [conf/recovery/two_gpu.yaml](conf/recovery/two_gpu.yaml) - Usage examples, save_total_limit: null, max_tokens
7. [conf/recovery/default.yaml](conf/recovery/default.yaml) - save_total_limit: null, max_tokens
8. [TOKEN_BASED_TRAINING.md](TOKEN_BASED_TRAINING.md) - New comprehensive guide for token-based training

## Next Steps

1. Run the test command above to verify your setup
2. Monitor `nvidia-smi` to confirm only 2 GPUs are active
3. Start full recovery training with your desired experiment name
4. Monitor progress in WandB

## Support

If you encounter issues:
- Check [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md#troubleshooting) troubleshooting section
- Verify GPU visibility: `CUDA_VISIBLE_DEVICES=0,1 nvidia-smi`
- Test Accelerate: `CUDA_VISIBLE_DEVICES=0,1 accelerate test --config_file accelerate_config_2xa100.yaml`
