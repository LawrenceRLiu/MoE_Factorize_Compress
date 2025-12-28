# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT NOTES, PLEASE READ:**
This project uses the `MoE_Compress` conda enviroment. You must activate it before running any python code or commands.

```bash
conda activate MoE_Compress
```

## Project Overview

This is a research project implementing **Shared-Core Compression for MoE (Mixture of Experts) LLMs**. The goal is to reduce parameter count by exploiting expert redundancy through factorization into shared cores and low-rank wrappers.

**Target Model:** Qwen-3-30B-A3B with 20-30% compression ratio

**Mathematical Formulation:**
```
W_e ≈ (I + U_out V_out^T) · C · (I + U_in V_in^T)
```
where C is the shared core (common to all experts) and U/V are low-rank adapters per expert.

**Two-Phase Pipeline:**
1. **Phase 1 - Zero-Shot Initialization**: Parallel compression using Adam optimization to minimize L2 reconstruction error
2. **Phase 2 - Recovery Training**: Knowledge distillation pretraining to restore performance

## Hardware Configuration

**Available GPUs:** 2x A100 80GB (cuda:0, cuda:1) + 6x A6000 48GB (cuda:2-7)

**Critical:** Always use `CUDA_VISIBLE_DEVICES=0,1` to restrict training to the 2x A100s for recovery training.

## Common Commands

### Zero-Shot Compression (Phase 1)
```bash
# Run parallel compression across GPUs
python scripts/run_compression.py experiment_name=<experiment_name>
```

### Recovery Training (Phase 2)

**Recommended approach using Accelerate:**
```bash
# For 2x A100 GPUs (RECOMMENDED)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=<experiment_name> \
    recovery.training.max_tokens=2_000_000_000

# With CPU offload if OOM
export ACCELERATE_CONFIG_FILE=accelerate_config_2xa100_cpu_offload.yaml
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=<experiment_name>
```

**Alternative using torchrun:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/run_recovery_training.py \
    recovery=two_gpu \
    experiment_name=<experiment_name>
```

### Asynchronous Evaluation
```bash
# Monitor and evaluate checkpoints as they're saved
python scripts/run_async_eval.py experiment_name=<experiment_name>
```

### Testing
```bash
# Quick test with limited data
python scripts/run_recovery_training.py \
    recovery.dataset.num_samples=1000 \
    recovery.training.max_steps=100 \
    recovery.training.save_steps=50
```

## Architecture Overview

### Directory Structure
- `src/` - Core source code
  - `compressed_moe.py` - SharedCoreExperts implementation
  - `shared_core.py` - SharedCoreLayer with low-rank wrappers
  - `zero_shot_init.py` - Phase 1 parallel compression
  - `recovery_trainer.py` - Phase 2 HuggingFace Trainer subclass
  - `async_eval.py` - Asynchronous checkpoint evaluation
  - `model_utils.py` - Model loading utilities
  - `models/modeling_qwen3_moe.py` - Modified Qwen MoE architecture
- `scripts/` - Executable scripts
  - `run_compression.py` - Phase 1 entry point
  - `run_recovery_training.py` - Phase 2 entry point
  - `run_async_eval.py` - Evaluation entry point
- `conf/` - Hydra configuration files
  - `config.yaml` - Main configuration
  - `compression/default.yaml` - Compression settings (rank, num_steps, lr)
  - `recovery/default.yaml` - Recovery training settings
  - `recovery/two_gpu.yaml` - Optimized for 2x A100s
  - `evaluation/default.yaml` - Evaluation tasks
- `models/{model_name}/{experiment_name}/` - Output directory structure
  - `checkpoints/checkpoint-{step}/` - Model checkpoints (checkpoint-0 is zero-shot init)
  - `evals/` - Evaluation results
  - `tmp/` - Temporary files

### Key Architectural Decisions

1. **SharedCoreLayer**: Implements `(I + U·V^T) · Core · (I + U·V^T)` factorization
   - Core is shared across all experts in a layer
   - Low-rank adapters (U, V) are per-expert
   - Initialized LoRA-style: U=0, V~N(0,1)

2. **FSDP for Distributed Training**: Uses FSDP (not DeepSpeed) for multi-GPU training
   - Full sharding (ZeRO-3) for maximum memory efficiency
   - Configured via both Accelerate config and TrainingArguments
   - **Critical:** Must set `device_map=null` when using FSDP

3. **Checkpoint Format**: All checkpoints saved as `checkpoint-{step}` for async evaluation
   - `checkpoint-0` is the zero-shot initialization
   - No `save_total_limit` - all checkpoints are kept for async eval
   - Each includes: `pytorch_model.bin`, `config.json`, tokenizer files, `training_metadata.json`

4. **Token-Based Training Budget**: Use `max_tokens` parameter instead of `max_steps`
   - Example: `recovery.training.max_tokens=2_000_000_000` for 2B tokens
   - Script auto-calculates `max_steps` based on batch size, sequence length, and num GPUs
   - Makes it easier to compare training budgets across different configurations

## Configuration Management

**Uses Hydra** for configuration management. Override via command line:

```bash
# Override specific settings
python scripts/run_recovery_training.py \
    recovery.training.learning_rate=1e-5 \
    recovery.training.save_steps=1000 \
    recovery.dataset.max_length=1024
```

**Key Configuration Files:**
- `conf/recovery/two_gpu.yaml` - Pre-configured for 2x A100s
- `accelerate_config_2xa100.yaml` - Accelerate FSDP config (no CPU offload)
- `accelerate_config_2xa100_cpu_offload.yaml` - With CPU offload for OOM cases

## Important Implementation Details

### FSDP Requirements
1. **Always set `device_map=null`** when using FSDP (conflicts otherwise)
2. **Configure FSDP in TrainingArguments** even when using Accelerate launch
   - Without this, Trainer may fall back to DDP and cause OOM
   - See `setup_fsdp_args()` in [scripts/run_recovery_training.py](scripts/run_recovery_training.py)
3. **Use `fsdp_use_orig_params=true`** for gradient checkpointing compatibility

### Memory Optimization

**RECOMMENDED SOLUTION**: The 2-GPU config uses **torchao 8-bit AdamW** (`adamw_torch_8bit`), which provides ~75% optimizer memory reduction while maintaining full FSDP compatibility and 2048 sequence length.

**Setup** (already configured in `conf/recovery/two_gpu.yaml`):
```bash
# Install torchao (required)
pip install torchao
```

Configuration:
- Sequence length: 2048 (full length!)
- Optimizer: `adamw_torch_8bit` (PyTorch's official 8-bit optimizer)
- Gradient accumulation: 32
- Expected peak memory: ~77GB/GPU (safely fits on 80GB A100s)

**Important**: `bitsandbytes` optimizers (adamw_bnb_8bit, paged_adamw_*) are incompatible with FSDP. Use `adamw_torch_8bit` from `torchao` instead.

If still encountering OOM errors:
1. Verify FSDP is sharding: both GPUs should show similar memory usage in `nvidia-smi`
2. Verify torchao is installed: `python -c "import torchao; print(torchao.__version__)"`
3. **Alternative**: Reduce sequence length to 1536 with `adamw_torch` (no external dependency)
4. **Last resort**: Use CPU offload config `accelerate_config_2xa100_cpu_offload.yaml` (2-3x slower)

See [FSDP_OPTIMIZER_COMPATIBILITY.md](FSDP_OPTIMIZER_COMPATIBILITY.md) for complete optimizer compatibility guide.

### Distributed Launch Methods
**Accelerate (Recommended):**
- Better HuggingFace integration
- Version-controlled config files
- Easier scaling
- Better FSDP error messages

**Torchrun (Alternative):**
- Direct PyTorch control
- Simpler for debugging
- No external config needed

**Critical:** Both methods require FSDP configuration in TrainingArguments to prevent DDP fallback.

## Workflow Example

Complete end-to-end workflow:
```bash
# Step 1: Zero-shot compression (Phase 1)
python scripts/run_compression.py experiment_name=my_experiment

# Step 2: Start async evaluation (in background)
python scripts/run_async_eval.py experiment_name=my_experiment &

# Step 3: Recovery training (Phase 2)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_2xa100.yaml \
    scripts/run_recovery_training.py \
    --config-name config \
    recovery=two_gpu \
    experiment_name=my_experiment \
    recovery.training.max_tokens=2_000_000_000

# Monitor on WandB: https://wandb.ai/your-username/moe-compression
```

## WandB Integration

All experiments automatically log to WandB project `moe-compression`:
- Recovery training logs: loss, learning_rate, epoch, step
- Async evaluation logs: task results (wikitext, mmlu, gsm8k, etc.) per checkpoint

## Dependencies

Python 3.14 environment with:
- Core: `transformers`, `torch`, `accelerate`, `datasets`, `hydra-core`, `wandb`
- Evaluation: `lm-eval`
- Optimization: `scipy`, `torchao` (for 8-bit AdamW)
- Optional: `flash-attn` (for faster attention if CUDA compatible)

Install additional packages as needed: `pip install <package>`

## Documentation References

For detailed information, see:
- [ResearchSpec.md](ResearchSpec.md) - Mathematical formulation and algorithms
- [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) - Complete Phase 2 guide
- [TWO_GPU_SETUP.md](TWO_GPU_SETUP.md) - FSDP vs other parallelism strategies
- [FSDP_OPTIMIZER_COMPATIBILITY.md](FSDP_OPTIMIZER_COMPATIBILITY.md) - **IMPORTANT**: Why bitsandbytes optimizers don't work with FSDP
- [OPTIMIZER_MEMORY_GUIDE.md](OPTIMIZER_MEMORY_GUIDE.md) - General optimizer memory optimization strategies
- [FSDP_LOADING_EXPLAINED.md](FSDP_LOADING_EXPLAINED.md) - Why FSDP model loading takes 4 minutes (and why it's normal)
- [OOM_TROUBLESHOOTING.md](OOM_TROUBLESHOOTING.md) - General memory optimization guide
- [TOKEN_BASED_TRAINING.md](TOKEN_BASED_TRAINING.md) - Using max_tokens parameter
- [ACCELERATE_SETUP.md](ACCELERATE_SETUP.md) - Accelerate configuration details
