# Quick Start Guide

## Prerequisites

1. **Conda environment**: `MoE_Compress` with base packages installed
2. **GPUs**: 8 GPUs (2× A100 80GB + 6× A6000 48GB recommended)
3. **Storage**: ~200GB for model weights and checkpoints

## Installation

```bash
# Activate environment
conda activate MoE_Compress

# Install additional dependencies
pip install -r requirements.txt

# Login to WandB (for experiment tracking)
wandb login
```

## Quick Test (Small Scale)

Test the pipeline on a smaller scale before running the full compression:

### 1. Test Compression (Single Layer)

Create a test config `conf/compression/test.yaml`:

```yaml
rank: 64
num_steps: 100  # Reduced for testing
lr: 1e-3
projections:
  - gate_proj  # Only one projection
```

Run compression:

```bash
python scripts/run_compression.py \
    compression=test \
    compression.num_layers=1  # Only compress first layer
```

### 2. Verify Compression

Check the output:

```bash
ls models/*/compressed/layer_0/
# Should see: gate_proj.pt, gate_proj_metadata.json
```

Inspect compression stats:

```bash
cat models/*/compressed/layer_0/gate_proj_metadata.json
```

## Full Pipeline

### Method 1: Automated (Recommended)

Run the complete pipeline using the example script:

```bash
./scripts/example_full_pipeline.sh
```

This will:
1. Compress the entire model (Phase 1)
2. Start async evaluation in background
3. Run knowledge distillation (Phase 2)
4. Continuously evaluate checkpoints

### Method 2: Manual Step-by-Step

#### Step 1: Zero-Shot Compression (~2-4 hours)

```bash
python scripts/run_compression.py
```

Monitor progress:
- Watch logs for per-layer compression statistics
- Check `models/*/compressed/` for saved weights

#### Step 2: Knowledge Distillation (~12-24 hours)

Terminal 1 - Start async evaluation:
```bash
python scripts/run_async_eval.py
```

Terminal 2 - Start distillation:
```bash
python scripts/run_distillation.py
```

Monitor progress:
- WandB dashboard for training metrics
- Evaluation results in `models/*/evaluation/`

## Configuration Tips

### Adjust Compression Ratio

Edit `conf/compression/qwen_3_30b.yaml`:

```yaml
# More compression (worse quality initially)
rank: 32

# Less compression (better quality)
rank: 128
```

### Speed Up for Testing

Edit `conf/distillation/default.yaml`:

```yaml
training:
  max_steps: 1000  # Limit training steps
  save_steps: 100  # Save more frequently
```

Edit `conf/evaluation/default.yaml`:

```yaml
test_mode:
  enabled: true
  limit: 100  # Only 100 samples per task
  tasks:
    - wikitext  # Only one task
```

### Reduce GPU Requirements

If you have fewer GPUs:

Edit `conf/config.yaml`:

```yaml
gpu_ids: [0, 1, 2, 3]  # Use only 4 GPUs

# In evaluation/default.yaml:
async_eval:
  gpu_ids: [3]  # Use last GPU for eval
```

## Monitoring Results

### WandB Dashboard

View training progress:
- Navigate to: https://wandb.ai
- Project: `moe-compression`
- Look for runs with your experiment name

Key metrics to watch:
- `kl_loss`: KL divergence between teacher and student
- `ce_loss`: Cross-entropy loss
- Task accuracies (from async eval)

### Local Results

```bash
# Compression statistics
cat models/*/compressed/compression_config.json

# Evaluation results
cat models/*/evaluation/evaluation_results.json

# Checkpoint evaluations
ls models/*/distilled/checkpoint-*
```

## Troubleshooting

### Out of Memory

**Problem**: CUDA out of memory during compression

**Solutions**:
1. Load model in lower precision:
   - Edit `src/zero_shot_init.py`: Change `torch.bfloat16` to `torch.float16`
2. Reduce batch size in optimization
3. Process fewer layers in parallel

### Slow Compression

**Problem**: Zero-shot initialization taking too long

**Solutions**:
1. Reduce `num_steps` in config
2. Use fewer GPUs but process more layers per GPU
3. Start with smaller `rank`

### Evaluation Failing

**Problem**: lm_eval tasks failing

**Solutions**:
1. Check lm_eval installation: `pip install lm-eval --upgrade`
2. Verify model can be loaded: Test with a single task
3. Reduce `batch_size` in eval config

### Teacher Model Too Large

**Problem**: Teacher model doesn't fit in VRAM during distillation

**Solutions**:
1. Enable 8-bit quantization (default):
   ```yaml
   teacher:
     load_in_8bit: true
   ```
2. Or use 4-bit:
   ```yaml
   teacher:
     load_in_4bit: true
   ```

## Next Steps

After successful compression and distillation:

1. **Export final model**: Convert to standard format for deployment
2. **Benchmark performance**: Run comprehensive evaluations
3. **Fine-tune**: Further fine-tune on domain-specific data
4. **Analyze**: Investigate which layers compress best

## Getting Help

- Check logs in `outputs/` (created by Hydra)
- Review WandB runs for error messages
- Examine evaluation results for performance issues

## Example Session

```bash
# 1. Test with small config
python scripts/run_compression.py compression=test compression.num_layers=1

# 2. If successful, run full compression
python scripts/run_compression.py

# 3. Evaluate baseline
python scripts/run_async_eval.py evaluation.evaluate_baseline=true

# 4. Run distillation (in new terminal)
python scripts/run_distillation.py

# 5. Monitor on WandB
# Open browser to https://wandb.ai
```

Estimated timeline:
- Test: ~10 minutes
- Full compression: 2-4 hours
- Distillation: 12-24 hours
- Total: ~1-2 days
