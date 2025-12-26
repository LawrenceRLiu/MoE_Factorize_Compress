# Knowledge Distillation Quick Reference

## TL;DR - Get Started in 3 Steps

```bash
# 1. Generate teacher logits (one-time, can skip if using small dataset)
python scripts/generate_teacher_logits.py \
    --teacher_model Qwen/Qwen3-30B-A3B-Base \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./teacher_logits

# 2. Configure (edit conf/distillation/default.yaml)
# Set your dataset, alpha, temperature, learning rates

# 3. Train!
python examples/distillation_example.py
```

## Key Files

| File | Purpose | When to Edit |
|------|---------|-------------|
| `conf/distillation/default.yaml` | Main config | Always - set your hyperparameters here |
| `src/distillation_trainer.py` | Trainer implementation | Rarely - only for custom loss functions |
| `src/teacher_logits.py` | Logits caching | Never - works out of box |
| `examples/distillation_example.py` | Training script | Sometimes - for custom pipelines |

## Critical Hyperparameters

### Loss Function (in `conf/distillation/default.yaml`)

```yaml
loss:
  alpha: 0.5              # ← START HERE: 0.5 is good default
  temperature: 2.0        # ← Try 2.0 first, then 3.0 if needed
  kl_reduction: "batchmean"  # Don't change this
```

**Tuning guide**:
- High CE loss, low KL loss → **Increase alpha** (e.g., 0.7)
- Low CE loss, high KL loss → **Decrease alpha** (e.g., 0.3)
- Student not learning from teacher → **Increase temperature** (e.g., 3.0)

### Learning Rates

```yaml
training:
  learning_rate: 5e-5     # ← Base LR, start here

parameter_lr_multipliers:
  # Train the compressed parts
  - pattern: ".*shared_core.*"
    lr_multiplier: 1.0    # ← Full LR for compressed experts

  # Be conservative with pre-trained parts
  - pattern: ".*embed_tokens.*"
    lr_multiplier: 0.1    # ← 10% LR for embeddings

  # Freeze what you don't want to change
  - pattern: ".*lm_head.*"
    lr_multiplier: 0.0    # ← 0 = frozen
```

**Common patterns**:
```yaml
".*shared_core.*"        # All shared core layers
".*wrapper.*"            # All wrapper layers (input/output)
".*expert.*"             # All expert layers
".*layer\.0\..*"         # First transformer layer
".*layer\.[0-9]\..*"    # Layers 0-9
".*embed.*"              # Embedding layers
".*norm.*"               # All normalization layers
".*"                     # Everything (use as last catch-all)
```

### Training Configuration

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2      # Adjust based on GPU memory
  gradient_accumulation_steps: 8      # Effective batch = 2 × 8 = 16
  learning_rate: 5e-5
  warmup_steps: 100
  bf16: true                          # Use bfloat16 for stability
```

**Memory tips**:
- OOM error? → Reduce `per_device_train_batch_size` to 1
- Want larger batch? → Increase `gradient_accumulation_steps`
- Still OOM? → Enable gradient checkpointing (see docs)

## Common Tasks

### Task 1: Basic Training (All Default Settings)

```bash
# Edit conf/distillation/default.yaml:
# - Set dataset.name and dataset.config_name
# - Keep all other defaults

python examples/distillation_example.py
```

### Task 2: Two-Stage Training (Freeze Then Unfreeze)

```yaml
# In conf/distillation/default.yaml:
stages:
  enabled: true
  configs:
    # Stage 1: Train only compressed experts
    - stage_name: "stage1_experts"
      num_epochs: 2
      parameter_lr_multipliers:
        - pattern: ".*shared_core.*|.*wrapper.*"
          lr_multiplier: 1.0
        - pattern: ".*"
          lr_multiplier: 0.0

    # Stage 2: Fine-tune everything
    - stage_name: "stage2_full"
      num_epochs: 1
      parameter_lr_multipliers:
        - pattern: ".*"
          lr_multiplier: 1.0
```

### Task 3: Custom Dataset

```yaml
# In conf/distillation/default.yaml:
dataset:
  name: "your-dataset-name"           # HuggingFace dataset
  config_name: "your-config"          # Dataset config
  split: "train"
  text_column: "text"                 # Column with text
  max_length: 2048
  max_samples: null                   # null = use all data
```

### Task 4: Only Train Specific Layers

```yaml
# Example: Only train layers 12-23 (middle layers)
parameter_lr_multipliers:
  - pattern: ".*layer\.(1[2-9]|2[0-3])\..*"    # Layers 12-23
    lr_multiplier: 1.0
  - pattern: ".*"                               # Everything else
    lr_multiplier: 0.0
```

### Task 5: Generate Teacher Logits Only

```bash
# Useful if you want to generate logits on a cluster
# and train on a different machine

python scripts/generate_teacher_logits.py \
    --teacher_model Qwen/Qwen3-30B-A3B-Base \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./teacher_logits \
    --top_k 64 \
    --batch_size 4 \
    --max_samples 10000              # Limit samples for testing

# Then copy ./teacher_logits/ to your training machine
# and set teacher.cache_dir in config
```

## Troubleshooting

### Problem: Training is too slow

**Solutions**:
1. Use cached teacher logits (set `teacher.use_cached_logits: true`)
2. Increase `dataloader_num_workers` (e.g., 8)
3. Reduce `logging_steps` to log less frequently
4. Use larger `gradient_accumulation_steps`

### Problem: Loss is NaN or very large

**Solutions**:
1. Reduce learning rate (try 1e-5)
2. Check your data (are there invalid tokens?)
3. Use `bf16: true` instead of `fp16`
4. Reduce temperature to 1.5 or 2.0
5. Check gradient clipping (`max_grad_norm: 1.0`)

### Problem: Student not learning from teacher (KL loss not decreasing)

**Solutions**:
1. Increase alpha (try 0.7 or 0.8)
2. Increase temperature (try 3.0 or 4.0)
3. Check that student model is properly initialized
4. Verify cached logits are loading correctly
5. Try training longer (more epochs)

### Problem: Student not learning task (CE loss not decreasing)

**Solutions**:
1. Decrease alpha (try 0.3 or 0.2)
2. Increase learning rate for task-relevant layers
3. Make sure labels are correct
4. Train for more epochs

### Problem: Out of memory

**Solutions**:
1. **Use cached teacher logits!** (Saves ~60GB for Qwen-30B)
2. Reduce batch size to 1
3. Increase gradient accumulation to maintain effective batch size
4. Enable gradient checkpointing:
   ```python
   student_model.gradient_checkpointing_enable()
   ```
5. Use smaller max_length (e.g., 1024 instead of 2048)
6. Reduce top_k in cached logits (e.g., 32 instead of 64)

## Monitoring Training

### What to Watch

```bash
# During training you'll see:
Step 100: Total Loss: 2.3456, CE Loss: 2.1234, KL Loss: 0.4567
```

**Healthy training**:
- Total loss should decrease steadily
- CE loss should decrease (student learning task)
- KL loss should decrease (student matching teacher)
- Ratio CE:KL should be roughly (1-α):α

**Unhealthy training**:
- Loss increasing → LR too high or data problem
- Loss flat → LR too low or converged
- CE decreasing but KL flat → Increase alpha or temperature
- KL decreasing but CE flat → Decrease alpha

### Using Weights & Biases

```yaml
# In conf/config.yaml:
wandb_project: "moe-compression"
experiment_name: "my-experiment"
```

Will automatically log:
- Loss components (total, CE, KL)
- Learning rates per parameter group
- Gradient norms
- GPU memory usage

## Recommended Starting Point

For your first run, use these settings:

```yaml
# conf/distillation/default.yaml
teacher:
  use_cached_logits: true
  top_k: 64

loss:
  alpha: 0.5
  temperature: 2.0

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  warmup_steps: 100

parameter_lr_multipliers:
  - pattern: ".*shared_core.*"
    lr_multiplier: 1.0
  - pattern: ".*wrapper.*"
    lr_multiplier: 1.0
  - pattern: ".*embed_tokens.*|.*lm_head.*"
    lr_multiplier: 0.1
  - pattern: ".*norm.*"
    lr_multiplier: 0.5
```

Then iterate based on results!

## Question 5 Answer: Yes, Offline Teacher is Better!

**Reasons**:
1. **Memory**: Saves ~60GB GPU memory (the entire teacher model)
2. **Speed**: 2x faster training (no teacher forward passes)
3. **Flexibility**: Can use all saved memory for larger student batch sizes
4. **Top-k is enough**: Only top 32-64 logits contain meaningful distillation signal
5. **Reusable**: Cache once, train many times with different hyperparameters

**Trade-off**: ~10-15GB disk space (totally worth it)

**Our implementation uses**:
- HDF5 with gzip compression
- Top-k values + indices storage
- Lazy loading during training
- Automatic reconstruction to full vocabulary

This is the recommended approach for all distillation tasks with large teachers!
