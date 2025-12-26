# Knowledge Distillation for Compressed MoE Models

This document describes the knowledge distillation system for training compressed Mixture-of-Experts (MoE) models.

## Overview

The distillation system implements knowledge transfer from a large teacher model (original MoE) to a smaller student model (compressed MoE using shared-core architecture). It features:

1. **Offline Teacher Logits Caching**: Pre-compute and store teacher logits to reduce memory requirements
2. **Blended Loss Function**: Combines cross-entropy and KL divergence losses
3. **Parameter-Specific Learning Rates**: Fine-grained control via regex pattern matching
4. **Multi-Stage Training**: Support for freezing/unfreezing different parameter groups

## Architecture

### Components

```
src/
├── distillation_trainer.py      # Custom HuggingFace Trainer with distillation loss
├── teacher_logits.py             # Teacher logits generation and caching
├── distillation_utils.py         # Data preparation and training utilities
└── ...

conf/distillation/default.yaml   # Distillation configuration

scripts/
└── generate_teacher_logits.py   # Standalone script for logit generation

examples/
└── distillation_example.py      # Complete distillation pipeline example
```

### Key Features

#### 1. Offline Teacher Logits Caching

**Why offline?**
- Reduces GPU memory by ~50% (no need to keep teacher in memory during training)
- Faster training (teacher inference done once)
- Enables larger batch sizes for student training
- Teacher logits are deterministic and can be reused

**How it works:**
- Teacher model generates logits for entire dataset
- Only top-k (32-64) logits + indices are stored per position
- Uses HDF5 with compression for efficient storage
- Cached logits loaded on-the-fly during training

**Storage requirements:**
```
For a dataset with:
- 100k samples
- 2048 sequence length
- top_k = 64
- float16 precision

Storage ≈ 100k × 2048 × 64 × 2 bytes ≈ 26 GB (compressed to ~10-15 GB)
```

#### 2. Blended Loss Function

The total loss combines cross-entropy (CE) and KL divergence:

```
L_total = (1 - α) × L_CE + α × T² × L_KL

where:
- α: distillation weight (0 to 1)
- T: temperature for softening distributions
- L_CE: standard language modeling loss
- L_KL: KL divergence between teacher and student distributions
```

**Temperature scaling:**
- Higher T → softer distributions → emphasizes teacher's uncertainty
- T² factor compensates for gradient magnitude change
- Typical values: T ∈ [2, 4]

**Alpha tuning:**
- α = 0: pure supervised learning (no distillation)
- α = 1: pure distillation (no ground truth labels)
- α = 0.5: balanced (recommended starting point)

#### 3. Parameter-Specific Learning Rates

Control learning rates for different parts of the model using regex patterns:

```yaml
parameter_lr_multipliers:
  - pattern: ".*shared_core.*"       # Compressed experts
    lr_multiplier: 1.0               # Full learning rate

  - pattern: ".*embed_tokens.*"      # Embeddings
    lr_multiplier: 0.1               # 10% of base LR

  - pattern: ".*norm.*"              # Layer norms
    lr_multiplier: 0.0               # Frozen
```

**Use cases:**
- Freeze embeddings and LM head (transfer from teacher)
- Focus training on compressed expert weights
- Gradual unfreezing strategies

#### 4. Multi-Stage Training

Train with different configurations in sequence:

**Example: Two-stage training**

Stage 1: Train only compressed experts (2 epochs)
```yaml
- stage_name: "stage1_experts_only"
  num_epochs: 2
  parameter_lr_multipliers:
    - pattern: ".*shared_core.*"
      lr_multiplier: 1.0
    - pattern: ".*experts.*"
      lr_multiplier: 1.0
    - pattern: ".*"
      lr_multiplier: 0.0  # Everything else frozen
```

Stage 2: Fine-tune entire model (1 epoch)
```yaml
- stage_name: "stage2_full_model"
  num_epochs: 1
  parameter_lr_multipliers:
    - pattern: ".*"
      lr_multiplier: 1.0  # All parameters trainable
```

## Usage

### Quick Start

```bash
# 1. Generate teacher logits (one-time)
python scripts/generate_teacher_logits.py \
    --teacher_model Qwen/Qwen3-30B-A3B-Base \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir ./teacher_logits \
    --top_k 64 \
    --batch_size 4

# 2. Run distillation training
python examples/distillation_example.py
```

### Configuration

Edit `conf/distillation/default.yaml`:

```yaml
# Teacher configuration
teacher:
  model_path: null  # Uses model.name from main config
  use_cached_logits: true
  cache_dir: "${output.temp_dir}/teacher_logits"
  top_k: 64
  batch_size: 4

# Loss configuration
loss:
  alpha: 0.5        # Distillation weight
  temperature: 2.0  # Temperature for KL divergence

# Training configuration
training:
  num_train_epochs: 3
  learning_rate: 5e-5
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  # ... (see default.yaml for full options)

# Parameter-specific learning rates
parameter_lr_multipliers:
  - pattern: ".*shared_core.*"
    lr_multiplier: 1.0
  - pattern: ".*embed_tokens.*"
    lr_multiplier: 0.1
  # ... (add more patterns as needed)
```

### Python API

```python
from omegaconf import OmegaConf
from src.distillation_utils import setup_distillation_pipeline, run_distillation_training

# Load configuration
config = OmegaConf.load("conf/config.yaml")

# Setup pipeline (handles teacher logits, dataset prep, model loading)
student_model, train_dataset, eval_dataset, tokenizer = setup_distillation_pipeline(
    config=config,
    teacher_model_path="Qwen/Qwen3-30B-A3B-Base",
    student_model_path="./models/checkpoint-0",  # Your compressed model
    force_regenerate_logits=False
)

# Run training
trainer = run_distillation_training(
    config=config,
    student_model=student_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Save final model
trainer.save_model("./final_model")
```

### Advanced: Custom Training Loop

```python
from src.distillation_trainer import DistillationTrainer, DistillationTrainingArguments

# Create custom training arguments
training_args = DistillationTrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    learning_rate=5e-5,
    distillation_alpha=0.5,
    distillation_temperature=2.0,
    # ... other args
)

# Define parameter groups
lr_multipliers = [
    {"pattern": ".*shared_core.*", "lr_multiplier": 1.0},
    {"pattern": ".*embed_tokens.*", "lr_multiplier": 0.1},
]

# Create trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    lr_multipliers=lr_multipliers
)

# Train
trainer.train()
```

## Recommended Hyperparameters

### For Compressed MoE Training

```yaml
loss:
  alpha: 0.5 - 0.7        # Higher α emphasizes teacher knowledge
  temperature: 2.0 - 3.0  # Moderate softening

training:
  learning_rate: 1e-5 - 5e-5  # Lower than pre-training
  warmup_steps: 100 - 500
  gradient_accumulation_steps: 8 - 32  # To maintain effective batch size

parameter_lr_multipliers:
  # Focus on compressed components
  - pattern: ".*shared_core.*"
    lr_multiplier: 1.0

  - pattern: ".*input_wrapper.*|.*output_wrapper.*"
    lr_multiplier: 1.0

  # Conservative on non-compressed parts
  - pattern: ".*embed_tokens.*|.*lm_head.*"
    lr_multiplier: 0.05 - 0.1
```

### Two-Stage Training Strategy

**Stage 1: Compressed experts only (warm-up)**
- 1-2 epochs
- Freeze everything except `shared_core` and wrapper layers
- Higher learning rate (5e-5)
- Focus on expert reconstruction quality

**Stage 2: Full model fine-tuning**
- 1-2 epochs
- Unfreeze all parameters
- Lower learning rate (1e-5)
- Refine entire model

## Memory Optimization

### GPU Memory Breakdown

**Without offline logits (teacher + student in memory):**
```
Teacher model:     ~60 GB (Qwen-30B in fp16)
Student model:     ~20 GB (compressed)
Activations:       ~10 GB
Total:             ~90 GB (requires 2x A100 80GB)
```

**With offline logits (student only):**
```
Student model:     ~20 GB
Cached logits:     ~2 GB (loaded in batches)
Activations:       ~10 GB
Total:             ~32 GB (fits on 1x A100 40GB)
```

### Tips for Large Models

1. **Use gradient checkpointing:**
```python
student_model.gradient_checkpointing_enable()
```

2. **Increase gradient accumulation:**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

3. **Use bf16 instead of fp32:**
```yaml
bf16: true
fp16: false
```

4. **Reduce top-k for cached logits:**
```yaml
teacher:
  top_k: 32  # Instead of 64
```

## Monitoring Training

### Key Metrics to Watch

1. **Loss components:**
   - `total_loss`: Combined loss
   - `ce_loss`: Cross-entropy (student accuracy on labels)
   - `kl_loss`: KL divergence (student-teacher agreement)

2. **Loss balance:**
   - If `ce_loss >> kl_loss`: Increase α
   - If `kl_loss >> ce_loss`: Decrease α

3. **Learning dynamics:**
   - `kl_loss` should decrease steadily (student learning from teacher)
   - `ce_loss` should decrease (student improving on task)

### Logging

```python
# During training, you'll see:
Step 100: Total Loss: 2.3456, CE Loss: 2.1234, KL Loss: 0.4567
Step 200: Total Loss: 2.2345, CE Loss: 2.0123, KL Loss: 0.4234
...
```

## Troubleshooting

### Issue: KL loss is very large

**Cause:** Student and teacher distributions are very different
**Solutions:**
- Increase temperature T (softer distributions)
- Decrease α (less weight on KL)
- Check student model initialization

### Issue: Training is very slow

**Cause:** Large batch size or inefficient data loading
**Solutions:**
- Reduce batch size
- Increase `dataloader_num_workers`
- Use cached logits (if not already)
- Enable gradient checkpointing

### Issue: OOM (Out of Memory)

**Cause:** Model + activations don't fit in GPU memory
**Solutions:**
- Use offline teacher logits caching
- Reduce batch size
- Increase gradient accumulation
- Enable gradient checkpointing
- Use model parallelism

### Issue: Student not improving

**Cause:** Various possible issues
**Solutions:**
- Verify student model is properly initialized (use zero-shot init)
- Check learning rate (may be too high/low)
- Verify cached logits are loading correctly
- Try increasing α (more teacher signal)
- Check if important parameters are frozen

## References

1. **Knowledge Distillation:** Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
2. **Temperature Scaling:** Müller et al., "When Does Label Smoothing Help?" (2019)
3. **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

## Citation

If you use this distillation system in your research, please cite:

```bibtex
@software{moe_compress_distillation,
  title = {Knowledge Distillation for Compressed MoE Models},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/MoE_Compress}
}
```
