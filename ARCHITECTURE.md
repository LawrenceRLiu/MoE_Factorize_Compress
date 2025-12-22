## Compressed MoE Model Architecture

### Overview

This document explains the custom model architecture changes needed to support compressed MoE models, and how the save/load mechanisms work.

## Problem: Why Custom Architecture?

**Standard HuggingFace models cannot directly support our compression scheme** because:

1. **Structural Changes**: We replace individual expert weight matrices with:
   - One shared core matrix (per layer/projection)
   - Multiple low-rank wrapper matrices (per expert)

2. **Forward Pass Changes**: The computation graph changes from:
   ```python
   # Original
   output = expert.weight @ input
   ```
   to:
   ```python
   # Compressed
   x = (I + U_in @ V_in^T) @ input          # Input wrapper
   x = core @ x                              # Shared core
   output = (I + U_out @ V_out^T) @ x       # Output wrapper
   ```

3. **Save/Load Incompatibility**: Standard `model.save_pretrained()` and `AutoModel.from_pretrained()` expect a specific state_dict structure that our compressed model doesn't match.

## Solution: Custom Architecture Components

### 1. Compression Statistics Module

**File**: [src/compression_stats.py](src/compression_stats.py)

**Purpose**: Track both total and active parameter counts correctly.

**Key Insight**: MoE models have two types of parameter counts:
- **Total parameters**: All expert weights (what we compress)
- **Active parameters**: Only top-k experts used per token (what affects inference speed)

**Example Output**:
```yaml
total_params_original: 470000000      # All 8 experts
total_params_compressed: 73000000     # Core + all wrappers
total_compression_ratio: 0.155        # 15.5% of original (84.5% reduction)

active_params_original: 117500000     # 2 experts (top-2 routing)
active_params_compressed: 132000000   # Core + 2 wrappers
active_compression_ratio: 1.123       # 12.3% increase in active params
```

**Usage in compression**:
```python
from src.compression_stats import CompressionStats

stats = CompressionStats()
stats.add_layer_stats(layer_idx=0, projection="gate_proj", ...)
stats.save("models/compressed/compression_statistics.yaml")
```

### 2. Compressed Model Architecture

**File**: [src/compressed_moe_model.py](src/compressed_moe_model.py)

#### Key Classes:

**`CompressedMoEExpert`**: Wrapper around a single compressed expert
- Maintains the same interface as original experts
- Uses SharedCore + LowRankWrappers internally

**`CompressedMoEBlock`**: Replacement for entire MoE layer
- Keeps original routing gate (unchanged)
- Replaces expert execution with compressed versions
- Handles top-k expert selection and weight combination

**`load_compressed_model()`**: Custom model loader
- Loads original model architecture
- Replaces MoE blocks with compressed versions
- Loads compressed weights from disk

**`save_compressed_model()`**: Custom model saver
- Extracts compressed layers
- Saves in our custom format
- Includes metadata for reconstruction

### 3. Custom Save/Load Format

Our compressed models are saved in this structure:

```
models/compressed/
├── compression_config.json           # Compression settings (rank, projections, etc.)
├── compression_statistics.yaml       # Detailed statistics
├── config.json                       # Original HuggingFace model config
├── layer_0/
│   ├── gate_proj.pt                 # Compressed weights
│   ├── gate_proj_metadata.json      # Dimensions, num_experts, etc.
│   ├── up_proj.pt
│   ├── up_proj_metadata.json
│   ├── down_proj.pt
│   └── down_proj_metadata.json
├── layer_1/
│   └── ...
└── ...
```

**Each `.pt` file contains**:
```python
{
    "layer_idx": 0,
    "projection": "gate_proj",
    "core": torch.Tensor,              # Shared core matrix [d_out, d_in]
    "wrappers": [                      # One per expert
        {
            "U_in": torch.Tensor,      # [d_in, rank]
            "V_in": torch.Tensor,      # [d_in, rank]
            "U_out": torch.Tensor,     # [d_out, rank]
            "V_out": torch.Tensor,     # [d_out, rank]
        },
        ...
    ],
    "metadata": {
        "num_experts": 8,
        "d_in": 4096,
        "d_out": 14336,
        "rank": 64
    }
}
```

## Integration with Training Pipeline

### Zero-Shot Compression ([src/zero_shot_init.py](src/zero_shot_init.py))

```python
# After compression completes, statistics are saved automatically
parallel_compression(model_name, config, gpu_ids)
# Creates:
# - models/compressed/layer_*/
# - models/compressed/compression_statistics.yaml
```

### Knowledge Distillation ([src/distillation.py](src/distillation.py))

**Loading Student**:
```python
# Automatically detects compressed model
student_path = Path(config.student_model_path)
if (student_path / "compression_config.json").exists():
    student_model = load_compressed_model(
        compressed_dir=config.student_model_path,
        original_model_name=config.teacher_model,
        ...
    )
```

**Saving Checkpoints**:
```python
# Custom callback handles saving
class CompressedModelSaveCallback(TrainerCallback):
    def on_save(self, ...):
        save_compressed_model(model, output_dir, tokenizer)
```

### Async Evaluation ([src/async_eval.py](src/async_eval.py))

**Current Limitation**: lm_eval CLI doesn't automatically know about our custom format.

**Workaround Options**:

1. **Export to standard format** (recommended for evaluation):
   ```python
   # After distillation, export to standard HF format
   # by materializing the compressed experts as full weight matrices
   export_to_hf_format(compressed_model, output_path)
   ```

2. **Use Python API instead of CLI**:
   ```python
   from lm_eval import evaluator
   model = load_compressed_model(checkpoint_path, ...)
   results = evaluator.simple_evaluate(model, tasks=tasks)
   ```

3. **Register custom model type** with lm_eval (advanced)

## Usage Examples

### Loading a Compressed Model

```python
from src.compressed_moe_model import load_compressed_model

# Load compressed model
model = load_compressed_model(
    compressed_dir="models/Qwen-3-30B/compressed",
    original_model_name="Qwen/Qwen-3-30B-A3B",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Use normally
outputs = model.generate(inputs, max_length=100)
```

### Checking Compression Statistics

```python
import yaml

# Load statistics
with open("models/compressed/compression_statistics.yaml", 'r') as f:
    stats = yaml.safe_load(f)

print(f"Total compression: {stats['total_compression_ratio']:.3f}")
print(f"Total reduction: {stats['total_reduction_percent']:.1f}%")
print(f"Active params change: {stats['active_change_percent']:+.1f}%")
```

### Saving After Fine-Tuning

```python
from src.compressed_moe_model import save_compressed_model

# After fine-tuning
save_compressed_model(
    model=finetuned_model,
    output_dir="models/Qwen-3-30B/finetuned",
    tokenizer=tokenizer
)
```

## Parameter Count Breakdown

### Original MoE Model
```
Assumptions: 8 experts, d_model=4096, d_ffn=14336, top-2 routing

Total params (all experts):
  = 8 experts × 3 projections × (4096 × 14336)
  = 8 × 3 × 58.7M
  = 1.41B params per layer

Active params (per token with top-2):
  = 2 experts × 3 projections × (4096 × 14336)
  = 2 × 3 × 58.7M
  = 352M params per layer
```

### Compressed MoE Model (rank=64)
```
Total params:
  Core: 3 projections × (4096 × 14336) = 176M
  Wrappers: 8 experts × 3 proj × (2×4096×64 + 2×14336×64) = 93M
  Total: 269M params per layer
  Ratio: 269M / 1.41B = 0.19 (81% reduction) ✓

Active params (per token with top-2):
  Core: 176M (used for all tokens)
  Active wrappers: 2 experts × 3 proj × (2×4096×64 + 2×14336×64) = 23M
  Total: 199M params per layer
  Ratio: 199M / 352M = 0.57 (43% reduction for active params!)
```

**Wait, this is actually better than expected!** The active parameters are also reduced because:
- The core is shared across all experts
- Each token only needs 2 expert wrappers (not full expert weights)

## Architecture-Specific Notes

### Qwen-MoE Structure
```python
model.model.layers[i].mlp.experts[j].gate_proj
model.model.layers[i].mlp.experts[j].up_proj
model.model.layers[i].mlp.experts[j].down_proj
model.model.layers[i].mlp.gate  # Routing gate
```

### Mixtral Structure
```python
model.model.layers[i].block_sparse_moe.experts[j].w1
model.model.layers[i].block_sparse_moe.experts[j].w2
model.model.layers[i].block_sparse_moe.experts[j].w3
model.model.layers[i].block_sparse_moe.gate  # Routing gate
```

## Troubleshooting

### Issue: "Cannot load compressed model"

**Cause**: Missing `compression_config.json`

**Solution**: Ensure zero-shot compression completed successfully

### Issue: "Shape mismatch when loading weights"

**Cause**: Trying to load compressed weights into standard model

**Solution**: Use `load_compressed_model()` instead of `AutoModel.from_pretrained()`

### Issue: "lm_eval cannot find model"

**Cause**: lm_eval CLI doesn't support custom format

**Solution**: Use lm_eval Python API or export to standard format

## Future Enhancements

1. **Export to standard format**: Materialize compressed experts as full weights for compatibility
2. **Lazy loading**: Only load compressed layers when needed
3. **8-bit/4-bit quantization**: Combine compression with quantization
4. **LoRA fine-tuning**: Train only the wrappers, freeze the core

## References

- Shared Core implementation: [src/shared_core.py](src/shared_core.py)
- Zero-shot compression: [src/zero_shot_init.py](src/zero_shot_init.py)
- Model architecture: [src/compressed_moe_model.py](src/compressed_moe_model.py)
- Statistics tracking: [src/compression_stats.py](src/compression_stats.py)
