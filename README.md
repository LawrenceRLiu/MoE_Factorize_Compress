# MoE Compression via Shared Core + Low-Rank Wrappers

Research implementation for compressing Mixture of Experts (MoE) language models using a novel **Shared Core + Low-Rank Wrapper** decomposition.

## Overview

This project implements a compression technique for MoE models that exploits expert redundancy by factorizing expert weights into:
- **Shared Core**: A single dense matrix shared by all experts in a layer
- **Low-Rank Wrappers**: Lightweight per-expert adapters using low-rank matrices

### Mathematical Formulation

For each expert weight matrix $W_e$, we approximate:

$$\hat{W}_e = (I + U_e^{out} V_e^{out^T}) \cdot C \cdot (I + U_e^{in} V_e^{in^T})$$

Where:
- $C$: Shared core matrix (learned, common to all experts)
- $U_e^{in}, V_e^{in}$: Input low-rank adapter (rank $r \ll d_{in}$)
- $U_e^{out}, V_e^{out}$: Output low-rank adapter (rank $r \ll d_{out}$)

### Methodology
1. **Zero-Shot Initialization**: Parallelized reconstruction of expert weights using Adam to minimize L2 norm between the original and approximated weights.
2. **Knowledge Distillation**: Fine-tuning the compressed model using teacher-student training with blended loss (CE + KL divergence) to recover performance. See [DISTILLATION.md](docs/DISTILLATION.md) for details.
3. **Asynchronous Evaluation**: Dedicated GPUs evaluate checkpoints during distillation to monitor performance without interrupting training.

## Project Structure

```
MoE_Compress/
├── src/                          # Source code
│   ├── models/                   # Modified HuggingFace model implementations
│   ├── shared_core.py           # Shared core + low-rank wrapper implementation
│   ├── compressed_moe.py        # Compressed MoE module
│   ├── zero_shot_init.py        # Zero-shot initialization
│   ├── distillation_trainer.py # Knowledge distillation trainer
│   ├── teacher_logits.py        # Teacher logits caching
│   └── distillation_utils.py   # Distillation utilities
├── conf/                        # Hydra configuration files
│   ├── config.yaml             # Main configuration
│   ├── compression/            # Compression configs
│   ├── distillation/           # Distillation configs
│   └── evaluation/             # Evaluation configs
├── scripts/                     # Executable scripts
│   ├── run_compression.py      # Run compression pipeline
│   ├── generate_teacher_logits.py  # Generate cached teacher logits
│   └── run_async_eval.py       # Asynchronous evaluation
├── examples/                    # Example usage scripts
│   └── distillation_example.py # Complete distillation pipeline
├── docs/                        # Documentation
│   └── DISTILLATION.md         # Detailed distillation guide
└── models/                      # Model checkpoints and outputs
```
## Installation

The project uses a conda environment named `MoE_Compress` with the following packages:
- transformers
- torch
- accelerate
- deepspeed
- lm_eval
- datasets
- hydra-core
- wandb
#TODO: provide environment.yml


## Quick Start

### 1. Compression
Compress a MoE model using zero-shot initialization:
```bash
python scripts/run_compression.py
```

### 2. Knowledge Distillation
Train the compressed model with knowledge distillation:

```bash
# Option A: Generate teacher logits first (recommended for large datasets)
python scripts/generate_teacher_logits.py

# For large pretraining datasets (FineWeb, SlimPajama), use streaming:
python scripts/generate_teacher_logits.py \
    distillation.dataset.name=HuggingFaceFW/fineweb \
    distillation.dataset.streaming=true

# For multi-GPU acceleration (4 GPUs):
./scripts/generate_teacher_logits_parallel.sh 4

# Then run distillation
python scripts/run_distillation.py

# Option B: All-in-one (for smaller datasets)
python scripts/run_distillation.py
```

For distributed training with FSDP (multiple GPUs):
```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 scripts/run_distillation.py
```

**New Features:**
- **Streaming Dataset Support**: Process datasets larger than memory (FineWeb, SlimPajama, etc.)
- **Multi-GPU Teacher Logit Generation**: Parallelize across multiple GPUs for faster generation
- **Unified Configuration**: All teacher logit settings now in `conf/distillation/default.yaml`

See [docs/DISTILLATION.md](docs/DISTILLATION.md) for detailed distillation documentation.
See [docs/TEACHER_LOGITS_ADVANCED.md](docs/TEACHER_LOGITS_ADVANCED.md) for streaming and multi-GPU guides.

### 3. Evaluation
Asynchronously evaluate checkpoints during training:
```bash
python scripts/run_async_eval.py
```

## Key Features

### Knowledge Distillation
- **Offline Teacher Logits**: Cache teacher predictions to reduce GPU memory by ~50%
- **Blended Loss**: Combines cross-entropy and KL divergence with configurable weights
- **Parameter-Specific Learning Rates**: Fine-grained control via regex pattern matching
- **Multi-Stage Training**: Support for freezing/unfreezing different parameter groups
- **Memory Efficient**: Uses top-k logits storage with HDF5 compression

See [docs/DISTILLATION.md](docs/DISTILLATION.md) for comprehensive documentation.

## Configuration

All experiments are configured via Hydra in `conf/`:
- `config.yaml`: Main configuration
- `compression/default.yaml`: Compression settings (rank, optimization steps)
- `distillation/default.yaml`: Distillation settings (alpha, temperature, learning rates)
- `evaluation/default.yaml`: Evaluation settings

Example distillation config:
```yaml
loss:
  alpha: 0.5              # Weight for distillation loss
  temperature: 2.0        # Temperature for KL divergence

parameter_lr_multipliers:
  - pattern: ".*shared_core.*"
    lr_multiplier: 1.0    # Train compressed experts
  - pattern: ".*embed_tokens.*"
    lr_multiplier: 0.1    # Conservative on embeddings
```

## Known Issues/Future Work
- Currently there is a bug where each script in `./scripts` independently uses hydra to access the config. Thus the config could be modified between each script is run and cause unexpected behavior. Ideally, we should create a single script which access and saves the config into a file which each script can then read from to ensure consistency.

