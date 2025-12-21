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

## Project Structure

```
MoE_Compress/
├── src/                          # Source code
│   ├── shared_core.py           # Core compression module
│   ├── zero_shot_init.py        # Zero-shot initialization
│   ├── distillation.py          # Knowledge distillation
│   ├── async_eval.py            # Asynchronous evaluation
│   └── utils.py                 # Utility functions
├── scripts/                      # Executable scripts
│   ├── run_compression.py       # Run zero-shot compression
│   ├── run_distillation.py      # Run knowledge distillation
│   └── run_async_eval.py        # Run async evaluation
├── conf/                         # Hydra configuration files
│   ├── config.yaml              # Main config
│   ├── compression/             # Compression configs
│   ├── distillation/            # Distillation configs
│   └── evaluation/              # Evaluation configs
└── models/                       # Model artifacts (created during runtime)
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

Additional required packages:
```bash
conda activate MoE_Compress
pip install bitsandbytes  # For quantization
pip install sentencepiece  # For tokenization
```

## Usage

### Phase 1: Zero-Shot Initialization

Compress the model using parallel zero-shot reconstruction:

```bash
python scripts/run_compression.py
```

This will:
1. Load the original MoE model (Qwen-3-30B-A3B)
2. For each layer and projection (gate_proj, up_proj, down_proj):
   - Initialize core as mean of expert weights
   - Initialize low-rank wrappers
   - Optimize to minimize reconstruction error
3. Save compressed weights to `models/compressed/`

**Configuration**: Edit `conf/compression/qwen_3_30b.yaml` to adjust:
- `rank`: Low-rank dimension (lower = more compression)
- `num_steps`: Optimization steps
- `lr`: Learning rate

### Phase 2: Knowledge Distillation

Recover performance using knowledge distillation:

```bash
# Start distillation (on training GPUs)
python scripts/run_distillation.py

# Start async evaluation (on evaluation GPUs)
python scripts/run_async_eval.py
```

The distillation script will:
1. Load teacher (original) and student (compressed) models
2. Train student to match teacher's output distribution
3. Save checkpoints periodically

The async evaluation script will:
1. Monitor checkpoint directory
2. Evaluate new checkpoints on benchmarks
3. Log results to WandB

**Configuration**: Edit `conf/distillation/default.yaml` and `conf/evaluation/default.yaml`

## GPU Configuration

The default setup assumes:
- **8 GPUs total**: 2× A100 (80GB) + 6× A6000 (48GB)
- **Phase 1**: All 8 GPUs used in parallel for compression
- **Phase 2**:
  - GPUs 0-5: Distillation training
  - GPUs 6-7: Async evaluation

Modify `gpu_ids` in `conf/config.yaml` to match your setup.

## Expected Results

For Qwen-3-30B-A3B with typical MoE structure:
- **Target compression**: 20-30% of original parameters
- **Active parameters**: May slightly increase during inference
- **Performance**: Recovered through knowledge distillation

Example with rank=64, 8 experts, d_model=4096, d_ffn=14336:
- Original: ~470M params per layer
- Compressed: ~73M params per layer
- **Compression ratio**: 0.155 (84.5% reduction)

## Monitoring

Results are logged to WandB project `moe-compression`. View:
- Compression statistics
- Distillation loss curves
- Evaluation metrics over time

## Advanced Usage

### Custom Model

To compress a different MoE model:

1. Create new config: `conf/compression/my_model.yaml`
2. Update `model.name` in `conf/config.yaml`
3. Adjust layer structure extraction in `src/zero_shot_init.py` if needed

### Adjusting Compression Ratio

- **Higher compression**: Decrease `rank` (e.g., rank=32)
- **Better quality**: Increase `rank` (e.g., rank=128)
- **More optimization**: Increase `num_steps`

### Test Mode

For quick testing with limited data:

```bash
# Edit conf/evaluation/default.yaml
test_mode:
  enabled: true
  limit: 100  # Only 100 samples per task
```

## Citation

If you use this code, please cite:

```bibtex
@article{moe_compression_2025,
  title={Shared-Core Compression for Mixture of Experts Language Models},
  author={Your Name},
  year={2025}
}
```

## License

[Specify your license]

## Acknowledgments

- Builds on Transformers, DeepSpeed, and lm-evaluation-harness
- Inspired by LoRA and other parameter-efficient techniques
