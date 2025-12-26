# FSDP (Fully Sharded Data Parallel) Guide for Knowledge Distillation

This guide explains how to use FSDP for distributed training of compressed MoE models.

## What is FSDP?

Fully Sharded Data Parallel (FSDP) is PyTorch's native solution for training large models across multiple GPUs. It shards:
- Model parameters
- Gradients
- Optimizer states

This allows training models that don't fit on a single GPU.

## Quick Start

### 1. Enable FSDP in Configuration

Edit `conf/distillation/default.yaml`:

```yaml
training:
  # Enable FSDP
  fsdp: "full_shard"  # or "shard_grad_op" for less aggressive sharding
  fsdp_config:
    fsdp_transformer_layer_cls_to_wrap: ["Qwen3MoeDecoderLayer"]  # Adjust for your model
    fsdp_backward_prefetch: "backward_pre"
    fsdp_state_dict_type: "FULL_STATE_DICT"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
```

### 2. Launch with torchrun

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 \
    scripts/run_distillation.py

# Multi-node (e.g., 2 nodes, 8 GPUs each)
# On node 0:
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<NODE0_IP> \
    --master_port=29500 \
    scripts/run_distillation.py

# On node 1:
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<NODE0_IP> \
    --master_port=29500 \
    scripts/run_distillation.py
```

## FSDP Sharding Strategies

### FULL_SHARD (Recommended)
- **What**: Shards parameters, gradients, and optimizer states
- **Memory**: Most efficient (~1/N memory per GPU for N GPUs)
- **Speed**: Moderate (communication overhead)
- **Use when**: Model doesn't fit on single GPU even with gradient checkpointing

```yaml
training:
  fsdp: "full_shard"
```

### SHARD_GRAD_OP
- **What**: Shards gradients and optimizer states only
- **Memory**: Less efficient than FULL_SHARD
- **Speed**: Faster than FULL_SHARD (less communication)
- **Use when**: Model fits on GPU but optimizer states don't

```yaml
training:
  fsdp: "shard_grad_op"
```

### HYBRID_SHARD
- **What**: FULL_SHARD within node, DDP across nodes
- **Memory**: Efficient within node
- **Speed**: Fast cross-node communication
- **Use when**: Multi-node training with fast intra-node networking

```yaml
training:
  fsdp: "hybrid_shard"
```

### NO_SHARD (Equivalent to DDP)
- **What**: No sharding, full model replica on each GPU
- **Memory**: Least efficient
- **Speed**: Fastest (no sharding overhead)
- **Use when**: Model fits comfortably on single GPU, just want data parallelism

```yaml
training:
  fsdp: "no_shard"
```

## Configuration Options

### Auto Wrap Policy

Controls how model layers are wrapped for sharding:

```yaml
fsdp_config:
  # Option 1: Transformer-based (recommended for transformers)
  fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
  fsdp_transformer_layer_cls_to_wrap: ["Qwen3MoeDecoderLayer"]

  # Option 2: Size-based (wrap layers above certain size)
  fsdp_auto_wrap_policy: "SIZE_BASED_WRAP"
  fsdp_min_num_params: 1e8  # 100M parameters
```

**How to find your transformer layer class:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("your-model")
print(model)  # Look for the decoder layer class name
```

### Backward Prefetch

Prefetches next layer's parameters during backward pass:

```yaml
fsdp_config:
  # Options: "backward_pre", "backward_post", null
  fsdp_backward_prefetch: "backward_pre"  # Recommended
```

- `backward_pre`: Prefetch before current layer backward (faster, more memory)
- `backward_post`: Prefetch after current layer backward (slower, less memory)
- `null`: No prefetching (slowest, least memory)

### CPU Offloading

Offload parameters/gradients to CPU to save GPU memory:

```yaml
fsdp_config:
  # Offload parameters to CPU
  fsdp_offload_params: true

  # Note: This significantly slows down training!
  # Only use if you can't fit model in GPU memory otherwise
```

**Trade-off**: Saves GPU memory but slows training by 2-3x due to CPU-GPU transfer.

### State Dict Type

Controls how checkpoints are saved:

```yaml
fsdp_config:
  fsdp_state_dict_type: "FULL_STATE_DICT"  # Recommended
```

Options:
- `FULL_STATE_DICT`: Save full unsharded model (can load anywhere, larger files)
- `SHARDED_STATE_DICT`: Save sharded model (smaller files, harder to load)
- `LOCAL_STATE_DICT`: Save each rank's shard separately

## Memory Optimization

### 1. Gradient Checkpointing + FSDP

Combine for maximum memory efficiency:

```yaml
training:
  gradient_checkpointing: true
  fsdp: "full_shard"
```

### 2. Mixed Precision

Use bf16 or fp16:

```yaml
training:
  bf16: true  # Recommended for FSDP
  fp16: false
```

### 3. Gradient Accumulation

Reduce per-GPU batch size, increase accumulation:

```yaml
training:
  per_device_train_batch_size: 1  # Minimal per-GPU
  gradient_accumulation_steps: 64  # Effective batch = 1 × 64 × num_gpus
```

## Complete FSDP Configuration Example

For Qwen3-30B compressed model on 8x A100 (80GB each):

```yaml
# conf/distillation/default.yaml

training:
  # FSDP settings
  fsdp: "full_shard"
  fsdp_config:
    fsdp_transformer_layer_cls_to_wrap: ["Qwen3MoeDecoderLayer"]
    fsdp_backward_prefetch: "backward_pre"
    fsdp_forward_prefetch: false
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_state_dict_type: "FULL_STATE_DICT"
    fsdp_offload_params: false  # We have enough GPU memory

  # Memory optimization
  gradient_checkpointing: true
  bf16: true

  # Batch settings
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

  # Other settings
  learning_rate: 5e-5
  num_train_epochs: 3
  save_steps: 500
  logging_steps: 10
```

Launch with:
```bash
torchrun --nproc_per_node=8 scripts/run_distillation.py
```

## Troubleshooting

### Issue: "RuntimeError: NCCL error"

**Solutions**:
1. Check all GPUs are visible: `nvidia-smi`
2. Set NCCL debug: `export NCCL_DEBUG=INFO`
3. Try different backend: `export NCCL_SOCKET_IFNAME=eth0`
4. Increase timeout:
   ```yaml
   fsdp_config:
     fsdp_sync_module_states: true
   ```

### Issue: Out of memory even with FSDP

**Solutions**:
1. Enable CPU offloading:
   ```yaml
   fsdp_config:
     fsdp_offload_params: true
   ```
2. Reduce batch size to 1
3. Increase gradient accumulation
4. Enable gradient checkpointing
5. Try `fsdp_forward_prefetch: false`

### Issue: Training is very slow

**Solutions**:
1. Disable CPU offloading if enabled
2. Use `backward_prefetch: "backward_pre"`
3. Enable `forward_prefetch: true`
4. Check network bandwidth between nodes
5. Use NCCL optimizations:
   ```bash
   export NCCL_IB_DISABLE=0
   export NCCL_NET_GDR_LEVEL=2
   ```

### Issue: Checkpoint is too large

**Solutions**:
1. Use sharded checkpoints:
   ```yaml
   fsdp_config:
     fsdp_state_dict_type: "SHARDED_STATE_DICT"
   ```
2. Save fewer checkpoints:
   ```yaml
   training:
     save_total_limit: 2
   ```

## FSDP vs DeepSpeed

| Feature | FSDP | DeepSpeed ZeRO |
|---------|------|----------------|
| Native PyTorch | ✓ | ✗ |
| Easy setup | ✓ | ✗ (requires config) |
| HF Trainer integration | ✓ | ✓ |
| CPU offloading | ✓ | ✓ |
| NVMe offloading | ✗ | ✓ |
| Activation checkpointing | ✓ | ✓ |
| Memory efficiency | Excellent | Excellent |

**Recommendation**: Use FSDP for simplicity. Use DeepSpeed if you need NVMe offloading or specific ZeRO optimizations.

## Performance Tips

### 1. Network Optimization (Multi-node)

```bash
# InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1

# GPUDirect RDMA
export NCCL_NET_GDR_LEVEL=2
export NCCL_NET_GDR_READ=1
```

### 2. Pin Memory

```yaml
training:
  dataloader_pin_memory: true
  dataloader_num_workers: 4
```

### 3. Prefetching

```yaml
fsdp_config:
  fsdp_backward_prefetch: "backward_pre"
  fsdp_forward_prefetch: true  # Only if you have memory
```

### 4. Optimal Batch Size

Find the largest batch size that fits in memory:
```bash
# Start small and increase
for bs in 1 2 4 8; do
    torchrun --nproc_per_node=8 \
        scripts/run_distillation.py \
        distillation.training.per_device_train_batch_size=$bs
done
```

## Monitoring FSDP Training

### GPU Memory Usage

```bash
watch -n 1 nvidia-smi
```

### NCCL Communication

```bash
export NCCL_DEBUG=INFO
```

### Training Metrics

HuggingFace Trainer automatically logs:
- Samples per second
- Training loss
- GPU memory usage
- Learning rate

## Example Launch Scripts

### Single Node (8 GPUs)

```bash
#!/bin/bash
# launch_fsdp.sh

export NCCL_DEBUG=INFO

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/run_distillation.py \
    distillation.training.fsdp="full_shard"
```

### Multi-Node (2 nodes, 8 GPUs each)

Node 0:
```bash
#!/bin/bash
# launch_node0.sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/run_distillation.py
```

Node 1:
```bash
#!/bin/bash
# launch_node1.sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/run_distillation.py
```

## Summary

**For single node (8 GPUs)**:
- Use `fsdp: "full_shard"`
- Enable gradient checkpointing
- Use bf16 precision
- Launch with: `torchrun --nproc_per_node=8 scripts/run_distillation.py`

**For multi-node**:
- Use `fsdp: "hybrid_shard"` for better cross-node performance
- Optimize NCCL settings for your network
- Launch with coordinated `torchrun` on each node

**Memory tips**:
- Model fits on GPU → Use `fsdp: "shard_grad_op"` or no FSDP
- Model doesn't fit → Use `fsdp: "full_shard"` + gradient checkpointing
- Still OOM → Enable CPU offloading (but training will be slow)

FSDP makes distributed training easy with minimal code changes!
