"""
Utility functions for MoE compression.
"""

import torch
import random
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device_map(gpu_ids: list, model_name: str = None) -> Dict[str, int]:
    """
    Create device map for model parallelism.

    Args:
        gpu_ids: List of GPU IDs to use
        model_name: Optional model name for auto device mapping

    Returns:
        Device map dictionary
    """
    if len(gpu_ids) == 1:
        return {"": gpu_ids[0]}
    else:
        # Auto device map across multiple GPUs
        return "auto"


def print_model_size(model: torch.nn.Module):
    """
    Print model size statistics.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model size:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Size (GB): {total_params * 4 / 1e9:.2f}")  # Assuming fp32


def count_moe_parameters(model) -> Dict[str, Any]:
    """
    Count parameters in MoE model, separating shared and expert-specific.

    Args:
        model: MoE model

    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    expert_params = 0
    shared_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        # Heuristic: if 'expert' or 'moe' in name, count as expert param
        if 'expert' in name.lower() or 'moe' in name.lower():
            expert_params += param_count
        else:
            shared_params += param_count

    return {
        "total": total_params,
        "expert": expert_params,
        "shared": shared_params,
        "expert_ratio": expert_params / total_params if total_params > 0 else 0
    }


def save_compression_stats(
    output_dir: str,
    stats: Dict[str, Any],
    filename: str = "compression_stats.json"
):
    """
    Save compression statistics to file.

    Args:
        output_dir: Output directory
        stats: Statistics dictionary
        filename: Output filename
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved compression stats to {output_path}")


def load_compression_stats(
    compressed_dir: str,
    filename: str = "compression_stats.json"
) -> Dict[str, Any]:
    """
    Load compression statistics from file.

    Args:
        compressed_dir: Directory containing compressed model
        filename: Stats filename

    Returns:
        Statistics dictionary
    """
    stats_path = Path(compressed_dir) / filename

    if not stats_path.exists():
        logger.warning(f"Stats file not found: {stats_path}")
        return {}

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats


def estimate_memory_usage(
    num_params: int,
    dtype: torch.dtype = torch.bfloat16,
    overhead_factor: float = 1.2
) -> float:
    """
    Estimate memory usage for a model.

    Args:
        num_params: Number of parameters
        dtype: Data type
        overhead_factor: Multiplicative factor for activation memory, etc.

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    base_memory_gb = (num_params * bytes_per_param) / 1e9
    total_memory_gb = base_memory_gb * overhead_factor

    return total_memory_gb


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Get memory information for all available GPUs.

    Returns:
        Dictionary mapping GPU ID to memory info (total, used, free in GB)
    """
    if not torch.cuda.is_available():
        return {}

    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total - reserved

        gpu_info[i] = {
            "name": props.name,
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free
        }

    return gpu_info


def log_gpu_memory():
    """Log GPU memory usage for all devices."""
    gpu_info = get_gpu_memory_info()

    if not gpu_info:
        logger.info("No GPUs available")
        return

    logger.info("GPU Memory Usage:")
    for gpu_id, info in gpu_info.items():
        logger.info(
            f"  GPU {gpu_id} ({info['name']}): "
            f"{info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB "
            f"(Free: {info['free_gb']:.2f}GB)"
        )


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")
