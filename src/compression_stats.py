"""
Compression statistics tracking module.

Tracks both total parameters (all experts) and active parameters (per-token).
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CompressionStats:
    """
    Track compression statistics for MoE models.

    Key metrics:
    - Total parameters: All expert weights combined
    - Active parameters: Parameters used per token (1-2 experts typically)
    """

    def __init__(self):
        self.layer_stats = []

    def add_layer_stats(
        self,
        layer_idx: int,
        projection: str,
        num_experts: int,
        d_in: int,
        d_out: int,
        rank: int,
        num_active_experts: int = 2  # Typical for MoE (top-k routing)
    ):
        """
        Add statistics for a single layer/projection.

        Args:
            layer_idx: Layer index
            projection: Projection name (gate_proj, up_proj, down_proj)
            num_experts: Total number of experts
            d_in: Input dimension
            d_out: Output dimension
            rank: Low-rank adapter rank
            num_active_experts: Number of experts activated per token (default=2)
        """
        # Original model parameters
        original_total_params = num_experts * d_in * d_out
        original_active_params = num_active_experts * d_in * d_out

        # Compressed model parameters
        # Total: shared core + all expert wrappers
        core_params = d_in * d_out
        wrapper_params_per_expert = 2 * d_in * rank + 2 * d_out * rank
        total_wrapper_params = num_experts * wrapper_params_per_expert
        compressed_total_params = core_params + total_wrapper_params

        # Active: shared core + active expert wrappers
        active_wrapper_params = num_active_experts * wrapper_params_per_expert
        compressed_active_params = core_params + active_wrapper_params

        # Ratios
        total_compression_ratio = compressed_total_params / original_total_params
        active_compression_ratio = compressed_active_params / original_active_params

        stats = {
            "layer_idx": layer_idx,
            "projection": projection,
            "num_experts": num_experts,
            "num_active_experts": num_active_experts,
            "d_in": d_in,
            "d_out": d_out,
            "rank": rank,

            # Original model
            "original_total_params": original_total_params,
            "original_active_params": original_active_params,

            # Compressed model
            "compressed_total_params": compressed_total_params,
            "compressed_active_params": compressed_active_params,
            "core_params": core_params,
            "wrapper_params_per_expert": wrapper_params_per_expert,

            # Compression ratios
            "total_compression_ratio": total_compression_ratio,
            "total_reduction_percent": (1 - total_compression_ratio) * 100,
            "active_compression_ratio": active_compression_ratio,
            "active_increase_percent": (active_compression_ratio - 1) * 100,
        }

        self.layer_stats.append(stats)

        logger.info(f"Layer {layer_idx} {projection} compression stats:")
        logger.info(f"  Total params: {original_total_params:,} -> {compressed_total_params:,} "
                   f"({total_compression_ratio:.3f}x, {stats['total_reduction_percent']:.1f}% reduction)")
        logger.info(f"  Active params: {original_active_params:,} -> {compressed_active_params:,} "
                   f"({active_compression_ratio:.3f}x, {stats['active_increase_percent']:.1f}% change)")

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all layers.

        Returns:
            Dictionary with total compression statistics
        """
        if not self.layer_stats:
            return {}

        total_original_total = sum(s["original_total_params"] for s in self.layer_stats)
        total_original_active = sum(s["original_active_params"] for s in self.layer_stats)
        total_compressed_total = sum(s["compressed_total_params"] for s in self.layer_stats)
        total_compressed_active = sum(s["compressed_active_params"] for s in self.layer_stats)

        total_ratio = total_compressed_total / total_original_total
        active_ratio = total_compressed_active / total_original_active

        return {
            "num_layers_compressed": len(self.layer_stats),

            # Total parameters (all experts)
            "total_params_original": total_original_total,
            "total_params_compressed": total_compressed_total,
            "total_compression_ratio": total_ratio,
            "total_reduction_percent": (1 - total_ratio) * 100,

            # Active parameters (per token)
            "active_params_original": total_original_active,
            "active_params_compressed": total_compressed_active,
            "active_compression_ratio": active_ratio,
            "active_change_percent": (active_ratio - 1) * 100,

            # Memory estimates (assuming bfloat16)
            "total_memory_original_gb": total_original_total * 2 / 1e9,
            "total_memory_compressed_gb": total_compressed_total * 2 / 1e9,
            "memory_saved_gb": (total_original_total - total_compressed_total) * 2 / 1e9,

            # Per-layer breakdown
            "per_layer_stats": self.layer_stats
        }

    def save(self, output_path: str):
        """
        Save statistics to YAML file.

        Args:
            output_path: Path to save statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_aggregate_stats()

        with open(output_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved compression statistics to {output_path}")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("COMPRESSION SUMMARY")
        logger.info("="*80)
        logger.info(f"Layers compressed: {stats['num_layers_compressed']}")
        logger.info(f"\nTOTAL PARAMETERS (all experts):")
        logger.info(f"  Original:   {stats['total_params_original']:,} ({stats['total_memory_original_gb']:.2f} GB)")
        logger.info(f"  Compressed: {stats['total_params_compressed']:,} ({stats['total_memory_compressed_gb']:.2f} GB)")
        logger.info(f"  Ratio:      {stats['total_compression_ratio']:.4f} ({stats['total_params_compressed']/stats['total_params_original']*100:.1f}% of original)")
        logger.info(f"  Reduction:  {stats['total_reduction_percent']:.2f}%")
        logger.info(f"  Saved:      {stats['memory_saved_gb']:.2f} GB")
        logger.info(f"\nACTIVE PARAMETERS (per token, top-k experts):")
        logger.info(f"  Original:   {stats['active_params_original']:,}")
        logger.info(f"  Compressed: {stats['active_params_compressed']:,}")
        logger.info(f"  Ratio:      {stats['active_compression_ratio']:.4f}")
        logger.info(f"  Change:     {stats['active_change_percent']:+.2f}%")
        logger.info("="*80 + "\n")

    @staticmethod
    def load(input_path: str) -> Dict[str, Any]:
        """
        Load statistics from YAML file.

        Args:
            input_path: Path to statistics file

        Returns:
            Statistics dictionary
        """
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)


def infer_num_active_experts(model_config) -> int:
    """
    Infer the number of active experts from model config.

    Args:
        model_config: HuggingFace model config

    Returns:
        Number of active experts (top-k)
    """
    # Check common config fields
    if hasattr(model_config, 'num_experts_per_tok'):
        return model_config.num_experts_per_tok
    elif hasattr(model_config, 'num_selected_experts'):
        return model_config.num_selected_experts
    elif hasattr(model_config, 'top_k'):
        return model_config.top_k
    else:
        # Default to 2 for most MoE models
        logger.warning("Could not infer num_active_experts from config, defaulting to 2")
        return 2
