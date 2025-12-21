#!/usr/bin/env python
"""
Main script for zero-shot compression of MoE models.

Usage:
    python scripts/run_compression.py [--config-name CONFIG]
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.zero_shot_init import parallel_compression, CompressionConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for zero-shot compression.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("MoE Compression - Zero-Shot Initialization")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create compression config
    compression_config = CompressionConfig(
        model_name=cfg.model.name,
        rank=cfg.compression.rank,
        num_steps=cfg.compression.num_steps,
        lr=cfg.compression.lr,
        output_dir=cfg.output.compressed_dir,
        projections=cfg.compression.projections
    )

    logger.info("\nCompression Configuration:")
    logger.info(f"  Model: {compression_config.model_name}")
    logger.info(f"  Rank: {compression_config.rank}")
    logger.info(f"  Optimization steps: {compression_config.num_steps}")
    logger.info(f"  Learning rate: {compression_config.lr}")
    logger.info(f"  Projections: {compression_config.projections}")
    logger.info(f"  Output directory: {compression_config.output_dir}")

    # GPU configuration
    gpu_ids = cfg.gpu_ids
    logger.info(f"\nGPU Configuration:")
    logger.info(f"  GPUs available: {gpu_ids}")

    # Run parallel compression
    try:
        parallel_compression(
            model_name=compression_config.model_name,
            config=compression_config,
            gpu_ids=gpu_ids,
            num_layers=cfg.compression.get('num_layers', None)
        )

        logger.info("\n" + "=" * 80)
        logger.info("Compression completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Compression failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
