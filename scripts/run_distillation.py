#!/usr/bin/env python
"""
Main script for knowledge distillation of compressed MoE models.

Usage:
    python scripts/run_distillation.py [--config-name CONFIG]
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation import run_distillation, DistillationConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for knowledge distillation.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("MoE Compression - Knowledge Distillation")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create distillation config
    distillation_config = DistillationConfig(
        teacher_model=cfg.model.name,
        student_model_path=cfg.output.compressed_dir,
        output_dir=cfg.output.distilled_dir,
        dataset_name=cfg.distillation.dataset.name,
        dataset_split=cfg.distillation.dataset.split,
        max_length=cfg.distillation.dataset.max_length,
        temperature=cfg.distillation.temperature,
        alpha=cfg.distillation.alpha,
        teacher_load_in_8bit=cfg.distillation.teacher.load_in_8bit,
        teacher_load_in_4bit=cfg.distillation.teacher.load_in_4bit,
        per_device_train_batch_size=cfg.distillation.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.distillation.training.gradient_accumulation_steps,
        learning_rate=cfg.distillation.training.learning_rate,
        num_train_epochs=cfg.distillation.training.num_train_epochs,
        max_steps=cfg.distillation.training.max_steps,
        warmup_steps=cfg.distillation.training.warmup_steps,
        logging_steps=cfg.distillation.logging.logging_steps,
        save_steps=cfg.distillation.logging.save_steps,
        eval_steps=cfg.distillation.logging.eval_steps,
        bf16=cfg.distillation.training.bf16,
        wandb_project=cfg.wandb_project,
        wandb_run_name=f"{cfg.experiment_name}_distillation"
    )

    logger.info("\nDistillation Configuration:")
    logger.info(f"  Teacher: {distillation_config.teacher_model}")
    logger.info(f"  Student: {distillation_config.student_model_path}")
    logger.info(f"  Dataset: {distillation_config.dataset_name}")
    logger.info(f"  Temperature: {distillation_config.temperature}")
    logger.info(f"  Alpha: {distillation_config.alpha}")
    logger.info(f"  Output: {distillation_config.output_dir}")

    # Run distillation
    try:
        run_distillation(distillation_config)

        logger.info("\n" + "=" * 80)
        logger.info("Distillation completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Distillation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
