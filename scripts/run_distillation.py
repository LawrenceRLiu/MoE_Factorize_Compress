#!/usr/bin/env python3
"""
Run knowledge distillation training for compressed MoE models.

This script uses Hydra for configuration management, ensuring consistency
with other scripts in the pipeline.

Usage:
    python scripts/run_distillation.py

Configuration is loaded from conf/config.yaml and conf/distillation/default.yaml
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.distillation_utils import (
    setup_distillation_pipeline,
    run_distillation_training
)
from src.utils import set_seed, log_gpu_memory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Main distillation training function.

    Args:
        config: Hydra configuration
    """
    logger.info("="*80)
    logger.info("Knowledge Distillation Training")
    logger.info("="*80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(config))

    # Log GPU information
    logger.info("\nGPU Information:")
    log_gpu_memory()

    # Set random seed
    set_seed(config.seed)
    logger.info(f"\nRandom seed set to: {config.seed}")

    # Setup distillation pipeline
    logger.info("\n" + "="*80)
    logger.info("Setting up distillation pipeline...")
    logger.info("="*80 + "\n")

    # Resolve model paths
    teacher_path = config.distillation.teacher.model_path
    if teacher_path is None:
        teacher_path = config.model.name
        logger.info(f"Using model.name as teacher: {teacher_path}")

    student_path = config.distillation.student.model_path
    if student_path is None:
        # Default to checkpoint-0 from zero-shot initialization
        student_path = str(Path(config.output.checkpoints_dir) / "checkpoint-0")
        logger.info(f"Using checkpoint-0 as student: {student_path}")

    # Check if student model exists
    if not Path(student_path).exists():
        logger.error(f"Student model not found at: {student_path}")
        logger.error("Please run compression/zero-shot initialization first!")
        raise FileNotFoundError(f"Student model not found: {student_path}")

    # Setup pipeline
    student_model, train_dataset, eval_dataset, tokenizer = setup_distillation_pipeline(
        config=config.distillation,
        teacher_model_path=teacher_path,
        student_model_path=student_path,
        force_regenerate_logits=False  # Set to True to regenerate cached logits
    )

    logger.info("\n" + "="*80)
    logger.info("Pipeline setup complete!")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    logger.info("="*80)

    # Run distillation training
    logger.info("\n" + "="*80)
    logger.info("Starting distillation training...")
    logger.info("="*80 + "\n")

    trainer = run_distillation_training(
        config=config.distillation,
        student_model=student_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Save final model
    final_output_dir = Path(config.distillation.training.output_dir) / "final"
    logger.info(f"\nSaving final model to {final_output_dir}")
    trainer.save_model(str(final_output_dir))

    # Save tokenizer
    tokenizer.save_pretrained(str(final_output_dir))
    logger.info(f"Tokenizer saved to {final_output_dir}")

    logger.info("\n" + "="*80)
    logger.info("Distillation training complete!")
    logger.info("="*80 + "\n")

    # Final GPU memory info
    logger.info("Final GPU memory state:")
    log_gpu_memory()


if __name__ == "__main__":
    main()
