#!/usr/bin/env python
"""
Main script for asynchronous checkpoint evaluation.

Usage:
    python scripts/run_async_eval.py [--config-name CONFIG]
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.async_eval import CheckpointEvaluator, EvalConfig, evaluate_baseline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for async evaluation.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("MoE Compression - Asynchronous Evaluation")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Optionally evaluate baseline first
    if cfg.evaluation.evaluate_baseline:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating Baseline Model")
        logger.info("=" * 80)

        baseline_results = evaluate_baseline(
            model_name=cfg.model.name,
            eval_tasks=cfg.evaluation.tasks,
            output_dir=cfg.output.eval_dir,
            gpu_ids=cfg.evaluation.async_eval.gpu_ids,
            batch_size=cfg.evaluation.batch_size,
            num_fewshot=cfg.evaluation.num_fewshot
        )

        logger.info(f"\nBaseline results: {baseline_results}")

    # Create eval config
    eval_config = EvalConfig(
        checkpoint_dir=cfg.output.distilled_dir,
        eval_tasks=cfg.evaluation.tasks,
        gpu_ids=cfg.evaluation.async_eval.gpu_ids,
        eval_batch_size=cfg.evaluation.batch_size,
        eval_interval=cfg.evaluation.async_eval.eval_interval,
        wandb_project=cfg.wandb_project,
        wandb_run_name=f"{cfg.experiment_name}_eval",
        num_fewshot=cfg.evaluation.num_fewshot,
        limit=cfg.evaluation.test_mode.limit if cfg.evaluation.test_mode.enabled else None
    )

    logger.info("\nEvaluation Configuration:")
    logger.info(f"  Checkpoint directory: {eval_config.checkpoint_dir}")
    logger.info(f"  Tasks: {eval_config.eval_tasks}")
    logger.info(f"  GPUs: {eval_config.gpu_ids}")
    logger.info(f"  Eval interval: {eval_config.eval_interval}s")

    # Run async evaluation
    try:
        evaluator = CheckpointEvaluator(eval_config)
        evaluator.run()

    except KeyboardInterrupt:
        logger.info("\nEvaluation stopped by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
