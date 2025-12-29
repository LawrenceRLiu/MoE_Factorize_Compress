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
import os
import torch
import json
import wandb
from pathlib import Path
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.async_eval import Evaluator, EvalConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES")
N_GPUS = len(VISIBLE.split(",")) if VISIBLE else torch.cuda.device_count()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for async evaluation.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("MoE Compression - Asynchronous Evaluation")
    logger.info(f"Detected {N_GPUS} GPUs for evaluation")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # load the wandb if available
    config_path = Path(cfg.output.config_dir) / "wandb_run_info.json"
    wandb_run = None
    if config_path.exists():
        with open(config_path, "r") as f:
            wandb_info = json.load(f)
        logger.info(f"WandB Run Info: {wandb_info}")
        #initialize wandb
        wandb_run = wandb.init(
            project=wandb_info["project"],
            entity=wandb_info["entity"],
            name=wandb_info["name"],
            id=wandb_info["id"],
            resume="must",
        )
        logger.info(f"Resumed WandB run: {wandb_run.name} (ID: {wandb_run.id})")


    # Create eval config
    eval_config = EvalConfig(
        checkpoint_dir=cfg.output.checkpoints_dir,
        temp_dir=cfg.output.temp_dir,
        eval_dir=cfg.output.eval_dir,
        eval_tasks=cfg.evaluation.tasks,
        config_dir=cfg.output.config_dir,
        batch_size=cfg.evaluation.batch_size,
        n_gpus=N_GPUS,
        n_gpus_per_model=cfg.evaluation.async_eval.n_gpus_per_model,
        eval_interval=cfg.evaluation.async_eval.eval_interval,
        wandb_run=wandb_run
    )
        

    logger.info("\nEvaluation Configuration:")
    logger.info(f"  {asdict(eval_config)}")
    # Run async evaluation
    try:
        evaluator = Evaluator(eval_config)
        evaluator.run()

    except KeyboardInterrupt:
        logger.info("\nEvaluation stopped by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
