"""
Recovery Training Script for Compressed MoE Models

Main script for Phase 2: Recovery pretraining on compressed models.
Uses Hydra for configuration management and FSDP for distributed training.

Usage:
    python scripts/run_recovery_training.py \
        model.compressed_checkpoint=/path/to/checkpoint-0 \
        experiment_name=my_experiment
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
import wandb

from src.recovery_trainer import (
    RecoveryTrainer,
    prepare_recovery_dataset,
    create_recovery_trainer,
)
from src.model_utils import load_compressed_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_fsdp_args(cfg: DictConfig, training_args_dict: dict) -> dict:
    """
    Configure FSDP settings for TrainingArguments.

    Args:
        cfg: Hydra configuration
        training_args_dict: Dictionary of training arguments

    Returns:
        Updated training arguments dictionary
    """
    if not cfg.recovery.fsdp.enabled:
        logger.info("FSDP is disabled")
        return training_args_dict

    logger.info("Configuring FSDP for distributed training")

    # Basic FSDP configuration
    fsdp_config = cfg.recovery.fsdp

    # Set FSDP strategy
    training_args_dict["fsdp"] = fsdp_config.fsdp_sharding_strategy

    # FSDP-specific settings
    training_args_dict["fsdp_config"] = {
        "fsdp_offload_params": fsdp_config.fsdp_offload_params,
        "fsdp_state_dict_type": fsdp_config.fsdp_state_dict_type,
        "fsdp_backward_prefetch": fsdp_config.fsdp_backward_prefetch,
        "fsdp_forward_prefetch": fsdp_config.fsdp_forward_prefetch,
        "fsdp_use_orig_params": fsdp_config.fsdp_use_orig_params,
    }

    # Auto-wrap transformer layers if not specified
    if fsdp_config.fsdp_transformer_layer_cls_to_wrap is not None:
        training_args_dict["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] = \
            fsdp_config.fsdp_transformer_layer_cls_to_wrap

    logger.info(f"FSDP Configuration: {training_args_dict['fsdp_config']}")

    return training_args_dict


def load_model_and_tokenizer(cfg: DictConfig):
    """
    Load compressed model and tokenizer.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading tokenizer and model")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.recovery.model.trust_remote_code
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Determine compressed checkpoint path
    compressed_checkpoint = cfg.recovery.model.compressed_checkpoint
    if compressed_checkpoint is None:
        # Default to checkpoint-0 in the checkpoints directory
        compressed_checkpoint = str(Path(cfg.output.checkpoints_dir) / "checkpoint-0")
        logger.info(f"No compressed checkpoint specified, using default: {compressed_checkpoint}")

    # Verify checkpoint exists
    checkpoint_path = Path(compressed_checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(
            f"Compressed checkpoint not found: {compressed_checkpoint}\n"
            f"Please run zero-shot initialization first or specify a valid checkpoint path."
        )

    logger.info(f"Loading compressed model from: {compressed_checkpoint}")

    # Load compressed model
    # Note: For FSDP, we don't use device_map="auto"
    dtype = getattr(torch, cfg.recovery.model.dtype)
    device_map = cfg.recovery.model.device_map

    try:
        model = load_compressed_model(
            compressed_dir=compressed_checkpoint,
            original_model_name=cfg.model.name,
            device_map=device_map,  # None for FSDP
            dtype=dtype,
            trust_remote_code=cfg.recovery.model.trust_remote_code
        )
        logger.info("Compressed model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load compressed model: {e}")
        raise

    # Enable gradient checkpointing if specified
    if cfg.recovery.training.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    return model, tokenizer


def setup_wandb(cfg: DictConfig):
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: Hydra configuration
    """
    if not cfg.recovery.wandb.enabled:
        logger.info("WandB logging is disabled")
        return

    # Generate run name if not provided
    run_name = cfg.recovery.wandb.run_name
    if run_name is None:
        run_name = f"recovery-{cfg.experiment_name}"

    logger.info(f"Initializing WandB: project={cfg.wandb_project}, run={run_name}")

    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.recovery.wandb.tags,
        notes=cfg.recovery.wandb.notes,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("="*80)
    logger.info("Recovery Training Configuration:")
    logger.info("="*80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*80)

    # Set random seed
    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    # Create output directories
    output_base = Path(cfg.output.base_dir)
    checkpoints_dir = Path(cfg.output.checkpoints_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_base}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")

    # Save configuration to output directory
    config_path = output_base / "recovery_config.yaml"
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Configuration saved to: {config_path}")

    # Setup WandB
    setup_wandb(cfg)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Prepare dataset
    logger.info("Preparing training dataset")
    train_dataset = prepare_recovery_dataset(
        dataset_name=cfg.recovery.dataset.name,
        tokenizer=tokenizer,
        max_length=cfg.recovery.dataset.max_length,
        split=cfg.recovery.dataset.split,
        streaming=cfg.recovery.dataset.streaming,
        num_samples=cfg.recovery.dataset.num_samples,
        text_column=cfg.recovery.dataset.text_column,
    )

    # Setup training arguments
    logger.info("Setting up training arguments")

    training_config = cfg.recovery.training

    training_args_dict = {
        # Output
        "output_dir": str(checkpoints_dir / "trainer_state"),
        "overwrite_output_dir": True,

        # Batch size and accumulation
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,

        # Learning rate
        "learning_rate": training_config.learning_rate,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "warmup_steps": training_config.warmup_steps,

        # Training duration
        "num_train_epochs": training_config.num_train_epochs,
        "max_steps": training_config.max_steps,

        # Optimization
        "optim": training_config.optim,
        "weight_decay": training_config.weight_decay,
        "max_grad_norm": training_config.max_grad_norm,

        # Precision
        "bf16": training_config.bf16,
        "fp16": training_config.fp16,

        # Checkpointing
        "save_steps": training_config.save_steps,
        "save_total_limit": training_config.save_total_limit,
        "save_strategy": "steps",

        # Logging
        "logging_steps": training_config.logging_steps,
        "logging_first_step": training_config.logging_first_step,
        "report_to": "wandb" if cfg.recovery.wandb.enabled else "none",

        # Evaluation
        "evaluation_strategy": training_config.evaluation_strategy,
        "eval_steps": training_config.eval_steps if training_config.evaluation_strategy == "steps" else None,

        # Other
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "dataloader_num_workers": training_config.dataloader_num_workers,
        "remove_unused_columns": training_config.remove_unused_columns,
        "ddp_find_unused_parameters": False,
    }

    # Add FSDP configuration
    training_args_dict = setup_fsdp_args(cfg, training_args_dict)

    # Create TrainingArguments
    training_args = TrainingArguments(**training_args_dict)

    # Create trainer
    logger.info("Creating RecoveryTrainer")
    trainer = create_recovery_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        checkpoints_dir=str(checkpoints_dir),
        base_model_name=cfg.model.name,
    )

    # Start training
    logger.info("="*80)
    logger.info("Starting recovery training")
    logger.info("="*80)

    try:
        trainer.train()
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        # Save final checkpoint
        if trainer.state.is_world_process_zero:
            final_checkpoint = checkpoints_dir / f"checkpoint-{trainer.state.global_step}"
            logger.info(f"Saving final checkpoint to: {final_checkpoint}")

        # Finish WandB run
        if cfg.recovery.wandb.enabled:
            wandb.finish()

    logger.info("Recovery training script completed")


if __name__ == "__main__":
    main()
