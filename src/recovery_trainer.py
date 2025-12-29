"""
Recovery Training for Compressed MoE Models

Implements a minimal subclass of HuggingFace Trainer for recovery pretraining
on compressed models. Uses FSDP for efficient distributed training.
"""

import torch
import logging
from transformers import (
    Trainer,
    TrainerCallback,
    AutoTokenizer,
)
from pathlib import Path
from typing import Optional, Dict, Any
import json

from src.model_utils import load_compressed_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressedModelCheckpointCallback(TrainerCallback):
    """
    Callback to properly save compressed models during recovery training.

    Saves checkpoints as checkpoint-{step} in the checkpoints_dir for async evaluation.
    Uses custom save logic to handle the compressed model architecture.
    """

    def __init__(self, checkpoints_dir: str, base_model_name: str):
        """
        Args:
            checkpoints_dir: Directory to save checkpoints
            base_model_name: Name of the base model (for reference)
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.base_model_name = base_model_name
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoints_dir}")

    def on_save(self, args, state, control, **kwargs):
        """
        Save compressed model checkpoint.

        This overrides the default save behavior to use our custom checkpoint format.
        Checkpoints are saved as checkpoint-{step} to match the expected format
        for async evaluation.
        """
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')

        if model is None:
            logger.warning("Model is None in on_save callback")
            return control

        # Determine checkpoint directory name
        checkpoint_dir = self.checkpoints_dir / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving compressed model checkpoint to {checkpoint_dir}")

        # Unwrap model if using FSDP or other wrappers
        unwrapped_model = self._unwrap_model(model)

        # Save model state dict
        # For FSDP, this will use the configured state_dict_type
        if hasattr(args, 'fsdp') and args.fsdp:
            # FSDP handles state dict gathering
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig

            # Configure full state dict saving
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(
                unwrapped_model,
                StateDictType.FULL_STATE_DICT,
                save_policy
            ):
                state_dict = unwrapped_model.state_dict()
        else:
            state_dict = unwrapped_model.state_dict()

        # Only save on rank 0 in distributed training
        if state.is_world_process_zero:
            # Save model weights
            torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")

            # Save model config if available
            if hasattr(unwrapped_model, 'config'):
                unwrapped_model.config.save_pretrained(checkpoint_dir)

            # Save tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_dir)

            # Save training metadata
            metadata = {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "base_model": self.base_model_name,
                "is_compressed": True,
            }
            with open(checkpoint_dir / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Checkpoint saved successfully at step {state.global_step}")

        return control

    def _unwrap_model(self, model):
        """
        Unwrap model from distributed wrappers.

        Args:
            model: Potentially wrapped model

        Returns:
            Unwrapped model
        """
        # Handle FSDP wrapper
        if hasattr(model, 'module'):
            return model.module
        return model


class RecoveryTrainer(Trainer):
    """
    Minimal subclass of HuggingFace Trainer for recovery pretraining.

    This trainer:
    1. Uses standard language modeling loss (no distillation)
    2. Saves checkpoints in the format checkpoint-{step}
    3. Supports FSDP for efficient distributed training
    4. Minimal changes to the base Trainer class
    """

    def __init__(
        self,
        checkpoints_dir: str,
        base_model_name: str,
        lr_schedule_config: Optional[list] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            checkpoints_dir: Directory to save checkpoints
            base_model_name: Name of the base model
            lr_schedule_config: Optional staged LR schedule configuration
            *args, **kwargs: Arguments for Trainer
        """
        # Store LR schedule config for create_scheduler
        self.lr_schedule_config = lr_schedule_config

        # Add checkpoint callback
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(
            CompressedModelCheckpointCallback(checkpoints_dir, base_model_name)
        )
        kwargs['callbacks'] = callbacks

        super().__init__(*args, **kwargs)

        logger.info("RecoveryTrainer initialized")
        logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")
        if lr_schedule_config:
            logger.info("Staged LR schedule will be applied")

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Create the learning rate scheduler.

        If lr_schedule_config is provided, wraps the base scheduler with our custom
        parameter-specific scheduler. Otherwise, uses the standard HF scheduler.

        Args:
            num_training_steps: Number of training steps
            optimizer: The optimizer (if None, uses self.optimizer)

        Returns:
            Learning rate scheduler
        """
        # Create the base scheduler using HF's method
        base_scheduler = super().create_scheduler(num_training_steps, optimizer)

        # If we have a staged LR schedule config, wrap the base scheduler
        if self.lr_schedule_config:
            logger.info("Wrapping base scheduler with ParameterGroupLRScheduler")
            from src.custom_lr_scheduler import create_parameter_group_scheduler

            custom_scheduler = create_parameter_group_scheduler(
                optimizer=self.optimizer if optimizer is None else optimizer,
                base_scheduler=base_scheduler,
                lr_schedule_config=self.lr_schedule_config,
                total_steps=num_training_steps,
                model=self.model,
            )

            if custom_scheduler is not None:
                logger.info("Using custom parameter group scheduler")
                return custom_scheduler

            logger.warning("Failed to create custom scheduler, falling back to base scheduler")

        return base_scheduler


def prepare_recovery_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 2048,
    split: str = "train",
    streaming: bool = True,
    num_samples: Optional[int] = None,
    text_column: str = "text"
):
    """
    Prepare dataset for recovery training.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        split: Dataset split
        streaming: Whether to stream the dataset
        num_samples: Number of samples to use (for testing, None for full dataset)
        text_column: Name of the text column in the dataset

    Returns:
        Processed dataset
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Streaming: {streaming}, Max length: {max_length}")

    # Load dataset
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )

    # Limit samples if specified (for testing)
    if num_samples is not None and streaming:
        logger.info(f"Limiting to {num_samples} samples for testing")
        dataset = dataset.take(num_samples)

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize text and prepare labels for language modeling."""
        # Get text from the specified column
        texts = examples.get(text_column)
        if texts is None:
            # Fallback: try to find any text column
            for key in examples.keys():
                if isinstance(examples[key], (list, str)) and key != "id":
                    texts = examples[key]
                    logger.warning(f"Text column '{text_column}' not found, using '{key}'")
                    break

        if texts is None:
            raise ValueError(f"Could not find text column '{text_column}' in dataset")

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )

        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    if streaming:
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    else:
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

    logger.info("Dataset prepared successfully")
    return dataset


def create_recovery_trainer(
    model,
    tokenizer,
    train_dataset,
    training_args,
    checkpoints_dir: str,
    base_model_name: str,
    lr_schedule_config: Optional[list] = None,
) -> RecoveryTrainer:
    """
    Create a RecoveryTrainer instance.

    Args:
        model: The compressed model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        training_args: TrainingArguments
        checkpoints_dir: Directory to save checkpoints
        base_model_name: Name of the base model
        lr_schedule_config: Optional staged LR schedule configuration

    Returns:
        RecoveryTrainer instance
    """
    trainer = RecoveryTrainer(
        checkpoints_dir=checkpoints_dir,
        base_model_name=base_model_name,
        lr_schedule_config=lr_schedule_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    return trainer
