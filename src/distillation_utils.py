"""
Utility functions for knowledge distillation training.

Includes data preparation, model loading, and training pipeline helpers.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging
from omegaconf import DictConfig
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

import os

from src.model_utils import get_model
from src.teacher_logits import TeacherLogitsGenerator, CachedLogitsDataset
from src.distillation_trainer import (
    DistillationTrainer,
    DistillationTrainingArguments,
    create_distillation_trainer
)

logger = logging.getLogger(__name__)


def prepare_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = False,
    num_proc: int = 4
) -> Dataset:
    """
    Load and tokenize a dataset for language modeling.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration name
        split: Dataset split (train/validation/test)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_column: Name of the text column in the dataset
        max_samples: Maximum number of samples to use
        streaming: Whether to use streaming
        num_proc: Number of processes for tokenization

    Returns:
        Tokenized dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_config:
        logger.info(f"  Config: {dataset_config}")
    logger.info(f"  Split: {split}")

    # Load dataset
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming
    )

    # Limit samples if specified
    if max_samples is not None:
        if streaming:
            logger.info(f"Limiting streaming dataset to {max_samples} samples")
            dataset = dataset.take(max_samples)
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    elif streaming:
        # CRITICAL: For streaming datasets, we MUST have max_samples
        # Otherwise we'll try to process the entire dataset (e.g., 13T tokens for FineWeb)
        raise ValueError(
            "ERROR: max_samples MUST be specified for streaming datasets!\n"
            "Streaming datasets like FineWeb contain billions of samples.\n"
            "You must explicitly set how many samples to process.\n\n"
            "Example:\n"
            "  distillation.dataset.max_samples=1000000  # Process 1M samples\n\n"
            "Recommended values:\n"
            "  - Testing: 1000-10000\n"
            "  - Small-scale: 100000-1000000\n"
            "  - Large-scale: 1000000-10000000\n"
        )

    # Set tokenizer's model_max_length to ensure it respects our max_length
    # Some tokenizers (like Qwen) have a default model_max_length that may differ
    original_model_max_length = tokenizer.model_max_length
    if tokenizer.model_max_length != max_length:
        logger.info(f"Tokenizer model_max_length ({tokenizer.model_max_length}) differs from requested max_length ({max_length})")
        logger.info(f"Temporarily setting tokenizer.model_max_length to {max_length}")
        tokenizer.model_max_length = max_length

    # Tokenization function
    def tokenize_function(examples):
        # Tokenize texts with explicit max_length to ensure correct sequence length
        # Use padding='max_length' with explicit max_length parameter
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',  # This should pad to max_length, not model_max_length
            return_tensors=None
        )

        # Verify that all sequences have the correct length
        # This catches issues where the tokenizer ignores our max_length setting
        for idx, input_ids in enumerate(tokenized['input_ids'][:1]):  # Check first sample
            actual_length = len(input_ids)
            if actual_length != max_length:
                logger.error(f"Tokenization error: expected length {max_length}, got {actual_length}")
                logger.error(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
                logger.error(f"Sample text (first 200 chars): {examples[text_column][idx][:200]}")

                # Manually truncate/pad to fix the issue
                logger.warning(f"Manually fixing sequence length from {actual_length} to {max_length}")
                for i in range(len(tokenized['input_ids'])):
                    if len(tokenized['input_ids'][i]) > max_length:
                        # Truncate
                        tokenized['input_ids'][i] = tokenized['input_ids'][i][:max_length]
                        tokenized['attention_mask'][i] = tokenized['attention_mask'][i][:max_length]
                    elif len(tokenized['input_ids'][i]) < max_length:
                        # Pad
                        pad_length = max_length - len(tokenized['input_ids'][i])
                        tokenized['input_ids'][i] = tokenized['input_ids'][i] + [tokenizer.pad_token_id] * pad_length
                        tokenized['attention_mask'][i] = tokenized['attention_mask'][i] + [0] * pad_length
                break

        # Log a warning if we encounter empty or very short texts
        # This can happen with wikitext where some samples are just headers or empty lines
        for text in examples[text_column][:3]:  # Check first 3 in batch
            if len(text.strip()) < 50:
                logger.warning(f"Short text detected (len={len(text.strip())}): '{text[:100]}'")

        return tokenized

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc if not streaming else None,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Restore original tokenizer model_max_length
    if tokenizer.model_max_length != original_model_max_length:
        logger.info(f"Restoring tokenizer.model_max_length to {original_model_max_length}")
        tokenizer.model_max_length = original_model_max_length

    if not streaming:
        logger.info(f"Dataset prepared: {len(tokenized_dataset)} samples")
    else:
        logger.info(f"Dataset prepared (streaming mode)")
    return tokenized_dataset


def generate_teacher_logits(
    teacher_model_path: str,
    dataset,  # Dataset or IterableDataset
    cache_dir: str,
    top_k: int = 64,
    batch_size: int = 4,
    force_regenerate: bool = False,
    worker_id: int = 0,
    total_workers: int = 1,
    streaming: bool = False,
    streaming_chunk_size: int = 1000
) -> Path:
    """
    Generate and cache teacher model logits.

    Args:
        teacher_model_path: Path or name of teacher model
        dataset: Tokenized dataset (Dataset or IterableDataset)
        cache_dir: Directory to cache logits
        top_k: Number of top logits to cache
        batch_size: Batch size for teacher inference
        force_regenerate: Whether to force regeneration
        worker_id: Worker ID for multi-worker runs
        total_workers: Total number of workers
        streaming: Whether dataset is streaming
        streaming_chunk_size: Chunk size for streaming datasets

    Returns:
        Path to cached logits file
    """
    logger.info("Loading teacher model...")

    # Always use device_map="auto" for simplicity and flexibility
    # HuggingFace automatically handles:
    # - Single GPU: puts entire model on that GPU
    # - Multiple GPUs: shards model across available GPUs
    # - Multi-worker: each worker only sees GPUs in CUDA_VISIBLE_DEVICES
    logger.info("Loading teacher model with device_map='auto'")
    logger.info(f"Worker {worker_id}/{total_workers}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Always auto - let HuggingFace handle device assignment
        trust_remote_code=True
    )

    # Create logits generator
    generator = TeacherLogitsGenerator(
        teacher_model=teacher_model,
        cache_dir=cache_dir,
        top_k=top_k,
    )

    # Determine max_length
    if streaming:
        # For streaming datasets, we need to get it from the first sample
        sample = next(iter(dataset))
        max_length = len(sample["input_ids"])
    else:
        # Get first sample's input_ids - could be list or tensor depending on dataset format
        first_input_ids = dataset[0]['input_ids']

        # Handle both list and tensor formats
        if isinstance(first_input_ids, list):
            max_length = len(first_input_ids)
            unique_tokens = len(set(first_input_ids))
        else:
            max_length = first_input_ids.shape[0]
            unique_tokens = len(torch.unique(first_input_ids))

        logger.info(f"First sample analysis:")
        logger.info(f"  - Length: {max_length}")
        logger.info(f"  - Unique tokens: {unique_tokens}")
        logger.info(f"  - First 20 tokens: {first_input_ids[:20] if len(first_input_ids) > 20 else first_input_ids}")
        logger.info(f"  - Last 20 tokens: {first_input_ids[-20:] if len(first_input_ids) > 20 else first_input_ids}")

        if unique_tokens < 10:
            logger.warning(f"WARNING: Very low token diversity ({unique_tokens} unique tokens)!")
            logger.warning("This may indicate an issue with tokenization or the dataset.")
            logger.warning("Common causes:")
            logger.warning("  - Empty or very short text samples")
            logger.warning("  - Padding tokens filling most of the sequence")
            logger.warning("  - Incorrect text column name")

        logger.info(f"Determined max_length: {max_length}")

    # Generate and cache logits
    cache_file = generator.generate_and_cache_logits(
        dataset=dataset,
        batch_size=batch_size,
        max_length=max_length,
        force_regenerate=force_regenerate,
        streaming=streaming,
        streaming_chunk_size=streaming_chunk_size,
        worker_id=worker_id,
        total_workers=total_workers
    )

    # Clean up teacher model to free memory
    del teacher_model
    torch.cuda.empty_cache()

    return cache_file


def load_student_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
) -> AutoModelForCausalLM:
    """
    Load student (compressed) model.

    Args:
        model_path: Path to compressed model checkpoint
        device_map: Device map for model parallelism
        torch_dtype: Torch dtype for model

    Returns:
        Loaded student model
    """
    logger.info(f"Loading student model from: {model_path}")

    # Try to infer model class
    try:
        # Check if it's a custom model
        from src.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load as custom model: {e}")
        # Fall back to AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )

    logger.info("Student model loaded successfully")
    return model


def create_training_args(
    config: DictConfig,
    output_dir: str
) -> DistillationTrainingArguments:
    """
    Create training arguments from config.

    Args:
        config: Distillation configuration
        output_dir: Output directory for checkpoints

    Returns:
        DistillationTrainingArguments instance
    """
    training_config = config.training
    loss_config = config.loss

    args = DistillationTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=config.evaluation.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        dataloader_num_workers=training_config.dataloader_num_workers,
        remove_unused_columns=training_config.remove_unused_columns,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        max_grad_norm=training_config.max_grad_norm,
        lr_scheduler_type=training_config.lr_scheduler_type,
        eval_steps=config.evaluation.eval_steps,
        # Distillation-specific
        distillation_alpha=loss_config.alpha,
        distillation_temperature=loss_config.temperature,
        kl_reduction=loss_config.kl_reduction,
    )

    return args


def run_distillation_training(
    config: DictConfig,
    student_model: AutoModelForCausalLM,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> DistillationTrainer:
    """
    Run knowledge distillation training.

    Args:
        config: Distillation configuration
        student_model: Student model to train
        train_dataset: Training dataset (with cached logits)
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer for data collation

    Returns:
        Trained DistillationTrainer
    """
    # Create training arguments
    training_args = create_training_args(
        config=config,
        output_dir=config.training.output_dir
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    ) if tokenizer else None

    # Get learning rate multipliers
    lr_multipliers = config.parameter_lr_multipliers
    default_lr_multiplier = config.get('default_lr_multiplier', 1.0)

    # Create trainer
    trainer = create_distillation_trainer(
        student_model=student_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        lr_multipliers=lr_multipliers,
        default_lr_multiplier=default_lr_multiplier,
        data_collator=data_collator
    )

    # Multi-stage training if enabled
    if config.stages.enabled:
        logger.info("Multi-stage training enabled")
        for stage_idx, stage_config in enumerate(config.stages.configs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting Stage {stage_idx + 1}: {stage_config.stage_name}")
            logger.info(f"{'='*80}\n")

            # Update trainer for this stage
            trainer.update_stage(stage_config)

            # Update number of epochs for this stage
            if 'num_epochs' in stage_config:
                trainer.args.num_train_epochs = stage_config['num_epochs']

            # Train for this stage
            trainer.train()

            logger.info(f"\nCompleted Stage {stage_idx + 1}\n")

    else:
        # Single-stage training
        logger.info("Starting training...")
        trainer.train()

    logger.info("Training complete!")
    return trainer


def setup_distillation_pipeline(
    config: DictConfig,
    teacher_model_path: Optional[str] = None,
    student_model_path: Optional[str] = None,
    force_regenerate_logits: bool = False
) -> Tuple[AutoModelForCausalLM, Dataset, Optional[Dataset], AutoTokenizer]:
    """
    Complete setup for distillation training pipeline.

    This is a convenience function that handles:
    1. Loading teacher and generating cached logits
    2. Loading student model
    3. Preparing datasets

    Args:
        config: Distillation configuration
        teacher_model_path: Path to teacher model (uses config if None)
        student_model_path: Path to student model (uses config if None)
        force_regenerate_logits: Whether to regenerate teacher logits

    Returns:
        Tuple of (student_model, train_dataset, eval_dataset, tokenizer)
    """
    # Resolve paths
    teacher_path = teacher_model_path or config.teacher.model_path or config.model.name
    student_path = student_model_path or config.student.model_path

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_path,
        trust_remote_code=True
    )

    # Prepare datasets
    logger.info("Preparing training dataset...")
    train_dataset = prepare_dataset(
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.config_name,
        split=config.dataset.split,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        text_column=config.dataset.text_column,
        max_samples=config.dataset.max_samples,
        streaming=config.dataset.streaming
    )

    eval_dataset = None
    if config.evaluation.eval_dataset.name:
        logger.info("Preparing evaluation dataset...")
        eval_dataset = prepare_dataset(
            dataset_name=config.evaluation.eval_dataset.name,
            dataset_config=config.evaluation.eval_dataset.config_name,
            split=config.evaluation.eval_dataset.split,
            tokenizer=tokenizer,
            max_length=config.dataset.max_length,
            text_column=config.dataset.text_column,
            max_samples=config.evaluation.eval_dataset.max_samples,
            streaming=False
        )

    # Generate/load teacher logits if using cached approach
    if config.teacher.use_cached_logits:
        logger.info("Setting up teacher logits caching...")
        cache_file = generate_teacher_logits(
            teacher_model_path=teacher_path,
            dataset=train_dataset,
            cache_dir=config.teacher.cache_dir,
            top_k=config.teacher.top_k,
            batch_size=config.teacher.batch_size,
            force_regenerate=force_regenerate_logits
        )

        # Wrap dataset with cached logits
        logger.info("Creating dataset with cached logits...")
        train_dataset = CachedLogitsDataset(
            cache_file=cache_file,
            original_dataset=train_dataset
        )

        if eval_dataset:
            # For eval, you might want to generate separate cached logits
            # or use the same approach
            logger.warning("Evaluation dataset doesn't have cached logits")

    # Load student model
    student_model = load_student_model(
        model_path=student_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.training.bf16 else torch.float16
    )

    return student_model, train_dataset, eval_dataset, tokenizer
