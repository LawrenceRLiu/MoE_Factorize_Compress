"""
Knowledge Distillation for Compressed MoE Models

Implements teacher-student training with KL divergence loss to recover
performance after compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from datasets import load_dataset
from typing import Dict, Optional, Any
import logging
from dataclasses import dataclass
import wandb
from pathlib import Path

from .compressed_moe_model import load_compressed_model, save_compressed_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressedModelSaveCallback(TrainerCallback):
    """
    Callback to properly save compressed models during training.

    Since we have a custom architecture, we can't use the default save method.
    This callback overrides the default saving behavior.
    """

    def on_save(self, args, state, control, **kwargs):
        """Save compressed model using custom save function."""
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')

        if model is not None:
            output_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            logger.info(f"Saving compressed model to {output_dir}")
            save_compressed_model(model, str(output_dir), tokenizer)

            # Prevent default save behavior
            control.should_save = False

        return control


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_model: str
    student_model_path: str
    output_dir: str
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_split: str = "train"
    max_length: int = 2048
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs. task loss
    use_teacher_quantization: bool = True  # Use 8-bit for teacher to save VRAM
    teacher_load_in_8bit: bool = True
    teacher_load_in_4bit: bool = False
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    bf16: bool = True
    wandb_project: str = "moe-compression"
    wandb_run_name: Optional[str] = None


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Implements KL divergence loss between teacher and student logits.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        *args,
        **kwargs
    ):
        """
        Args:
            teacher_model: The teacher model (frozen)
            temperature: Temperature for distillation
            alpha: Weight for distillation loss (1-alpha for task loss)
            *args, **kwargs: Arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Always in eval mode
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute distillation loss.

        Loss = alpha * KL(teacher || student) + (1 - alpha) * CE(student, labels)

        Args:
            model: Student model
            inputs: Batch inputs
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Get labels
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs["input_ids"].clone()

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Ensure same shape (sometimes last token is dropped)
        min_length = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_length, :]
        teacher_logits = teacher_logits[:, :min_length, :]
        labels = labels[:, :min_length]

        # Distillation loss: KL divergence
        # KL(P || Q) = sum(P * (log(P) - log(Q)))
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Task loss: Standard cross-entropy
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100
        )

        # Combined loss
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "kl_loss": kl_loss.item(),
                "ce_loss": ce_loss.item(),
                "total_loss": loss.item()
            })

        if return_outputs:
            return loss, student_outputs
        return loss


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_length: int = 2048,
    split: str = "train",
    streaming: bool = True,
    num_samples: Optional[int] = None
):
    """
    Prepare dataset for distillation.

    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        split: Dataset split
        streaming: Whether to stream the dataset
        num_samples: Number of samples to use (for testing)

    Returns:
        Processed dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )

    # Limit samples if specified
    if num_samples is not None and streaming:
        dataset = dataset.take(num_samples)

    # Tokenization function
    def tokenize_function(examples):
        # Assuming text field - adjust based on dataset
        text_column = "text" if "text" in examples else list(examples.keys())[0]

        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )

        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize dataset
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

    return dataset


def setup_distillation(config: DistillationConfig):
    """
    Set up teacher, student, and trainer for distillation.

    Args:
        config: Distillation configuration

    Returns:
        Tuple of (trainer, tokenizer)
    """
    # Initialize wandb
    if config.wandb_run_name:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__
        )

    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher model
    logger.info(f"Loading teacher model: {config.teacher_model}")
    teacher_kwargs = {
        "pretrained_model_name_or_path": config.teacher_model,
        "trust_remote_code": True,
    }

    if config.teacher_load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        teacher_kwargs["quantization_config"] = quantization_config
    elif config.teacher_load_in_8bit:
        teacher_kwargs["load_in_8bit"] = True
    else:
        teacher_kwargs["dtype"] = torch.bfloat16

    teacher_model = AutoModelForCausalLM.from_pretrained(**teacher_kwargs)
    teacher_model.eval()

    # Load student model (compressed)
    logger.info(f"Loading compressed student model from: {config.student_model_path}")

    # Check if this is a compressed model directory or original model
    student_path = Path(config.student_model_path)
    if (student_path / "compression_config.json").exists():
        # Load compressed model
        logger.info("Detected compressed model, loading with custom loader...")
        student_model = load_compressed_model(
            compressed_dir=config.student_model_path,
            original_model_name=config.teacher_model,
            device_map="auto",
            dtype=torch.bfloat16
        )
    else:
        # Fallback to standard loading (for testing or non-compressed models)
        logger.warning("No compression_config.json found, loading as standard model")
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )

    # Prepare dataset
    train_dataset = prepare_dataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        max_length=config.max_length,
        split=config.dataset_split,
        streaming=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        bf16=config.bf16,
        report_to="wandb" if config.wandb_run_name else "none",
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # Create distillation trainer with custom save callback
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=config.temperature,
        alpha=config.alpha,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[CompressedModelSaveCallback()]
    )

    return trainer, tokenizer


def run_distillation(config: DistillationConfig):
    """
    Main function to run knowledge distillation.

    Args:
        config: Distillation configuration
    """
    logger.info("Setting up distillation")
    trainer, tokenizer = setup_distillation(config)

    logger.info("Starting distillation training")
    trainer.train()

    logger.info("Saving final model")
    # Use custom save for compressed model
    save_compressed_model(trainer.model, config.output_dir, tokenizer)

    logger.info("Distillation complete!")
