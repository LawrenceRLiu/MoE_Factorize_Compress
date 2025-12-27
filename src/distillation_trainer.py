"""
Knowledge Distillation Trainer for Compressed MoE Models

Custom HuggingFace Trainer subclass that implements knowledge distillation
with blended loss (cross-entropy + KL divergence) and parameter-specific learning rates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import re
from dataclasses import dataclass

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_utils import PreTrainedModel

from src.teacher_logits import reconstruct_teacher_logits

logger = logging.getLogger(__name__)


@dataclass
class DistillationTrainingArguments(TrainingArguments):
    """
    Extended training arguments for knowledge distillation.

    Adds distillation-specific parameters to the base TrainingArguments.
    """
    # Distillation loss parameters
    distillation_alpha: float = 0.5  # Weight for distillation loss
    distillation_temperature: float = 2.0  # Temperature for softening distributions
    kl_reduction: str = "batchmean"  # Reduction for KL divergence loss

    # Multi-stage training
    enable_stages: bool = False
    current_stage: int = 0


class ParameterGroupManager:
    """
    Manages parameter groups with regex-based learning rate multipliers.

    Allows fine-grained control over which parameters are trained and at what rate.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        lr_multipliers: List[Dict[str, Union[str, float]]],
        weight_decay: float = 0.01,
        default_lr_multiplier: float = 1.0
    ):
        """
        Initialize parameter group manager.

        Args:
            model: PyTorch model
            base_lr: Base learning rate
            lr_multipliers: List of dicts with 'pattern' and 'lr_multiplier' keys
            weight_decay: Weight decay factor
            default_lr_multiplier: Default LR multiplier for parameters that don't match any pattern.
                Set to 0.0 to freeze all non-matching parameters by default.
                Set to 1.0 to train all non-matching parameters at full LR (default).
        """
        self.model = model
        self.base_lr = base_lr
        self.lr_multipliers = lr_multipliers
        self.weight_decay = weight_decay
        self.default_lr_multiplier = default_lr_multiplier

    def create_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates based on regex patterns.

        Returns:
            List of parameter group dictionaries for optimizer
        """
        # Create a mapping of parameter name to its learning rate multiplier
        param_lr_map = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Find matching pattern
            multiplier = self.default_lr_multiplier  # Use configurable default
            matched_pattern = None

            for config in self.lr_multipliers:
                pattern = config['pattern']
                if re.search(pattern, name):
                    multiplier = config['lr_multiplier']
                    matched_pattern = pattern
                    break  # Use first matching pattern

            param_lr_map[name] = {
                'param': param,
                'lr_multiplier': multiplier,
                'pattern': matched_pattern or 'default'
            }

        # Group parameters by learning rate multiplier
        lr_groups = {}
        for name, info in param_lr_map.items():
            multiplier = info['lr_multiplier']
            if multiplier not in lr_groups:
                lr_groups[multiplier] = []
            lr_groups[multiplier].append((name, info['param']))

        # Create parameter groups for optimizer
        parameter_groups = []

        for multiplier, params_list in lr_groups.items():
            lr = self.base_lr * multiplier

            # Separate parameters based on whether they should have weight decay
            # Typically, we don't apply weight decay to biases and layer norms
            decay_params = []
            no_decay_params = []

            for name, param in params_list:
                if any(nd in name for nd in ['bias', 'LayerNorm', 'layernorm', 'norm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            # Add groups (only if lr > 0, otherwise frozen)
            if lr > 0:
                if decay_params:
                    parameter_groups.append({
                        'params': decay_params,
                        'lr': lr,
                        'weight_decay': self.weight_decay
                    })
                if no_decay_params:
                    parameter_groups.append({
                        'params': no_decay_params,
                        'lr': lr,
                        'weight_decay': 0.0
                    })

                # Log parameter group info
                total_params = sum(p.numel() for p in decay_params + no_decay_params)
                logger.info(
                    f"Parameter group: lr_multiplier={multiplier:.2f}, "
                    f"lr={lr:.2e}, params={total_params:,}"
                )
            else:
                # Frozen parameters
                frozen_params = decay_params + no_decay_params
                for param in frozen_params:
                    param.requires_grad = False
                total_params = sum(p.numel() for p in frozen_params)
                logger.info(f"Frozen parameter group: params={total_params:,}")

        return parameter_groups

    def log_parameter_groups(self):
        """Log detailed information about parameter groups."""
        logger.info("=" * 80)
        logger.info("Parameter Group Configuration:")
        logger.info("=" * 80)

        for name, param in self.model.named_parameters():
            # Find matching pattern
            multiplier = 1.0
            matched_pattern = "default"

            for config in self.lr_multipliers:
                pattern = config['pattern']
                if re.search(pattern, name):
                    multiplier = config['lr_multiplier']
                    matched_pattern = pattern
                    break

            lr = self.base_lr * multiplier
            status = "FROZEN" if lr == 0 else "TRAINABLE"

            logger.info(
                f"{status:10s} | {name:60s} | "
                f"LR: {lr:.2e} | Pattern: {matched_pattern}"
            )

        logger.info("=" * 80)


class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation.

    Implements a blended loss function combining:
    1. Cross-entropy loss (standard language modeling)
    2. KL divergence loss (distillation from teacher)

    Loss = (1 - alpha) * CE_loss + alpha * T^2 * KL_loss

    Features:
    - Parameter-specific learning rates via regex matching
    - Support for cached teacher logits (memory efficient)
    - Multi-stage training support
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: DistillationTrainingArguments,
        lr_multipliers: Optional[List[Dict[str, Union[str, float]]]] = None,
        default_lr_multiplier: float = 1.0,
        **kwargs
    ):
        """
        Initialize distillation trainer.

        Args:
            model: Student model to train
            args: Training arguments with distillation parameters
            lr_multipliers: List of regex patterns and LR multipliers
            default_lr_multiplier: Default LR multiplier for non-matching parameters
            **kwargs: Additional arguments passed to Trainer
        """
        super().__init__(model=model, args=args, **kwargs)

        self.distillation_alpha = args.distillation_alpha
        self.temperature = args.distillation_temperature
        self.kl_reduction = args.kl_reduction
        self.lr_multipliers = lr_multipliers or []
        self.default_lr_multiplier = default_lr_multiplier

        logger.info(f"Distillation alpha: {self.distillation_alpha}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"KL reduction: {self.kl_reduction}")
        logger.info(f"Default LR multiplier: {self.default_lr_multiplier}")

    def create_optimizer(self):
        """
        Create optimizer with parameter-specific learning rates.

        Overrides the default optimizer creation to support regex-based
        learning rate multipliers.
        """
        if self.optimizer is not None:
            return self.optimizer

        # Use parameter group manager if lr_multipliers are specified
        if self.lr_multipliers:
            logger.info("Creating optimizer with parameter-specific learning rates...")

            param_manager = ParameterGroupManager(
                model=self.model,
                base_lr=self.args.learning_rate,
                lr_multipliers=self.lr_multipliers,
                weight_decay=self.args.weight_decay,
                default_lr_multiplier=self.default_lr_multiplier
            )

            # Log parameter groups
            param_manager.log_parameter_groups()

            # Create parameter groups
            parameter_groups = param_manager.create_parameter_groups()

            if not parameter_groups:
                raise ValueError("No trainable parameters found!")

            # Create optimizer with parameter groups
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Remove lr and weight_decay from kwargs as we set them per group
            optimizer_kwargs.pop('lr', None)
            optimizer_kwargs.pop('weight_decay', None)

            self.optimizer = optimizer_cls(parameter_groups, **optimizer_kwargs)

        else:
            # Use default optimizer creation
            logger.info("Creating optimizer with default settings...")
            super().create_optimizer()

        return self.optimizer

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute blended distillation loss.

        Loss = (1 - alpha) * CE_loss + alpha * T^2 * KL_loss

        Args:
            model: Student model
            inputs: Input batch with labels and teacher logits
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor (and optionally model outputs)
        """
        # Extract teacher logits if present
        teacher_logits_values = inputs.pop("teacher_logits_values", None)
        teacher_logits_indices = inputs.pop("teacher_logits_indices", None)
        seq_length = inputs.pop("seq_length", None)  # Actual content length (for reference)
        vocab_size = inputs.pop("vocab_size", None)

        # Get attention mask before forward pass (we'll need it for masking losses)
        attention_mask = inputs.get("attention_mask", None)

        # Forward pass through student model
        outputs = model(**inputs)
        student_logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Get labels (for cross-entropy)
        labels = inputs.get("labels")

        # 1. Compute standard cross-entropy loss
        if labels is not None:
            # Shift for language modeling (predict next token)
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute CE loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)

        # 2. Compute KL divergence loss (if teacher logits provided)
        if teacher_logits_values is not None and teacher_logits_indices is not None:
            # Reconstruct full teacher logits from top-k
            teacher_logits_full = reconstruct_teacher_logits(
                values=teacher_logits_values,
                indices=teacher_logits_indices,
                vocab_size=vocab_size[0].item() if vocab_size is not None else student_logits.size(-1)
            )  # (batch_size, seq_len, vocab_size)

            # Move to same device as student logits
            teacher_logits_full = teacher_logits_full.to(student_logits.device)

            # Shift logits for next-token prediction
            shift_student_logits = student_logits[..., :-1, :].contiguous()
            shift_teacher_logits = teacher_logits_full[..., :-1, :].contiguous()

            # Apply temperature scaling
            student_log_probs = F.log_softmax(
                shift_student_logits / self.temperature,
                dim=-1
            )
            teacher_probs = F.softmax(
                shift_teacher_logits / self.temperature,
                dim=-1
            )

            # Compute KL divergence with attention mask
            if attention_mask is not None:
                # Shift attention mask to match shifted logits (for next-token prediction)
                # Original: [batch, seq_len], Shifted: [batch, seq_len-1]
                shift_mask = attention_mask[..., :-1].contiguous()

                # Compute KL divergence per position (no reduction yet)
                # Shape: [batch, seq_len-1, vocab_size]
                kl_div_per_position = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='none',
                    log_target=False
                )

                # Sum over vocabulary dimension to get KL per position
                # Shape: [batch, seq_len-1]
                kl_div_per_token = kl_div_per_position.sum(dim=-1)

                # Apply mask: only compute loss on non-padding positions
                # Shape: [batch, seq_len-1]
                masked_kl_div = kl_div_per_token * shift_mask

                # Compute final loss based on reduction method
                if self.kl_reduction == 'batchmean':
                    # Average over all non-padding tokens
                    num_valid_tokens = shift_mask.sum()
                    kl_loss = masked_kl_div.sum() / num_valid_tokens if num_valid_tokens > 0 else masked_kl_div.sum()
                elif self.kl_reduction == 'mean':
                    # Average over all positions (including padding)
                    kl_loss = masked_kl_div.mean()
                elif self.kl_reduction == 'sum':
                    # Sum over all non-padding positions
                    kl_loss = masked_kl_div.sum()
                else:
                    raise ValueError(f"Unknown kl_reduction: {self.kl_reduction}")
            else:
                # No attention mask - use original behavior
                kl_loss = F.kl_div(
                    student_log_probs.view(-1, student_log_probs.size(-1)),
                    teacher_probs.view(-1, teacher_probs.size(-1)),
                    reduction=self.kl_reduction,
                    log_target=False
                )

            # Scale by temperature^2 (as per distillation literature)
            kl_loss = kl_loss * (self.temperature ** 2)

        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)

        # 3. Blend losses
        total_loss = (1 - self.distillation_alpha) * ce_loss + self.distillation_alpha * kl_loss

        # Log individual loss components (for monitoring)
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}: "
                f"Total Loss: {total_loss.item():.4f}, "
                f"CE Loss: {ce_loss.item():.4f}, "
                f"KL Loss: {kl_loss.item():.4f}"
            )

        return (total_loss, outputs) if return_outputs else total_loss

    def update_stage(self, stage_config: Dict[str, Any]):
        """
        Update training configuration for multi-stage training.

        Args:
            stage_config: Configuration for the new training stage
        """
        logger.info(f"Updating to stage: {stage_config.get('stage_name', 'unnamed')}")

        # Update learning rate multipliers
        if 'parameter_lr_multipliers' in stage_config:
            self.lr_multipliers = stage_config['parameter_lr_multipliers']

            # Recreate optimizer with new parameter groups
            self.optimizer = None
            self.create_optimizer()

            # Recreate scheduler
            self.lr_scheduler = None
            self.create_scheduler(
                num_training_steps=self.args.max_steps,
                optimizer=self.optimizer
            )

        # Update other training parameters if specified
        if 'learning_rate' in stage_config:
            self.args.learning_rate = stage_config['learning_rate']

        if 'num_train_epochs' in stage_config:
            self.args.num_train_epochs = stage_config['num_train_epochs']

        logger.info("Stage update complete")


def create_distillation_trainer(
    student_model: PreTrainedModel,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: Optional[torch.utils.data.Dataset],
    training_args: DistillationTrainingArguments,
    lr_multipliers: Optional[List[Dict[str, Union[str, float]]]] = None,
    default_lr_multiplier: float = 1.0,
    data_collator: Optional[Any] = None,
) -> DistillationTrainer:
    """
    Factory function to create a configured DistillationTrainer.

    Args:
        student_model: Compressed MoE model to train
        train_dataset: Training dataset (with cached teacher logits)
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        lr_multipliers: Parameter-specific learning rate configurations
        default_lr_multiplier: Default LR multiplier for non-matching parameters
        data_collator: Data collator for batching

    Returns:
        Configured DistillationTrainer instance
    """
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lr_multipliers=lr_multipliers,
        default_lr_multiplier=default_lr_multiplier,
        data_collator=data_collator,
    )

    return trainer
