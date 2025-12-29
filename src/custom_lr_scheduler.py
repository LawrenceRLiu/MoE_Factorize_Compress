"""
Custom LR Scheduler for Parameter-Specific Learning Rates

This module provides a custom LR scheduler that wraps any PyTorch/HuggingFace LR scheduler
and applies parameter-specific multipliers on top of the base schedule.

This is a cleaner approach than using callbacks, as it integrates directly with the
optimizer and doesn't have to fight with HuggingFace Trainer's callback ordering.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class ParamGroupPattern:
    """A regex pattern with associated learning rate multiplier."""
    pattern: str
    lr_multiplier: float
    _compiled_pattern: Optional[re.Pattern] = None

    def __post_init__(self):
        """Compile the regex pattern on initialization."""
        self._compiled_pattern = re.compile(self.pattern)

    def matches(self, param_name: str) -> bool:
        """Check if parameter name matches this pattern."""
        return self._compiled_pattern.search(param_name) is not None


@dataclass
class LRStageConfig:
    """Configuration for one stage of learning rate scheduling."""
    fraction: float  # Fraction of total steps for this stage
    base_lr_multiplier: float  # Default multiplier for all parameters
    param_patterns: List[ParamGroupPattern]  # Specific patterns with custom multipliers

    def get_lr_multiplier(self, param_name: str) -> float:
        """
        Get the learning rate multiplier for a given parameter.

        Args:
            param_name: Full parameter name (e.g., "model.layers.0.mlp.experts.core")

        Returns:
            Learning rate multiplier for this parameter
        """
        # Check patterns in order - first match wins
        for pattern in self.param_patterns:
            if pattern.matches(param_name):
                return pattern.lr_multiplier

        # Fall back to base multiplier
        return self.base_lr_multiplier


class ParameterGroupLRScheduler(LRScheduler):
    """
    LR Scheduler that applies parameter-specific multipliers to a base schedule.

    This scheduler wraps any standard PyTorch/HuggingFace LR scheduler and applies
    per-parameter multipliers on top of the base learning rate schedule.

    It supports multiple stages during training where different multipliers can be active.

    Args:
        optimizer: The optimizer to schedule
        base_scheduler: The underlying LR scheduler (e.g., cosine with warmup)
        stages: List of stage configurations
        total_steps: Total number of training steps
        param_names: Mapping from parameter ID to parameter name

    Example:
        >>> from transformers import get_cosine_schedule_with_warmup
        >>>
        >>> # Create base scheduler
        >>> base_scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=100, num_training_steps=1000
        ... )
        >>>
        >>> # Define stages
        >>> stages = [
        ...     LRStageConfig(
        ...         fraction=0.5,
        ...         base_lr_multiplier=0.0,  # Freeze most params
        ...         param_patterns=[
        ...             ParamGroupPattern(pattern=r".*\.wrapper\..*", lr_multiplier=1.0),
        ...         ]
        ...     ),
        ...     LRStageConfig(
        ...         fraction=0.5,
        ...         base_lr_multiplier=0.1,  # Unfreeze at 10%
        ...         param_patterns=[
        ...             ParamGroupPattern(pattern=r".*\.wrapper\..*", lr_multiplier=1.0),
        ...         ]
        ...     ),
        ... ]
        >>>
        >>> # Create custom scheduler
        >>> scheduler = ParameterGroupLRScheduler(
        ...     optimizer, base_scheduler, stages, total_steps=1000, param_names=param_names
        ... )
        >>>
        >>> # Use like any LR scheduler
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # This applies both base schedule and multipliers
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: LRScheduler,
        stages: List[LRStageConfig],
        total_steps: int,
        param_names: Dict[int, str],
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.stages = stages
        self.total_steps = total_steps
        self.param_names = param_names
        self._step_count = 0

        # Calculate step boundaries for each stage
        self.stage_boundaries = []
        current_step = 0
        for stage in stages:
            steps_in_stage = int(stage.fraction * total_steps)
            end_step = current_step + steps_in_stage
            self.stage_boundaries.append({
                'start_step': current_step,
                'end_step': end_step,
                'config': stage
            })
            current_step = end_step

        # Adjust last stage to exactly match total_steps (handle rounding)
        if self.stage_boundaries:
            self.stage_boundaries[-1]['end_step'] = total_steps

        # Store the base learning rates (before any multipliers)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Don't call super().__init__() - we're wrapping another scheduler
        # Initialize last_epoch
        self.last_epoch = -1

        logger.info("=" * 80)
        logger.info("ParameterGroupLRScheduler initialized")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Number of stages: {len(stages)}")
        for i, boundary in enumerate(self.stage_boundaries):
            logger.info(
                f"  Stage {i}: steps {boundary['start_step']}-{boundary['end_step']} "
                f"(base_multiplier={boundary['config'].base_lr_multiplier})"
            )
        logger.info("=" * 80)

    def get_current_stage(self, step: int) -> tuple[int, Dict]:
        """Get the current stage configuration for a given step."""
        for i, boundary in enumerate(self.stage_boundaries):
            if boundary['start_step'] <= step < boundary['end_step']:
                return i, boundary

        # If we're past the end, return the last stage
        return len(self.stage_boundaries) - 1, self.stage_boundaries[-1]

    def step(self, epoch=None):
        """
        Step the scheduler.

        This first steps the base scheduler to get the new base learning rates,
        then applies our parameter-specific multipliers.
        """
        # Step the base scheduler first
        self.base_scheduler.step(epoch)

        # Get current step
        self._step_count += 1
        current_step = self._step_count

        # Get current stage
        stage_idx, boundary = self.get_current_stage(current_step)
        stage_config = boundary['config']

        # Apply multipliers to each param group
        for param_group in self.optimizer.param_groups:
            # Get the base LR that was just set by the base scheduler
            base_lr = param_group['lr']
            if isinstance(base_lr, torch.Tensor):
                base_lr = base_lr.item()

            # Get parameter name for this group
            # HF Trainer creates one param per group, so we can use the first param
            if len(param_group['params']) > 0:
                first_param = param_group['params'][0]
                param_name = self.param_names.get(id(first_param), "unknown")

                # Get multiplier for this parameter
                lr_multiplier = stage_config.get_lr_multiplier(param_name)
                new_lr = base_lr * lr_multiplier

                # Set the new LR
                if isinstance(param_group['lr'], torch.Tensor):
                    param_group['lr'].fill_(new_lr)
                else:
                    param_group['lr'] = new_lr

    def get_last_lr(self):
        """
        Return the last computed learning rates.

        This is called by HF Trainer for logging.
        """
        return [group['lr'] if not isinstance(group['lr'], torch.Tensor)
                else group['lr'].item()
                for group in self.optimizer.param_groups]

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            '_step_count': self._step_count,
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self._step_count = state_dict['_step_count']
        self.last_epoch = state_dict.get('last_epoch', -1)


def create_parameter_group_scheduler(
    optimizer: Optimizer,
    base_scheduler: LRScheduler,
    lr_schedule_config: List[Dict[str, Any]],
    total_steps: int,
    model: torch.nn.Module,
) -> Optional[ParameterGroupLRScheduler]:
    """
    Factory function to create a ParameterGroupLRScheduler from config.

    Args:
        optimizer: The optimizer
        base_scheduler: The base LR scheduler (e.g., from get_cosine_schedule_with_warmup)
        lr_schedule_config: List of stage configurations from Hydra
        total_steps: Total number of training steps
        model: The model (used to get parameter names)

    Returns:
        ParameterGroupLRScheduler instance, or None if no config provided
    """
    if not lr_schedule_config:
        logger.info("No lr_schedule config provided - using base scheduler only")
        return None

    # Build param name cache
    param_names = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names[id(param)] = name

    # Convert config dicts to dataclass instances
    stages = []
    for stage_dict in lr_schedule_config:
        # Parse param patterns
        patterns = []
        for pattern_dict in stage_dict.get('param_patterns', []):
            patterns.append(ParamGroupPattern(
                pattern=pattern_dict['pattern'],
                lr_multiplier=pattern_dict['lr_multiplier']
            ))

        # Create stage config
        stage = LRStageConfig(
            fraction=stage_dict['fraction'],
            base_lr_multiplier=stage_dict['base_lr_multiplier'],
            param_patterns=patterns
        )
        stages.append(stage)

    return ParameterGroupLRScheduler(
        optimizer=optimizer,
        base_scheduler=base_scheduler,
        stages=stages,
        total_steps=total_steps,
        param_names=param_names,
    )
