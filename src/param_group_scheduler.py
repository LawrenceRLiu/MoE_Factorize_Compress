"""
Parameter Group Scheduler for Staged Training

Enables different learning rates for different parameter groups during training,
with automatic transitions between stages.

Example:
    Stage 1 (50% of training): Train only wrappers and core (freeze everything else)
    Stage 2 (50% of training): Train everything but with different LR scales
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import TrainingArguments
from src.utils import dict_to_str
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


class ParameterGroupScheduler:
    """
    Manages staged learning rate scheduling for different parameter groups.

    Divides training into stages where different parameter groups can have
    different learning rates. Transitions happen automatically based on step count.

    Args:
        stages: List of stage configurations
        total_steps: Total number of training steps
        base_lr: Base learning rate (from TrainingArguments)
        model: The model to schedule parameters for

    Example configuration (as dict for Hydra):
        lr_schedule:
          - fraction: 0.5
            base_lr_multiplier: 0.0  # freeze everything by default
            param_patterns:
              - pattern: ".*\\.core$"
                lr_multiplier: 1.0
              - pattern: ".*\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
                lr_multiplier: 1.0

          - fraction: 0.5
            base_lr_multiplier: 0.1  # 10% LR for everything
            param_patterns:
              - pattern: ".*\\.core$"
                lr_multiplier: 1.0
              - pattern: ".*\\.experts\\.\\d+\\.(input_wrapper|output_wrapper)\\.(U|V)$"
                lr_multiplier: 1.0
    """

    def __init__(
        self,
        stages: List[LRStageConfig],
        total_steps: int,
        base_lr: float,
        model: torch.nn.Module
    ):
        self.stages = stages
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.model = model

        # Validate fractions sum to ~1.0
        total_fraction = sum(stage.fraction for stage in stages)
        if not (0.99 <= total_fraction <= 1.01):
            logger.warning(
                f"Stage fractions sum to {total_fraction:.3f}, expected ~1.0. "
                f"This may cause unexpected behavior."
            )

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

        # Cache parameter names and their group assignments
        self._param_name_cache = {}
        self._build_param_cache()

        logger.info("=" * 80)
        logger.info("ParameterGroupScheduler initialized")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Base LR: {base_lr}")
        logger.info(f"Number of stages: {len(stages)}")
        for i, boundary in enumerate(self.stage_boundaries):
            logger.info(
                f"  Stage {i}: steps {boundary['start_step']}-{boundary['end_step']} "
                f"(base_multiplier={boundary['config'].base_lr_multiplier})"
            )
        logger.info("=" * 80)

    def _build_param_cache(self):
        """Build cache of parameter names for faster lookup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._param_name_cache[id(param)] = name

    def get_current_stage(self, step: int) -> Tuple[int, Dict]:
        """
        Get the current stage configuration for a given step.

        Args:
            step: Current training step

        Returns:
            Tuple of (stage_index, stage_boundary_dict)
        """
        for i, boundary in enumerate(self.stage_boundaries):
            if boundary['start_step'] <= step < boundary['end_step']:
                return i, boundary

        # If we're past the end, return the last stage
        logger.warning(f"Step {step} exceeds total training steps; using last stage")
        return len(self.stage_boundaries) - 1, self.stage_boundaries[-1]

    def update_optimizer_param_groups(self, optimizer: torch.optim.Optimizer, step: int, base_lr: Optional[float] = None) -> Dict[str, float]:
        """
        Update optimizer parameter groups based on current step.

        This method applies parameter-specific LR multipliers to a base learning rate.
        The base LR typically comes from the HuggingFace LR scheduler (e.g., cosine with warmup).

        Args:
            optimizer: The optimizer to update
            step: Current training step
            base_lr: Base learning rate to apply multipliers to. If None, uses self.base_lr.
                     In practice, this should be the LR from the first param group after
                     the HuggingFace scheduler has updated it.

        Returns:
            Dictionary mapping parameter group names to their new learning rates
        """
        stage_idx, boundary = self.get_current_stage(step)
        stage_config = boundary['config']

        # If no base_lr provided, try to infer it from the first param group
        # (which has been updated by HuggingFace's LR scheduler)
        if base_lr is None:
            if len(optimizer.param_groups) > 0:
                first_lr = optimizer.param_groups[0]['lr']
                if isinstance(first_lr, torch.Tensor):
                    base_lr = first_lr.item()
                else:
                    base_lr = first_lr
            else:
                base_lr = self.base_lr

        # Track which parameters get which learning rates for logging
        lr_assignments = {}

        # Update each parameter group
        for param_group in optimizer.param_groups:
            # Get parameter names in this group
            # Note: param_groups may contain individual params or groups of params
            params = param_group['params']

            # For simplicity, we'll set LR based on the first parameter's name
            # (In practice, HuggingFace Trainer typically creates one param per group anyway)
            if len(params) > 0:
                first_param = params[0]
                param_name = self._param_name_cache.get(id(first_param), "unknown")

                # Get LR multiplier for this parameter
                lr_multiplier = stage_config.get_lr_multiplier(param_name)
                new_lr = base_lr * lr_multiplier

                # Update the parameter group
                # Some optimizers (like torchao's AdamW8bit) use tensor LRs
                current_lr = param_group['lr']
                if isinstance(current_lr, torch.Tensor):
                    # For tensor-based LRs, use fill_() as required by torchao
                    param_group['lr'].fill_(new_lr)
                else:
                    # For scalar LRs (standard PyTorch optimizers)
                    param_group['lr'] = new_lr

                lr_assignments[param_name] = new_lr

        return lr_assignments

    def get_param_group_summary(self, step: int) -> Dict[str, Any]:
        """
        Get a summary of current parameter groups and their learning rates.

        Useful for logging and debugging.

        Args:
            step: Current training step

        Returns:
            Dictionary with summary information
        """
        stage_idx, boundary = self.get_current_stage(step)
        stage_config = boundary['config']

        # Group parameters by their learning rate multiplier
        lr_groups = {}
        for name in self._param_name_cache.values():
            lr_mult = stage_config.get_lr_multiplier(name)
            if lr_mult not in lr_groups:
                lr_groups[lr_mult] = []
            lr_groups[lr_mult].append(name)

        return {
            'stage_idx': stage_idx,
            'stage_start_step': boundary['start_step'],
            'stage_end_step': boundary['end_step'],
            'base_lr_multiplier': stage_config.base_lr_multiplier,
            'lr_groups': list(lr_groups.keys()),
            'num_params_by_lr': {lr: len(names) for lr, names in lr_groups.items()},
            #sample the parameters for each lr group
            'sample_params_by_lr': {lr: names[:5] for lr, names in lr_groups.items()}
        }
        
    def get_all_param_group_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summaries of all stages for inspection.

        Returns:
            List of dictionaries with summary information for each stage
        """
        summaries = []
        for i, boundary in enumerate(self.stage_boundaries):
            stage_config = boundary['config']

            # Group parameters by their learning rate multiplier
            lr_groups = {}
            for name in self._param_name_cache.values():
                lr_mult = stage_config.get_lr_multiplier(name)
                if lr_mult not in lr_groups:
                    lr_groups[lr_mult] = []
                lr_groups[lr_mult].append(name)

            summaries.append({
                'stage_idx': i,
                'stage_start_step': boundary['start_step'],
                'stage_end_step': boundary['end_step'],
                'base_lr_multiplier': stage_config.base_lr_multiplier,
                'lr_groups': list(lr_groups.keys()),
                'num_params_by_lr': {lr: len(names) for lr, names in lr_groups.items()},
                #sample the parameters for each lr group
                'sample_params_by_lr': {lr: names[:5] for lr, names in lr_groups.items()}
            })
        return summaries


class ParameterGroupSchedulerCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that applies parameter group scheduling.

    This callback applies per-parameter LR multipliers to work alongside the HuggingFace
    LR scheduler (e.g., cosine with warmup).

    CRITICAL EXECUTION ORDER:
    The HuggingFace Trainer's _inner_training_loop method does this (simplified):
    1. self.callback_handler.on_step_begin(...)
    2. tr_loss_step = self.training_step(model, inputs)
       - Inside training_step: forward, backward, optimizer.step()
    3. self.lr_scheduler.step()  ← LR updated AFTER optimizer step!
    4. self.callback_handler.on_step_end(...)

    This means:
    - The LR scheduler updates the LR for the NEXT step (step N+1)
    - We apply our multipliers in on_step_end AFTER the scheduler has run
    - These multiplied LRs will be used in the next training_step (step N+1)

    Args:
        scheduler: The ParameterGroupScheduler instance
    """

    def __init__(self, scheduler: ParameterGroupScheduler):
        self.scheduler = scheduler
        self.current_stage = -1

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the beginning of training to set initial learning rates.

        This ensures that step 0 has the correct multiplied LRs before the first training_step.
        """
        optimizer = kwargs.get('optimizer')

        if optimizer is None:
            logger.warning("Optimizer not available in on_train_begin")
            return

        # Apply multipliers for step 0
        step = 0

        # IMPORTANT: Use the configured base_lr, not the optimizer's current LR
        # The LR scheduler sets all param groups to 0.0 during warmup initialization
        # But we want to apply multipliers to the actual configured LR
        base_lr = self.scheduler.base_lr

        logger.info("=" * 80)
        logger.info("Setting initial learning rates for step 0")
        logger.info(f"Using configured base LR: {base_lr}")

        # Also log what the optimizer currently has (for debugging)
        if len(optimizer.param_groups) > 0:
            current_lr = optimizer.param_groups[0]['lr']
            if isinstance(current_lr, torch.Tensor):
                current_lr = current_lr.item()
            logger.info(f"Optimizer's current LR (before our multipliers): {current_lr}")

        # Log the initial stage info
        stage_idx, boundary = self.scheduler.get_current_stage(step)
        self.current_stage = stage_idx
        logger.info(f"Starting in STAGE {stage_idx}")
        logger.info(f"  Stage runs from step {boundary['start_step']} to {boundary['end_step']}")
        logger.info(f"  Base LR multiplier: {boundary['config'].base_lr_multiplier}")

        summary = self.scheduler.get_param_group_summary(step)
        logger.info("  Initial Parameter Group Summary:")
        logger.info(f"=" * 40)
        logger.info(f"{dict_to_str(summary)}")
        logger.info("=" * 40)

        # Apply multipliers for step 0
        self.scheduler.update_optimizer_param_groups(optimizer, step, base_lr=base_lr)
        logger.info("Initial learning rates applied")
        logger.info("=" * 80)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step, AFTER lr_scheduler.step() has been called.

        At this point:
        - optimizer.step() has already been called (using LRs from previous on_step_end)
        - lr_scheduler.step() has just set all param groups to the new base LR
        - We now apply our multipliers to prepare for the NEXT training step

        This ensures:
        1. lr_scheduler.step() doesn't overwrite our multipliers
        2. Our multiplied LRs are ready for the next training_step
        3. Logging (which happens in on_log) will see the correct LRs
        """
        optimizer = kwargs.get('optimizer')

        if optimizer is None:
            # Optimizer not created yet
            return

        step = state.global_step

        # Check if we're transitioning to a new stage
        new_stage, boundary = self.scheduler.get_current_stage(step)
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            logger.info("=" * 80)
            logger.info(f"TRANSITIONING TO STAGE {new_stage} at step {step}")
            logger.info(f"  Stage runs from step {boundary['start_step']} to {boundary['end_step']}")
            logger.info(f"  Base LR multiplier: {boundary['config'].base_lr_multiplier}")

            # Log parameter group summary
            summary = self.scheduler.get_param_group_summary(step)
            logger.info("  Parameter Group Summary:")
            logger.info(f"=" * 40)
            logger.info(f"{dict_to_str(summary)}")
            logger.info("=" * 40)
            logger.info("=" * 80)

        # Get the base LR that was just set by lr_scheduler.step()
        # All param groups should have the same LR at this point (before we apply multipliers)
        base_lr = None
        if len(optimizer.param_groups) > 0:
            first_lr = optimizer.param_groups[0]['lr']
            if isinstance(first_lr, torch.Tensor):
                base_lr = first_lr.item()
            else:
                base_lr = first_lr

        # Apply our per-parameter multipliers for the NEXT step
        self.scheduler.update_optimizer_param_groups(optimizer, step, base_lr=base_lr)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging - add parameter group info to logs and fix the reported LR."""
        if logs is None:
            return

        optimizer = kwargs.get('optimizer')
        step = state.global_step
        stage_idx, _ = self.scheduler.get_current_stage(step)

        # Add stage information to logs
        logs['param_schedule/stage'] = stage_idx

        # Add per-group learning rates (sample a few for visibility)
        summary = self.scheduler.get_param_group_summary(step)
        for lr_mult, count in summary['num_params_by_lr'].items():
            logs[f'param_schedule/lr_mult_{lr_mult:.3f}_count'] = count

        # IMPORTANT: Fix the 'learning_rate' that Trainer reports
        # Since we apply multipliers in on_step_end (after lr_scheduler.step()),
        # the optimizer param groups should already have the correct multiplied LRs
        # We report the maximum LR being used (most informative for multi-LR training)
        if optimizer is not None and len(optimizer.param_groups) > 0:
            max_lr = 0.0
            total_lr = 0.0
            total_params = 0

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                if isinstance(lr, torch.Tensor):
                    lr = lr.item()

                num_params = len(param_group['params'])
                max_lr = max(max_lr, lr)
                total_lr += lr * num_params
                total_params += num_params

            # Override the default 'learning_rate' with the max LR
            # Also log the average LR weighted by number of parameters
            logs['learning_rate'] = max_lr
            logs['learning_rate_avg'] = total_lr / total_params if total_params > 0 else 0.0

            # Log individual LRs for different groups (up to 5 for visibility)
            unique_lrs = set()
            for param_group in optimizer.param_groups[:min(5, len(optimizer.param_groups))]:
                lr = param_group['lr']
                if isinstance(lr, torch.Tensor):
                    lr = lr.item()
                unique_lrs.add(lr)

            for i, lr in enumerate(sorted(unique_lrs, reverse=True)):
                logs[f'lr/group_{i}'] = lr


def create_param_group_scheduler(
    lr_schedule_config: List[Dict[str, Any]],
    total_steps: int,
    base_lr: float,
    model: torch.nn.Module
) -> Optional[ParameterGroupScheduler]:
    """
    Factory function to create a ParameterGroupScheduler from Hydra config.

    Args:
        lr_schedule_config: List of stage configurations from Hydra
        total_steps: Total number of training steps
        base_lr: Base learning rate
        model: The model to schedule

    Returns:
        ParameterGroupScheduler instance, or None if lr_schedule_config is empty/None

    Example config:
        lr_schedule:
          - fraction: 0.5
            base_lr_multiplier: 0.0
            param_patterns:
              - pattern: ".*\\.core$"
                lr_multiplier: 1.0
    """
    if not lr_schedule_config:
        logger.info("No lr_schedule config provided - using default single learning rate")
        return None

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

    out = ParameterGroupScheduler(
        stages=stages,
        total_steps=total_steps,
        base_lr=base_lr,
        model=model
    )
    for i, stage_summary in enumerate(out.get_all_param_group_summaries()):
        logger.info("=" * 20 + f" Stage {i} Summary " + "=" * 20)
        logger.info(f"{dict_to_str(stage_summary)}")
        logger.info("=" * 40)
    return out
    
