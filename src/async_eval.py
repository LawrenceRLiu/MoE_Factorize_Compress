"""
Asynchronous Checkpoint Evaluation

Monitors checkpoint directory and evaluates new checkpoints using lm_eval harness.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import wandb
import subprocess


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for async evaluation."""
    checkpoint_dir: str
    eval_tasks: List[str]
    gpu_ids: List[int]  # GPUs to use for evaluation
    eval_batch_size: int = 8
    eval_interval: int = 60  # Seconds between checks
    wandb_project: str = "moe-compression"
    wandb_run_name: Optional[str] = None
    num_fewshot: int = 0
    limit: Optional[int] = None  # Limit samples per task (for testing)


class CheckpointEvaluator:
    """
    Evaluates model checkpoints using lm-evaluation-harness.

    Monitors a directory for new checkpoints and runs evaluation tasks.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.evaluated_checkpoints: Set[str] = set()
        self.results_file = self.checkpoint_dir / "evaluation_results.json"
        self.evaluated_file = self.checkpoint_dir / ".evaluated_checkpoints"

        # Load previously evaluated checkpoints
        self._load_evaluated_state()

        # Initialize wandb if specified
        if config.wandb_run_name:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config)
            )

    def _load_evaluated_state(self):
        """Load the set of already evaluated checkpoints."""
        if self.evaluated_file.exists():
            with open(self.evaluated_file, 'r') as f:
                self.evaluated_checkpoints = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.evaluated_checkpoints)} previously evaluated checkpoints")

    def _mark_evaluated(self, checkpoint_name: str):
        """Mark a checkpoint as evaluated."""
        self.evaluated_checkpoints.add(checkpoint_name)
        with open(self.evaluated_file, 'a') as f:
            f.write(f"{checkpoint_name}\n")

    def find_new_checkpoints(self) -> List[Path]:
        """
        Find new checkpoints that haven't been evaluated yet.

        Returns:
            List of checkpoint directories to evaluate
        """
        if not self.checkpoint_dir.exists():
            return []

        # Look for checkpoint-* directories
        checkpoint_dirs = sorted(
            [d for d in self.checkpoint_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1])
        )

        # Filter out already evaluated
        new_checkpoints = [
            d for d in checkpoint_dirs
            if d.name not in self.evaluated_checkpoints
        ]

        return new_checkpoints

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Evaluate a single checkpoint using lm-evaluation-harness.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Build lm_eval command
        # Using the lm_eval CLI tool from the lm-evaluation-harness library
        device_str = ",".join(str(gpu) for gpu in self.config.gpu_ids)

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16,trust_remote_code=True",
            "--tasks", ",".join(self.config.eval_tasks),
            "--device", f"cuda:{self.config.gpu_ids[0]}",  # Primary device
            "--batch_size", str(self.config.eval_batch_size),
            "--num_fewshot", str(self.config.num_fewshot),
            "--output_path", str(checkpoint_path / "eval_results"),
        ]

        if len(self.config.gpu_ids) > 1:
            cmd.extend(["--device", f"cuda:{device_str}"])

        if self.config.limit:
            cmd.extend(["--limit", str(self.config.limit)])

        # Run evaluation
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse results
            results_file = checkpoint_path / "eval_results" / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    eval_results = json.load(f)

                # Extract key metrics
                metrics = self._extract_metrics(eval_results)
                logger.info(f"Evaluation complete. Metrics: {metrics}")

                return {
                    "checkpoint": checkpoint_path.name,
                    "metrics": metrics,
                    "full_results": eval_results
                }
            else:
                logger.error(f"Results file not found: {results_file}")
                return {"checkpoint": checkpoint_path.name, "error": "Results file not found"}

        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return {"checkpoint": checkpoint_path.name, "error": str(e)}

    def _extract_metrics(self, results: Dict) -> Dict:
        """
        Extract key metrics from lm_eval results.

        Args:
            results: Full evaluation results from lm_eval

        Returns:
            Dictionary of key metrics
        """
        metrics = {}

        # lm_eval results structure: {"results": {"task_name": {"metric": value, ...}, ...}}
        if "results" in results:
            for task, task_results in results["results"].items():
                # Common metrics to extract
                for metric in ["acc", "acc_norm", "exact_match", "word_perplexity", "byte_perplexity"]:
                    if metric in task_results:
                        metrics[f"{task}_{metric}"] = task_results[metric]

        return metrics

    def log_results(self, results: Dict):
        """
        Log evaluation results to file and wandb.

        Args:
            results: Evaluation results dictionary
        """
        # Save to JSON file
        all_results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                all_results = json.load(f)

        all_results.append(results)

        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Log to wandb
        if self.config.wandb_run_name and "metrics" in results:
            wandb.log({
                "checkpoint": results["checkpoint"],
                **results["metrics"]
            })

    def run(self):
        """
        Main evaluation loop.

        Continuously monitors checkpoint directory and evaluates new checkpoints.
        """
        logger.info("Starting async evaluation loop")
        logger.info(f"Monitoring directory: {self.checkpoint_dir}")
        logger.info(f"Eval tasks: {self.config.eval_tasks}")
        logger.info(f"Check interval: {self.config.eval_interval}s")

        # Optionally evaluate the original model first as baseline
        # This would require knowing the original model path
        # For now, we'll just monitor for checkpoints

        try:
            while True:
                # Find new checkpoints
                new_checkpoints = self.find_new_checkpoints()

                if new_checkpoints:
                    logger.info(f"Found {len(new_checkpoints)} new checkpoint(s) to evaluate")

                    for checkpoint_path in new_checkpoints:
                        # Evaluate
                        results = self.evaluate_checkpoint(checkpoint_path)

                        # Log results
                        self.log_results(results)

                        # Mark as evaluated
                        self._mark_evaluated(checkpoint_path.name)

                else:
                    logger.debug("No new checkpoints found")

                # Sleep before next check
                logger.debug(f"Sleeping for {self.config.eval_interval}s")
                time.sleep(self.config.eval_interval)

        except KeyboardInterrupt:
            logger.info("Evaluation loop interrupted by user")
        except Exception as e:
            logger.error(f"Evaluation loop error: {e}", exc_info=True)
        finally:
            if self.config.wandb_run_name:
                wandb.finish()


def evaluate_baseline(
    model_name: str,
    eval_tasks: List[str],
    output_dir: str,
    gpu_ids: List[int],
    batch_size: int = 8,
    num_fewshot: int = 0
) -> Dict:
    """
    Evaluate the baseline (original) model.

    Args:
        model_name: HuggingFace model name
        eval_tasks: List of evaluation tasks
        output_dir: Directory to save results
        gpu_ids: GPUs to use
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples

    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating baseline model: {model_name}")

    output_path = Path(output_dir) / "baseline_eval"
    output_path.mkdir(parents=True, exist_ok=True)

    device_str = ",".join(str(gpu) for gpu in gpu_ids)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},dtype=bfloat16,trust_remote_code=True",
        "--tasks", ",".join(eval_tasks),
        "--device", f"cuda:{gpu_ids[0]}",
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_path),
    ]

    if len(gpu_ids) > 1:
        cmd.extend(["--device", f"cuda:{device_str}"])

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        results_file = output_path / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            logger.error("Results file not found")
            return {}

    except subprocess.CalledProcessError as e:
        logger.error(f"Baseline evaluation failed: {e}")
        return {}
