"""
Asynchronous Checkpoint Evaluation

Monitors checkpoint directory and evaluates new checkpoints using lm_eval harness.
"""

import os
from omegaconf import OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Optional, Set, Union
from dataclasses import dataclass, asdict
import wandb
import sys
import subprocess

from .compressed_moe_model import load_compressed_model, export_to_hf_format


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_single_task(
    checkpoint_path: Union[str, Path],
    task_name: str,
    num_fewshot: int,
    save_path: Union[str, Path],
    gpu_ids: List[int],
    n_gpus_per_model: int = 1,
    batch_size: Union[int, str] = "auto",
):
    """
    Evaluate a single task using lm-evaluation-harness.
    
    Saves the results to the specified path for later analysis.

    Args:
        checkpoint_path: Path to model checkpoint (HuggingFace format)
        task_name: Name of the evaluation task
        num_fewshot: Number of few-shot examples, -1 for perplexity evaluation
        save_path: Path to save evaluation results
        gpu_ids: List of GPU IDs to use
        n_gpus_per_model: Number of GPUs to allocate per model, 
        if this is less than the number of gpus, we will parallelize across multiple gpus
        batch_size: Batch size for evaluation ("auto" or int)
    """
    
    #checking to see if the the number of gpus required to run the model is less than the number of gpus available
    assert n_gpus_per_model <= len(gpu_ids), \
        "Number of gpus per model cannot be greater than the number of gpus available"
    
    #set the environment variable to use the specified gpus
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids[:(len(gpu_ids)//n_gpus_per_model)*n_gpus_per_model])
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16,trust_remote_code=True,parallelize={n_gpus_per_model > 1}",
        "--tasks", task_name,
        "--output_path", str(save_path),
    ]
    if num_fewshot >= 0:
        cmd += ["--num_fewshot", str(num_fewshot)]
    else:
        logger.info("Using perplexity evaluation (num_fewshot < 0)")
    
    if len(gpu_ids)//n_gpus_per_model > 1:
        #we parallelize across multiple gpus
        prefix = [
             sys.executable,
             "-m", 
             "accelerate.commands.launch",
             "--multi_gpu",
             "--num_processes", str(len(gpu_ids)//n_gpus_per_model),
             "-m",
        ]
    else:
        prefix = [sys.executable, "-m"]
    cmd = prefix + cmd
    if batch_size != "auto":
        cmd += ["--batch_size", str(batch_size)]
        
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
        
        


@dataclass
class EvalConfig:
    """Configuration for async evaluation."""
    checkpoint_dir: str
    temp_dir: str
    eval_dir: str
    eval_tasks: List[str]
    fewshots: List[int]
    config_dir: Optional[str] = None
    batch_size: Union[int, str] = "auto"  # Batch size for evaluation
    gpu_ids: List[int] = None  # GPUs to use for evaluation
    n_gpus_per_model: int = 1  # GPUs per model for parallelization
    eval_interval: int = 60  # Seconds between checks
    
    def __post_init__(self):
        # Ensure directories exist
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.temp_dir = Path(self.temp_dir)
        self.eval_dir = Path(self.eval_dir)
        
        if self.config_dir:
            self.config_dir = Path(self.config_dir)
        else:
            self.config_dir = self.checkpoint_dir.parent / "compression_config.yaml"
            
        with open(self.config_dir, 'r') as f:
            comp_config = OmegaConf.load(f)
        self.original_model_name = comp_config.model_name
        
        


class Evaluator:
    """
    Evaluates model checkpoints using lm-evaluation-harness.

    Monitors a directory for new checkpoints and runs evaluation tasks.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        #TODO:: Wandb integration
        
    def _evaluate(self, model_path: Optional[Path] = None) -> Dict:
        
        if model_path is None:
            #then we evaluate the original model
            model_path = self.config.original_model_name
            results_path = self.config.eval_dir / "baseline_eval"
        else:
            #we expect the model_path to be a path to the temporary full hf model
            #replace the temp dir with eval dir
            relative_path = model_path.relative_to(self.config.temp_dir) #something along the lines of checkpoint-0
            results_path = self.config.eval_dir / relative_path / "eval_results"
            
            #TODO: Check this at a later point
            
        results_path.mkdir(parents=True, exist_ok=True)
        out = {}
        
        for task, fewshot in zip(self.config.eval_tasks, self.config.fewshots):
            logger.info(f"Evaluating task: {task} with {fewshot} few-shots")
            evaluate_single_task(
                checkpoint_path=model_path,
                task_name=task,
                num_fewshot=fewshot,
                save_path=results_path / "{task}_fewshot_{fewshot}.json",
                gpu_ids=self.config.gpu_ids,
                n_gpus_per_model=self.config.n_gpus_per_model,
                batch_size=self.config.batch_size
            )
            #load the results
            with open(results_path / f"{task}_fewshot_{fewshot}.json", 'r') as f:
                task_results = json.load(f)
            out[f"{task}_fewshot_{fewshot}"] = task_results
            
            #TODO: log to wandb if needed
            
        logger.info(f"Completed evaluation for model: {model_path}")
        logger.info(f"Results: {out}")
        return out
    
    def evaluate_baseline(self) -> Dict:
        """
        Evaluate the baseline (original) model.

        Returns:
            Evaluation results
        """
        logger.info("Evaluating baseline model")
        return self._evaluate(model_path=None)
    
    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory, expected to be of the form 
            {checkpoint_dir}/checkpoint-*

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Check if this is a compressed model (has compression_config.json)
        eval_path = checkpoint_path
        if (checkpoint_path / "compression_config.json").exists():
            logger.info("Detected compressed model, exporting to HF format for evaluation...")

            export_path = self.config.temp_dir / checkpoint_path.name / "exported_for_eval"
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Only export if not already done
            if not export_path.exists():
                try:
                    export_to_hf_format(
                        compressed_model_path=str(checkpoint_path),
                        original_model_name=self.config.original_model_name,
                        output_path=str(export_path),
                        device_map=f"cuda:{self.config.gpu_ids[0]}",  # Use eval GPU
                        torch_dtype=torch.bfloat16
                    )
                    eval_path = export_path
                    logger.info(f"Export complete, will evaluate: {eval_path}")
                except Exception as e:
                    logger.error(f"Export failed: {e}", exc_info=True)
                    logger.warning("Will try to evaluate compressed model directly (may fail)")
            else:
                logger.info(f"Using existing export: {export_path}")
                eval_path = export_path

        return self._evaluate(model_path=eval_path)
            
            
                  

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

        If the checkpoint is in compressed format, it will be exported to
        standard HuggingFace format first (if use_export=True).

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Check if this is a compressed model (has compression_config.json)
        eval_path = checkpoint_path
        if self.config.use_export and (checkpoint_path / "compression_config.json").exists():
            logger.info("Detected compressed model, exporting to HF format for evaluation...")

            export_path = checkpoint_path / "exported_for_eval"

            # Only export if not already done
            if not export_path.exists():
                try:
                    export_to_hf_format(
                        compressed_model_path=str(checkpoint_path),
                        original_model_name=self.config.original_model_name,
                        output_path=str(export_path),
                        device_map=f"cuda:{self.config.gpu_ids[0]}",  # Use eval GPU
                        torch_dtype=torch.bfloat16
                    )
                    eval_path = export_path
                    logger.info(f"Export complete, will evaluate: {eval_path}")
                except Exception as e:
                    logger.error(f"Export failed: {e}", exc_info=True)
                    logger.warning("Will try to evaluate compressed model directly (may fail)")
            else:
                logger.info(f"Using existing export: {export_path}")
                eval_path = export_path

        # Build lm_eval command
        device_str = ",".join(str(gpu) for gpu in self.config.gpu_ids)

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={eval_path},dtype=bfloat16,trust_remote_code=True",
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
