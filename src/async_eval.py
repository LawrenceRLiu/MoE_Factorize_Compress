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
import glob

from .model_utils import get_hf_equivalent_model
from . import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_single_task(
    checkpoint_path: Union[str, Path],
    task_name: str,
    num_fewshot: int,
    save_path: Union[str, Path],
    n_gpus: int,
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
        n_gpus: Number of GPUUs to use for evaluation
        n_gpus_per_model: Number of GPUs to allocate per model, 
        if this is less than the number of gpus, we will parallelize across multiple gpus
        batch_size: Batch size for evaluation ("auto" or int)
    """
    
    #checking to see if the the number of gpus required to run the model is less than the number of gpus available
    assert n_gpus_per_model <= n_gpus, \
        "Number of gpus per model cannot be greater than the number of gpus available"
    
    #set the environment variable to use the specified gpus
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in range((n_gpus//n_gpus_per_model)*n_gpus_per_model))
    
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
    
    if n_gpus//n_gpus_per_model > 1:
        #we parallelize across multiple gpus
        prefix = [
             sys.executable,
             "-m", 
             "accelerate.commands.launch",
             "--multi_gpu",
             "--num_processes", str(n_gpus//n_gpus_per_model),
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
    eval_tasks: Dict[str, int]
    config_dir: Optional[str] = None
    batch_size: Union[int, str] = "auto"  # Batch size for evaluation
    n_gpus: int = 1
    n_gpus_per_model: int = 1  # GPUs per model for parallelization
    eval_interval: int = 60  # Seconds between checks
    wandb_run: Optional[object] = None  # wandb run object for logging
    
    def __post_init__(self):
        # Ensure directories exist
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.temp_dir = Path(self.temp_dir)
        self.eval_dir = Path(self.eval_dir)
        
        if self.config_dir:
            self.config_dir = Path(self.config_dir) / "compression_config.yaml"
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
        self.wandb_run = config.wandb_run

        # Set up wandb metrics if enabled
        if self.wandb_run is not None:
            # Define the iteration as the step metric
            self.wandb_run.define_metric("iteration")
            # Define all eval metrics to use iteration as x-axis (wildcard pattern)
            self.wandb_run.define_metric("eval/*", step_metric="iteration")
            self.wandb_run.define_metric("baseline/*", step_metric="iteration")

    def _log_to_wandb(
        self,
        task_name: str,
        fewshot: int,
        task_results: Dict,
        is_baseline: bool,
        model_path: Union[str, Path]
    ):
        """
        Log evaluation results to wandb.

        Args:
            task_name: Name of the evaluation task
            fewshot: Number of few-shot examples
            task_results: Results dictionary from lm_eval
            is_baseline: Whether this is the baseline model evaluation
            model_path: Path to the model being evaluated
        """
        # Extract metrics from the results
        if "results" not in task_results:
            logger.warning(f"No results found in task_results for {task_name}")
            return

        metrics = {}
        for task, task_metrics in task_results["results"].items():
            for metric_name, metric_value in task_metrics.items():
                # Skip stderr metrics and alias
                if "stderr" in metric_name or metric_name == "alias":
                    continue

                # Create metric key with appropriate prefix
                if is_baseline:
                    key = f"baseline/{task}/{metric_name}"
                else:
                    key = f"eval/{task}/{metric_name}"

                metrics[key] = metric_value

        # Determine the iteration for logging
        if is_baseline:
            # For baseline, we log at iteration 0
            iteration = 0
        else:
            # Extract checkpoint step from model_path (e.g., checkpoint-1000 -> 1000)
            checkpoint_name = Path(model_path).name
            if checkpoint_name.startswith("checkpoint-"):
                iteration = int(checkpoint_name.split("-")[1])
            else:
                logger.warning(f"Could not extract step from checkpoint name: {checkpoint_name}")
                iteration = 0

        # Add iteration to the metrics dict
        metrics["iteration"] = iteration

        # Log to wandb using the run object
        self.wandb_run.log(metrics)
        logger.info(f"Logged {len(metrics)-1} metrics to wandb at iteration {iteration}")

    def _evaluate(self, model_path: Optional[Path] = None, is_baseline: bool = False) -> Dict:

        if model_path is None:
            #then we evaluate the original model
            model_path = self.config.original_model_name
            results_path = self.config.eval_dir / "baseline_eval"
        else:
            #we expect the model_path to be a path to the temporary full hf model
            #replace the temp dir with eval dir
            relative_path = model_path.relative_to(self.config.temp_dir) #something along the lines of checkpoint-0
            results_path = self.config.eval_dir / relative_path / "eval_results"
            
        results_path.mkdir(parents=True, exist_ok=True)
        out = {}
        
        for task, fewshot in self.config.eval_tasks.items():
            logger.info(f"Evaluating task: {task} with {fewshot} few-shots saving to {results_path}")
            evaluate_single_task(
                checkpoint_path=model_path,
                task_name=task,
                num_fewshot=fewshot,
                save_path=results_path / f"{task}_fewshot_{fewshot}.json",
                n_gpus = self.config.n_gpus,
                n_gpus_per_model=self.config.n_gpus_per_model,
                batch_size=self.config.batch_size
            )
            #load the results
            results_paths = glob.glob(str(results_path / f"{task}_fewshot_{fewshot}_*.json"))
            if len(results_paths) > 1:
                logger.warning(f"Multiple result files found for {task} fewshot {fewshot}, using the first one found.")
            
            with open(results_paths[0], 'r') as f:
                task_results = json.load(f)
            out[f"{task}_fewshot_{fewshot}"] = task_results

            # Log to wandb if enabled
            if self.wandb_run is not None:
                self._log_to_wandb(task, fewshot, task_results, is_baseline, model_path)

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
        return self._evaluate(model_path=None, is_baseline=True)
    
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
        logger.info(f"GPU Stats: \n{utils.gpu_mem_info()}")
        # Check if this is a compressed model (has compression_config.json)
        eval_path = checkpoint_path
        
        equivalent_model = get_hf_equivalent_model(
            compressed_model_path=str(checkpoint_path),
            original_model_name=self.config.original_model_name,
            device_map="auto",
            dtype=torch.bfloat16
        )
        
        #save the equivalent to a temporary path
        temp_path = self.config.temp_dir / checkpoint_path.name # something like temp_dir/checkpoint-0
        equivalent_model.save_pretrained(temp_path)
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.original_model_name)
        tokenizer.save_pretrained(temp_path)
        
        del equivalent_model
        del tokenizer
        
        utils.clean()
        logger.info(f"GPU Stats: \n{utils.gpu_mem_info()}")
        
        return self._evaluate(model_path=temp_path)
            
    
    def run(self):
        """
        Main loop to monitor checkpoint directory and evaluate new checkpoints.
        """
        evaluated_checkpoints: Set[Path] = set()
        
        # First evaluate the baseline model
        self.evaluate_baseline()
        
        logger.info("Starting checkpoint monitoring...")
        
        while True:
            checkpoint_dirs = sorted(self.config.checkpoint_dir.glob("checkpoint-*"), key=os.path.getmtime)
            for checkpoint_dir in checkpoint_dirs:
                if checkpoint_dir not in evaluated_checkpoints:
                    logger.info(f"New checkpoint found: {checkpoint_dir}")
                    self.evaluate_checkpoint(checkpoint_dir)
                    logger.info(f"Finished evaluating checkpoint: {checkpoint_dir}")
                    
                    #TODO: Implement rotation/deletion of old checkpoints if needed
            logger.info(f"Sleeping for {self.config.eval_interval} seconds before next check...")
            raise NotImplementedError("stopping here for now")
            time.sleep(self.config.eval_interval)