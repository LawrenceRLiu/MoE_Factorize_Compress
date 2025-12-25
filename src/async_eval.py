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
        # self.evaluate_baseline()
        
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