import torch
import torch.nn as nn
import numpy as np
import random
import wandb
import gc
import os
import glob
import shutil


def recursive_apply(module: nn.Module, func_name: str, func_kwargs: dict = {}):
    """recursively applies a function to a module and its children

    Args:
        module (nn.Module): the module to apply the function to
        func (function): the function to apply

    """
    for name, child in module.named_children():
        if hasattr(child, func_name):
            if callable(getattr(child, func_name)):
                getattr(child, func_name)(**func_kwargs)
        # otherwise look for its children
        else:
            recursive_apply(child, func_name, func_kwargs)


def recursive_find(module: nn.Module, name: str) -> nn.Module:
    # print(name)
    if name == "":
        return module
    if "." not in name:
        return getattr(module, name)
    else:
        return recursive_find(
            getattr(module, name[: name.find(".")]), name[name.find(".") + 1 :]
        )


def intialize_wandb(args, config: dict = None):
    if not args.use_wandb:
        return

    project_name = None if not hasattr(args, "wandb_project") else args.wandb_project
    run_name = None if not hasattr(args, "wandb_run_name") else args.wandb_run_name
    run_id = None if not hasattr(args, "wandb_run_id") else args.wandb_run_id

    wandb.init(
        project=project_name, name=run_name, id=run_id, config=config, resume="allow"
    )


def seed(seed, seed_all: bool = False):
    torch.manual_seed(seed)
    if seed_all:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
def clean_dir(dir):
    """cleans a directory by removing all files in it"""
    if not os.path.exists(dir):
        return
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def get_gpu_memory(device: torch.device, return_str: bool = False):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - reserved_memory - allocated_memory

    as_gb = lambda x: round(x / 1024**3, 2)
    if return_str:
        return f"Total Memory: {as_gb(total_memory)}GB, Reserved Memory: {as_gb(reserved_memory)}GB, Allocated Memory: {as_gb(allocated_memory)}GB, Free Memory: {as_gb(free_memory)}GB"
    else:
        print(
            f"Total Memory: {as_gb(total_memory)}GB, Reserved Memory: {as_gb(reserved_memory)}GB, Allocated Memory: {as_gb(allocated_memory)}GB, Free Memory: {as_gb(free_memory)}GB"
        )


def find_run_num(save_path: str) -> str:
    """counts the number of runs in a directory and returns 'run_x' where x is the next run number"""
    n_prev_runs = len(glob.glob(os.path.join(save_path, "run_*")))
    return f"run_{n_prev_runs}"


# from https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings
def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )

def gumble_noise(x):
    """adds gumbel noise to a tensor

    Args:
        x (torch.Tensor): the tensor to add noise to

    Returns:
        torch.Tensor: the tensor with noise added
    """
    noise = torch.empty_like(x).uniform_()
    noise = -torch.log(-torch.log(noise))
    return noise


def power_iteration(matrix, num_iterations=100, tolerance=1e-6):
    """
    Performs power iteration to find the dominant eigenvector

    Args:
        matrix (torch.Tensor): The square matrix for which to find the dominant eigenpair.
        num_iterations (int): The maximum number of iterations.
        tolerance (float): The convergence tolerance for the eigenvector.

    Returns:
        tuple: A tuple containing the dominant eigenvector
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    # Initialize a random vector
    eigenvector = torch.rand(n, 1, dtype=matrix.dtype, device=matrix.device)
    eigenvector = eigenvector / torch.norm(eigenvector) # Normalize

    for i in range(num_iterations):
        # Apply the matrix to the current eigenvector estimate
        new_eigenvector = torch.matmul(matrix, eigenvector)


        # Normalize the new eigenvector
        new_eigenvector = new_eigenvector / torch.norm(new_eigenvector)

        # Check for convergence
        if torch.norm(new_eigenvector - eigenvector) < tolerance:
            break

        eigenvector = new_eigenvector

    return eigenvector.squeeze(1)

from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from typing import Optional
import copy

def blank_init(compression_config:DictConfig,
               n_in: int,
               n_out: int,
               dtype: torch.dtype = torch.float32,
               device: torch.device = None,
               add_bias: Optional[bool] = False):
    
    config_use = copy.deepcopy(compression_config)
    with open_dict(config_use):
        config_use._target_ = config_use.init_config._target_ +  ".blank_init"
        del config_use.init_config
    compression_module = instantiate(config_use,
        n_in=n_in,
        n_out=n_out,
        dtype=dtype,
        device=device,
        add_bias=add_bias,
        _recursive_=False
    )
    return compression_module



import time 
from typing import Dict
class Timer:
    running_times:Dict[str, float] = {}
    running_counts:Dict[str, int] = {} 
    start_times:Dict[str, float] = {}   
    
    def start(self, name: str):
        """Starts a timer for a given name."""
        if name in self.start_times:
            raise ValueError(f"Timer '{name}' is already running.")
        self.start_times[name] = time.time()
        if name not in self.running_times:
            self.running_times[name] = 0.0
            self.running_counts[name] = 0
    def stop(self, name: str):
        """Stops the timer for a given name and updates the running time."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' is not running.")
        elapsed_time = time.time() - self.start_times[name]
        self.running_times[name] += elapsed_time
        self.running_counts[name] += 1
        del self.start_times[name]
    
    def get_times(self) -> Dict[str, float]:
        """Returns the total running times for all timers."""
        return {name: self.running_times[name]/ self.running_counts[name] for name in self.running_times if self.running_counts[name] > 0}
    
    def summarize(self) -> str:
        """Returns a string summary of the running times."""
        summary = []
        for name, total_time in self.running_times.items():
            count = self.running_counts[name]
            avg_time = total_time / count if count > 0 else 0
            summary.append(f"{name}: {total_time:.4f}s (avg: {avg_time:.4f}s over {count} runs)")
        return "\n".join(summary)
    

if __name__ == "__main__":

    for gpu in range(torch.cuda.device_count()):
        print(f"GPU {gpu}")
        print(get_gpu_memory(gpu, return_str=True))
