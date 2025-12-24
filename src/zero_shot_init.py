"""
Zero-Shot Initialization for Shared Core Compression

Parallelizes compression across multiple GPUs, with each GPU handling
one layer/projection combination.
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import os
import traceback
from queue import Empty
from threading import Lock
from omegaconf import OmegaConf
import shutil

from .shared_core import initialize_from_experts, SharedCoreLayer
from .compression_stats import CompressionStats, infer_num_active_experts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for zero-shot compression."""
    model_name: str
    rank: int
    num_steps: int = 1000
    lr: float = 1e-3
    output_dir: str = "./models/compressed"
    config_dir: str = "./models/compressed/config"
    projections: List[str] = None  # e.g., ["gate_proj", "up_proj", "down_proj"]

    def __post_init__(self):
        if self.projections is None:
            self.projections = ["gate_proj", "up_proj", "down_proj"]



def extract_all_expert_weights(
    model_name: str,
    projections: List[str],
    output_dir: str,
    device: str = "cuda:0"
):
    """
    Extract all expert weights from the model and save to disk.

    This is a one-time operation that loads the model once and extracts
    all expert weights for all layers and projections.

    Args:
        model_name: HuggingFace model identifier
        projections: List of projection names to extract
        output_dir: Directory to save extracted weights
        device: Device to load model on
    """
    logger.info(f"Loading model {model_name} for weight extraction...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    )

    # Get number of layers
    num_layers = len(model.model.layers)
    logger.info(f"Extracting weights from {num_layers} layers")

    extraction_dir = Path(output_dir) / "extracted_weights"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Extract weights for each layer and projection
    for layer_idx in range(num_layers):
        logger.info(f"Extracting layer {layer_idx}/{num_layers}")
        layer = model.model.layers[layer_idx]

        # Navigate to experts
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
        elif hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
            experts = layer.block_sparse_moe.experts
        else:
            raise AttributeError(f"Could not locate experts in layer {layer_idx}")

        for projection in projections:
            # precalculate the save path:
            save_path = extraction_dir / f"layer_{layer_idx}_{projection}.pt"
            if save_path.exists():
                logger.info(f"  Skipping {projection}, already extracted at {save_path}")
                continue
            # Extract expert weights for this projection
            expert_weights = []
            expert_biases = []
            for expert in experts:
                if hasattr(expert, projection):
                    weight = getattr(expert, projection).weight.data.cpu().clone()
                    bias = getattr(expert, projection).bias.data.cpu().clone() if getattr(expert, projection).bias is not None else None
                    expert_weights.append(weight)
                    expert_biases.append(bias)
                else:
                    raise AttributeError(f"Expert does not have projection: {projection}")

            data = {
                "expert_weights": expert_weights,
                "num_experts": len(expert_weights),
                "d_out": expert_weights[0].shape[0],
                "d_in": expert_weights[0].shape[1]
            }
            
            if any(b is not None for b in expert_biases):
                #check that they are all present
                if not all(b is not None for b in expert_biases):
                    raise ValueError(f"Some experts are missing biases for projection {projection} in layer {layer_idx}")
                data["expert_biases"] = expert_biases
            # Save to disk
            torch.save(data, save_path)

            logger.info(f"  Saved {projection}: {len(expert_weights)} experts, shape {expert_weights[0].shape}")

    # Clean up
    del model
    torch.cuda.empty_cache()
    logger.info(f"Weight extraction complete! Saved to {extraction_dir}")

    return extraction_dir


def compression_worker(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_queue: mp.Queue,
    stop_event: mp.Event,
    extraction_dir: str,
    config: CompressionConfig
):
    """
    Worker process that pulls tasks from task_queue and processes them.

    Uses a GPU queue to dynamically allocate available GPUs to workers.
    Continues processing until task_queue is empty or stop_event is set.

    Args:
        worker_id: Unique identifier for this worker
        task_queue: Queue containing (layer_idx, projection) tasks
        result_queue: Queue to put results into
        gpu_queue: Queue containing available GPU IDs
        stop_event: Event to signal workers to stop
        extraction_dir: Directory containing extracted weights
        config: Compression configuration
    """
    gpu_id = None

    try:
        # Get a GPU from the queue
        gpu_id = gpu_queue.get()
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        logger.info(f"[Worker {worker_id}] Started on GPU {gpu_id}")

        while not stop_event.is_set():
            try:
                # Try to get a task (non-blocking with timeout)
                task = task_queue.get(timeout=1)

                if task is None:  # Sentinel value to stop worker
                    break

                layer_idx, projection = task

                logger.info(f"[Worker {worker_id} | GPU {gpu_id}] Processing layer {layer_idx}, {projection}")

                # Load pre-extracted weights from disk
                weight_path = Path(extraction_dir) / f"layer_{layer_idx}_{projection}.pt"
                if not weight_path.exists():
                    raise FileNotFoundError(f"Extracted weights not found: {weight_path}")

                weight_data = torch.load(weight_path, map_location="cpu")
                expert_weights = weight_data["expert_weights"]
                expert_biases = weight_data.get("expert_biases", None) #shape of (num_experts, d_out) if present
                # Run zero-shot initialization
                core, wrappers = initialize_from_experts(
                    expert_weights=expert_weights,
                    rank=config.rank,
                    num_steps=config.num_steps,
                    lr=config.lr,
                    device=device,
                    logger=logger,
                    layer_name=f"Layer {layer_idx} {projection}"
                )

                # Package results
                result = {
                    "layer_idx": layer_idx,
                    "projection": projection,
                    "core": core.cpu(),
                    "wrappers": [
                        {
                            "U_in": U_in.cpu(),
                            "V_in": V_in.cpu(),
                            "U_out": U_out.cpu(),
                            "V_out": V_out.cpu()
                        }
                        for U_in, V_in, U_out, V_out in wrappers
                    ],
                    "metadata": {
                        "num_experts": weight_data["num_experts"],
                        "d_in": weight_data["d_in"],
                        "d_out": weight_data["d_out"],
                        "rank": config.rank
                    }
                }
                
                #add the biases if available
                if expert_biases is not None:
                    result["biases"] = [b.cpu() for b in expert_biases]

                # Save result
                output_dir = Path(config.output_dir) / f"layer_{layer_idx}"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"{projection}.pt"
                torch.save(result, output_path)

                # Save metadata as JSON
                metadata_path = output_dir / f"{projection}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(result["metadata"], f, indent=2)

                # Put result in queue
                result_queue.put({
                    "status": "success",
                    "layer_idx": layer_idx,
                    "projection": projection,
                    "worker_id": worker_id,
                    "gpu_id": gpu_id
                })

                logger.info(f"[Worker {worker_id} | GPU {gpu_id}] Completed layer {layer_idx}, {projection}")

                # Clean up
                torch.cuda.empty_cache()

            except Empty:
                # No tasks available, continue waiting
                continue
            except Exception as e:
                # Task-specific error - log and report but don't stop worker
                logger.error(f"[Worker {worker_id} | GPU {gpu_id}] Error processing task: {e}")
                traceback.print_exc()

                result_queue.put({
                    "status": "error",
                    "layer_idx": layer_idx if 'layer_idx' in locals() else None,
                    "projection": projection if 'projection' in locals() else None,
                    "error": str(e),
                    "worker_id": worker_id,
                    "gpu_id": gpu_id
                })
                #for now treat all errors as critical
                stop_event.set()
                break
                # For critical errors, set stop event
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    stop_event.set()
                    break

    except Exception as e:
        # Worker-level error (e.g., GPU allocation failed)
        logger.error(f"[Worker {worker_id}] Critical error: {e}")
        traceback.print_exc()
        stop_event.set()
        result_queue.put({
            "status": "critical_error",
            "error": str(e),
            "worker_id": worker_id,
            "gpu_id": gpu_id
        })

    finally:
        # Return GPU to queue
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            logger.info(f"[Worker {worker_id}] Released GPU {gpu_id} and exiting")


def parallel_compression(
    model_name: str,
    config: CompressionConfig,
    gpu_ids: List[int],
    num_layers: Optional[int] = None,
    skip_extraction: bool = False
):
    """
    Parallel compression across multiple GPUs with queue-based worker pool.

    Phase 1: Extract all expert weights from model and save to disk (once)
    Phase 2: Workers pull tasks from queue and compress in parallel

    Args:
        model_name: HuggingFace model identifier
        config: Compression configuration
        gpu_ids: List of GPU IDs to use
        num_layers: Number of layers to compress (if None, infer from model config)
        skip_extraction: If True, skip Phase 1 and assume weights already extracted
    """
    # Get model config to determine number of layers and active experts
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if num_layers is None:
        num_layers = model_config.num_hidden_layers
        logger.info(f"Model has {num_layers} layers")

    # Infer number of active experts for statistics
    num_active_experts = infer_num_active_experts(model_config)
    logger.info(f"Model uses {num_active_experts} active experts per token")

    # Create output directory
    #add a `checkpoint-0` suffix to indicate zero-shot init
    config.output_dir = os.path.join(config.output_dir, "checkpoint-0")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save compression config
    config_path = Path(config.config_dir) / "compression_config.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(config=OmegaConf.structured(config), f=config_path)

    logger.info(f"Saved compression config to {config_path}")

    # ===== PHASE 1: Extract weights =====
    extraction_dir = output_dir / "extracted_weights"
    if not skip_extraction:
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Extracting expert weights from model")
        logger.info("="*60)
        extraction_dir = extract_all_expert_weights(
            model_name=model_name,
            projections=config.projections,
            output_dir=str(output_dir),
            device=f"cuda:{gpu_ids[0]}"  # Use first GPU for extraction
        )
    else:
        logger.info(f"Skipping weight extraction (using existing weights in {extraction_dir})")
        if not extraction_dir.exists():
            raise FileNotFoundError(f"Extraction directory not found: {extraction_dir}")

    # ===== PHASE 2: Queue-based parallel compression =====
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Queue-based parallel compression")
    logger.info("="*60)

    # Create work items: (layer_idx, projection)
    work_items = []
    for layer_idx in range(num_layers):
        for projection in config.projections:
            work_items.append((layer_idx, projection))

    total_tasks = len(work_items)
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"Number of workers: {len(gpu_ids)}")

    # Multiprocessing setup
    mp.set_start_method('spawn', force=True)

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    gpu_queue = mp.Queue()
    stop_event = mp.Event()

    # Populate task queue
    for task in work_items:
        task_queue.put(task)

    # Populate GPU queue
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    # Start worker processes
    num_workers = len(gpu_ids)
    workers = []

    logger.info(f"Starting {num_workers} worker processes...")
    for worker_id in range(num_workers):
        p = mp.Process(
            target=compression_worker,
            args=(worker_id, task_queue, result_queue, gpu_queue, stop_event, str(extraction_dir), config)
        )
        p.start()
        workers.append(p)

    # Monitor progress
    completed_tasks = 0
    failed_tasks = 0
    results = []

    logger.info("Workers started. Monitoring progress...")

    while completed_tasks + failed_tasks < total_tasks:
        try:
            # Get result with timeout
            result = result_queue.get(timeout=2)
            results.append(result)

            if result["status"] == "success":
                completed_tasks += 1
                logger.info(
                    f"Progress: {completed_tasks + failed_tasks}/{total_tasks} "
                    f"(Success: {completed_tasks}, Failed: {failed_tasks}) | "
                    f"Layer {result['layer_idx']}, {result['projection']} "
                    f"[Worker {result['worker_id']} | GPU {result['gpu_id']}]"
                )
            elif result["status"] == "error":
                failed_tasks += 1
                logger.error(
                    f"Task failed: Layer {result['layer_idx']}, {result['projection']} - "
                    f"Error: {result['error']}"
                )
            elif result["status"] == "critical_error":
                logger.error(f"Critical error from worker {result['worker_id']}: {result['error']}")
                stop_event.set()
                break

        except Empty:
            # No results yet, check if stop event is set
            if stop_event.is_set():
                logger.error("Stop event detected. Terminating workers...")
                break
            continue
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Stopping workers...")
            stop_event.set()
            break

    # Send sentinel values to stop workers gracefully
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all workers to finish
    logger.info("Waiting for workers to finish...")
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            logger.warning(f"Worker {worker.pid} did not finish gracefully, terminating...")
            worker.terminate()
            worker.join()

    logger.info("\n" + "="*60)
    logger.info(f"Compression complete!")
    logger.info(f"Completed: {completed_tasks}/{total_tasks}")
    logger.info(f"Failed: {failed_tasks}/{total_tasks}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    
    #clear the extraction directory
    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)
        logger.info(f"Cleaned up extraction directory: {extraction_dir}")

    # Return summary
    return {
        "total_tasks": total_tasks,
        "completed": completed_tasks,
        "failed": failed_tasks,
        "results": results
    }
