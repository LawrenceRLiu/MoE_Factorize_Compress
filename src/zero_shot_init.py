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
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import os

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
    projections: List[str] = None  # e.g., ["gate_proj", "up_proj", "down_proj"]

    def __post_init__(self):
        if self.projections is None:
            self.projections = ["gate_proj", "up_proj", "down_proj"]


class MoELayerCompressor:
    """
    Handles compression of a single MoE layer's projections.

    Args:
        model_name: HuggingFace model identifier
        layer_idx: Index of the layer to compress
        rank: Rank for low-rank adapters
        config: Compression configuration
    """

    def __init__(
        self,
        model_name: str,
        layer_idx: int,
        rank: int,
        config: CompressionConfig
    ):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.rank = rank
        self.config = config

    def extract_expert_weights(
        self,
        model,
        layer_idx: int,
        projection: str
    ) -> Tuple[List[torch.Tensor], int, int]:
        """
        Extract expert weights for a specific projection from a layer.

        Args:
            model: The loaded MoE model
            layer_idx: Layer index
            projection: Projection name (e.g., "gate_proj")

        Returns:
            expert_weights: List of weight tensors
            d_out: Output dimension
            d_in: Input dimension
        """
        # Navigate to the MoE layer
        # This assumes Qwen/Mixtral style MoE structure
        # Adjust path based on actual model architecture
        try:
            # Try Qwen-MoE structure
            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                experts = layer.mlp.experts
            elif hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                # Mixtral-style
                experts = layer.block_sparse_moe.experts
            else:
                raise AttributeError("Could not locate experts in layer")

            # Extract weights from each expert
            expert_weights = []
            for expert in experts:
                if hasattr(expert, projection):
                    weight = getattr(expert, projection).weight.data.clone()
                    expert_weights.append(weight)
                else:
                    raise AttributeError(f"Expert does not have projection: {projection}")

            if not expert_weights:
                raise ValueError(f"No expert weights extracted for layer {layer_idx}, projection {projection}")

            d_out, d_in = expert_weights[0].shape
            logger.info(f"Extracted {len(expert_weights)} experts, shape: ({d_out}, {d_in})")

            return expert_weights, d_out, d_in

        except Exception as e:
            logger.error(f"Failed to extract weights: {e}")
            raise

    def compress_projection(
        self,
        model,
        projection: str,
        device: str = "cuda"
    ) -> Dict:
        """
        Compress a single projection (e.g., gate_proj) of a layer.

        Args:
            model: The loaded MoE model
            projection: Projection name
            device: Device to run compression on

        Returns:
            Dictionary containing compressed weights and metadata
        """
        logger.info(f"Compressing layer {self.layer_idx}, projection {projection}")

        # Extract expert weights
        expert_weights, d_out, d_in = self.extract_expert_weights(
            model, self.layer_idx, projection
        )

        # Run zero-shot initialization
        core, wrappers = initialize_from_experts(
            expert_weights=expert_weights,
            rank=self.rank,
            num_steps=self.config.num_steps,
            lr=self.config.lr,
            device=device
        )

        # Package results
        result = {
            "layer_idx": self.layer_idx,
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
                "num_experts": len(expert_weights),
                "d_in": d_in,
                "d_out": d_out,
                "rank": self.rank
            }
        }

        return result


def compress_layer_projection(
    gpu_id: int,
    model_name: str,
    layer_idx: int,
    projection: str,
    config: CompressionConfig,
    output_queue: Optional[mp.Queue] = None
):
    """
    Worker function to compress a single layer/projection on a specific GPU.

    Args:
        gpu_id: GPU device ID to use
        model_name: HuggingFace model identifier
        layer_idx: Layer index to compress
        projection: Projection name to compress
        config: Compression configuration
        output_queue: Optional queue to return results (for multiprocessing)
    """
    try:
        # Set device
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        logger.info(f"[GPU {gpu_id}] Loading model for layer {layer_idx}, {projection}")

        # Load model (consider loading in lower precision if memory constrained)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True
        )

        # Create compressor
        compressor = MoELayerCompressor(model_name, layer_idx, config.rank, config)

        # Compress the projection
        result = compressor.compress_projection(model, projection, device)

        # Save result
        output_dir = Path(config.output_dir) / f"layer_{layer_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{projection}.pt"
        torch.save(result, output_path)
        logger.info(f"[GPU {gpu_id}] Saved to {output_path}")

        # Also save metadata as JSON
        metadata_path = output_dir / f"{projection}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(result["metadata"], f, indent=2)

        # Return result if queue provided
        if output_queue is not None:
            output_queue.put((layer_idx, projection, "success"))

        # Clean up
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Error: {e}", exc_info=True)
        if output_queue is not None:
            output_queue.put((layer_idx, projection, f"error: {e}"))


def parallel_compression(
    model_name: str,
    config: CompressionConfig,
    gpu_ids: List[int],
    num_layers: Optional[int] = None
):
    """
    Parallel compression across multiple GPUs.

    Each GPU is assigned a (layer, projection) pair to compress independently.

    Args:
        model_name: HuggingFace model identifier
        config: Compression configuration
        gpu_ids: List of GPU IDs to use
        num_layers: Number of layers to compress (if None, infer from model config)
    """
    # Get model config to determine number of layers and active experts
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if num_layers is None:
        num_layers = model_config.num_hidden_layers
        logger.info(f"Model has {num_layers} layers")

    # Infer number of active experts for statistics
    num_active_experts = infer_num_active_experts(model_config)
    logger.info(f"Model uses {num_active_experts} active experts per token")

    # Create work items: (layer_idx, projection)
    work_items = []
    for layer_idx in range(num_layers):
        for projection in config.projections:
            work_items.append((layer_idx, projection))

    logger.info(f"Total work items: {len(work_items)}")
    logger.info(f"Available GPUs: {gpu_ids}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save compression config
    config_path = output_dir / "compression_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Multiprocessing setup
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # Process work items
    num_gpus = len(gpu_ids)
    processes = []
    work_idx = 0
    completed = 0

    while work_idx < len(work_items) or processes:
        # Launch new processes if GPUs available and work remaining
        while len(processes) < num_gpus and work_idx < len(work_items):
            layer_idx, projection = work_items[work_idx]
            gpu_id = gpu_ids[len(processes)]

            logger.info(f"Launching compression on GPU {gpu_id}: layer {layer_idx}, {projection}")

            p = mp.Process(
                target=compress_layer_projection,
                args=(gpu_id, model_name, layer_idx, projection, config, result_queue)
            )
            p.start()
            processes.append(p)
            work_idx += 1

        # Check for completed processes
        for p in processes[:]:
            if not p.is_alive():
                p.join()
                processes.remove(p)

                # Get result from queue
                if not result_queue.empty():
                    layer_idx, projection, status = result_queue.get()
                    completed += 1
                    logger.info(f"Completed ({completed}/{len(work_items)}): layer {layer_idx}, {projection} - {status}")

    logger.info("All compression tasks completed!")
    logger.info(f"Results saved to: {output_dir}")

    # Collect and save compression statistics
    logger.info("\nCollecting compression statistics...")
    stats = CompressionStats()

    for layer_idx in range(num_layers):
        layer_dir = output_dir / f"layer_{layer_idx}"
        for projection in config.projections:
            metadata_path = layer_dir / f"{projection}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                stats.add_layer_stats(
                    layer_idx=layer_idx,
                    projection=projection,
                    num_experts=metadata["num_experts"],
                    d_in=metadata["d_in"],
                    d_out=metadata["d_out"],
                    rank=metadata["rank"],
                    num_active_experts=num_active_experts
                )

    # Save aggregate statistics
    stats.save(output_dir / "compression_statistics.yaml")
    logger.info("Compression statistics saved!")


def load_compressed_model(
    original_model,
    compressed_dir: str,
    device: str = "cuda"
):
    """
    Load compressed weights into a model with SharedCoreLayer modules.

    This function replaces the original MoE expert layers with compressed versions.

    Args:
        original_model: The original model to modify
        compressed_dir: Directory containing compressed weights
        device: Device to load weights onto

    Returns:
        Model with compressed layers
    """
    compressed_path = Path(compressed_dir)

    # Load compression config
    config_path = compressed_path / "compression_config.json"
    with open(config_path, 'r') as f:
        comp_config = json.load(f)

    projections = comp_config["projections"]
    rank = comp_config["rank"]

    # Iterate through layers and load compressed weights
    layer_dirs = sorted(compressed_path.glob("layer_*"))

    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.split("_")[1])
        logger.info(f"Loading compressed layer {layer_idx}")

        # Load each projection
        for projection in projections:
            proj_path = layer_dir / f"{projection}.pt"
            if not proj_path.exists():
                logger.warning(f"Missing {proj_path}, skipping")
                continue

            # Load compressed data
            data = torch.load(proj_path, map_location=device)

            # Create SharedCoreLayer
            metadata = data["metadata"]
            shared_layer = SharedCoreLayer(
                num_experts=metadata["num_experts"],
                d_in=metadata["d_in"],
                d_out=metadata["d_out"],
                rank=rank,
                init_core=data["core"].to(device)
            )

            # Load wrapper parameters
            for expert_idx, wrapper_data in enumerate(data["wrappers"]):
                expert = shared_layer.get_expert(expert_idx)
                expert.input_wrapper.U.data = wrapper_data["U_in"].to(device)
                expert.input_wrapper.V.data = wrapper_data["V_in"].to(device)
                expert.output_wrapper.U.data = wrapper_data["U_out"].to(device)
                expert.output_wrapper.V.data = wrapper_data["V_out"].to(device)

            # TODO: Replace the projection in the original model
            # This requires modifying the model architecture
            # For now, just store the compressed layer
            logger.info(f"  Loaded {projection}: {shared_layer.count_parameters()}")

    return original_model
