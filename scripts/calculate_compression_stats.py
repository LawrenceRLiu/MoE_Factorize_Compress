# Simple script to calculate the compression statistics


import torch 
import hydra
import argparse
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import human_readable
from src.model_utils import get_model
from src.compressed_moe import SharedCoreExperts
from src.shared_core import SharedCoreLayer

@dataclass
class CompressionStats:
    original_n_params: int
    compressed_n_params: int
    active_n_params: int = -1
    active_compressed_n_params: int = -1
    compressed_moe_flops: int = -1 # this is just a relative count of the flops for the compressed MoE
    # may need to be multiplied by some factor to get actual FLOPS
    
    @property
    def total_compression_frac(self) -> float:
        return self.compressed_n_params / self.original_n_params
    
    @property
    def active_compression_frac(self) -> float:
        if self.active_n_params <= 0 or self.active_compressed_n_params <= 0:
            return -1.0
        return self.active_compressed_n_params / self.active_n_params
    
    def calculate_storage(self, parameter_count, fp32=True) -> str:
        return human_readable(parameter_count * (4 if fp32 else 2))+"B" 
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_n_params": self.original_n_params,
            "compressed_n_params": self.compressed_n_params,
            "active_n_params": self.active_n_params,
            "active_compressed_n_params": self.active_compressed_n_params,
            "compressed_moe_flops": self.compressed_moe_flops,
            "total_compression_frac": self.total_compression_frac,
            "active_compression_frac": self.active_compression_frac,
            "total_storage": {
                "original": {
                    "fp32": self.calculate_storage(self.original_n_params, fp32=True),
                    "fp16": self.calculate_storage(self.original_n_params, fp32=False)
                },
                "compressed": {
                    "fp32": self.calculate_storage(self.compressed_n_params, fp32=True),
                    "fp16": self.calculate_storage(self.compressed_n_params, fp32=False)
                },
            },
            "active_storage": {
                "original": {
                    "fp32": self.calculate_storage(self.active_n_params, fp32=True),
                    "fp16": self.calculate_storage(self.active_n_params, fp32=False)
                },
                "compressed": {
                    "fp32": self.calculate_storage(self.active_compressed_n_params, fp32=True),
                    "fp16": self.calculate_storage(self.active_compressed_n_params, fp32=False)
                },
            }
        }
        
    def __str__(self) -> str:
        #we won't return the storage stats here for brevity
        out = "Compression Statistics:\n\n"
        out += f"Original number of parameters: {human_readable(self.original_n_params)}\n"
        out += f"Compressed number of parameters: {human_readable(self.compressed_n_params)}\n"
        out += f"Total compression ratio: {self.total_compression_frac:.2f}x\n\n"
        if self.active_n_params > 0 and self.active_compressed_n_params > 0:
            out += f"Active original number of parameters: {human_readable(self.active_n_params)}\n"
            out += f"Active compressed number of parameters: {human_readable(self.active_compressed_n_params)}\n"
            out += f"Compressed MoE FLOPS (relative): {human_readable(self.compressed_moe_flops)}\n"
            out += f"Active compression ratio: {self.active_compression_frac:.2f}x\n"
            out += f"Active Flop Ratio: {self.compressed_moe_flops / self.active_n_params:.2f}x\n" # relative to original active params which is assumed to be roughly equal to original flops
        return out
        
    

if __name__ == "__main__":
    
    #argparse to get the compressed model directory and original model name
    parser = argparse.ArgumentParser(description="Calculate compression statistics for a compressed MoE model.\
        \n Example usage: python scripts/calculate_compression_stats.py --compressed_dir models/Qwen/Qwen3-30B-A3B-Base/experiment/checkpoints --save_path SAVE")
    parser.add_argument("--compressed_dir", type=str, required=True, 
                        help="Path to the directory containing the compressed model. this should not include the checkpoint subdir but the parent dir")
    parser.add_argument("--save_path", type=str, required=False, 
                        help="Path to save the statistics as a yaml file, if `SAVE` then the calculations will be saved to the config path",
                        default=None)
    parser.add_argument("--config_path", type=str, required=False, help="Path to the compression config file", default=None)
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    #load the config
    if args.config_path is None:
        config_path = Path(args.compressed_dir).parent / "compression_config.yaml"
    else:
        config_path = Path(args.config_path)

    with open(config_path, 'r') as f:
        comp_config = OmegaConf.load(f)
    original_model_name = comp_config.model_name
    logger.info(f"Extracted original model name: {original_model_name}")
    
    # Load compressed model
    compressed_model, _ = get_model(original_model_name)
    logger.info(f"Loading compressed model from {Path(args.compressed_dir) / 'checkpoint-0'}...")
    compressed_model = compressed_model.from_pretrained(
        Path(args.compressed_dir) / "checkpoint-0",
        device_map="cpu",
        dtype=torch.float16
    )
    logger.info("Compressed model loaded.")
    
    
    
    # load the original model for comparison
    from transformers import AutoModelForCausalLM
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        device_map="cpu",
        dtype=torch.float16
    )
    
    
    # Calculate the stats
    def infer_num_active_experts(model_config) -> int:
        for attr in ("num_experts_per_tok", "num_experts_per_token", "num_active_experts", "top_k"):
            val = getattr(model_config, attr, None)
            if val is not None:
                return int(val)
        return -1

    original_n_params = sum(p.numel() for p in original_model.parameters())
    compressed_n_params = sum(p.numel() for p in compressed_model.parameters())
    active_n_params = -1
    active_compressed_n_params = -1

    num_active_experts = infer_num_active_experts(original_model.config)
    logger.info(f"Inferred number of active experts during inference: {num_active_experts}") 
    if num_active_experts > 0:
        original_moe_total = 0
        original_moe_active = 0
        for layer in original_model.model.layers:
            experts = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                experts = layer.mlp.experts
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
                experts = layer.block_sparse_moe.experts
            if experts:
                num_experts = len(experts)
                per_expert_params = sum(p.numel() for p in experts[0].parameters())
                active_experts = min(num_active_experts, num_experts)
                original_moe_total += per_expert_params * num_experts
                original_moe_active += per_expert_params * active_experts

        compressed_moe_total = 0
        compressed_moe_flops = 0
        compressed_moe_active = 0
        for layer in compressed_model.model.layers:
            try: 
                moe_block = layer.mlp.experts
            except AttributeError:
                raise NotImplementedError("Only SharedCoreExperts MoE is supported in this script.")

            for shared_layer in [moe_block.up_shared_core, 
                                    moe_block.down_shared_core, 
                                    moe_block.gate_shared_core]:
                assert isinstance(shared_layer, SharedCoreLayer)
                
                stats = shared_layer.count_parameters()
                # logger.info(f"Shared Layer Stats: {stats}")
                active_experts = min(num_active_experts, shared_layer.num_experts)
                compressed_moe_total += stats["total_params"]
                compressed_moe_active += (
                    stats["core_params"] + stats["wrapper_params_per_expert"] * active_experts
                )
                compressed_moe_flops += (
                    (stats["core_params"] + stats["wrapper_params_per_expert"]) * active_experts
                )

        original_non_moe = original_n_params - original_moe_total
        compressed_non_moe = compressed_n_params - compressed_moe_total
        active_n_params = original_non_moe + original_moe_active
        active_compressed_n_params = compressed_non_moe + compressed_moe_active
        compressed_moe_flops = compressed_moe_flops + compressed_non_moe

    stats = CompressionStats(
        original_n_params=original_n_params,
        compressed_n_params=compressed_n_params,
        active_n_params=active_n_params,
        active_compressed_n_params=active_compressed_n_params,
        compressed_moe_flops=compressed_moe_flops
    )
    logger.info(stats)
    
    # save the stats if needed
    if args.save_path is not None:
        save_path = Path(args.save_path)
        if args.save_path.upper() == "SAVE":
            save_path = config_path.parent / "compression_stats.yaml"
        with open(save_path, 'w') as f:
            OmegaConf.save(OmegaConf.create(stats.to_dict()), f)
        logger.info(f"Saved compression statistics to {save_path}")
    
    
