"""
Compressed MoE Model Architecture

This module defines a custom model class that integrates compressed SharedCore layers
into the HuggingFace model architecture. Since we're making fundamental architecture
changes, we can't use standard save/load functions.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Tuple, List
from omegaconf import OmegaConf

from .shared_core import SharedCoreLayer, CompressedExpert


logger = logging.getLogger(__name__)


class CompressedMoEExpert(nn.Module):
    """
    Wrapper around CompressedExpert that matches the original expert interface.

    This allows seamless replacement of original experts while maintaining
    the same forward signature.
    """

    def __init__(
        self,
        compressed_expert: CompressedExpert,
        original_expert_structure: Dict[str, any] = None
    ):
        super().__init__()
        self.compressed_expert = compressed_expert
        self.original_structure = original_structure or {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass matching original expert interface."""
        return self.compressed_expert(hidden_states)


class CompressedMoEBlock(nn.Module):
    """
    Compressed MoE block replacing the original sparse MoE layer.

    This module handles routing and expert execution using compressed experts.
    """

    def __init__(
        self,
        gate: nn.Module,  # Original routing gate
        shared_core_layers: Dict[str, SharedCoreLayer],  # {proj_name: SharedCoreLayer}
        num_experts: int,
        num_experts_per_tok: int = 2
    ):
        super().__init__()
        self.gate = gate  # Keep original routing mechanism
        self.shared_core_layers = nn.ModuleDict(shared_core_layers)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through compressed MoE block.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten for routing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing weights using original gate
        router_logits = self.gate(hidden_states_flat)
        routing_weights = torch.softmax(router_logits, dim=-1)

        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Process through compressed experts
        final_output = torch.zeros_like(hidden_states_flat)

        # For each selected expert, apply the compressed transformations
        # This is a simplified version - actual implementation depends on architecture
        for i in range(self.num_experts_per_tok):
            expert_idx = selected_experts[:, i]
            expert_weight = routing_weights[:, i:i+1]

            # Get unique expert indices for batching
            unique_experts = torch.unique(expert_idx)

            for exp_id in unique_experts:
                mask = expert_idx == exp_id
                if not mask.any():
                    continue

                # Get inputs for this expert
                expert_input = hidden_states_flat[mask]

                # Process through compressed layers
                # Assuming FFN structure: gate_proj, up_proj, down_proj
                if 'gate_proj' in self.shared_core_layers and 'up_proj' in self.shared_core_layers:
                    # SwiGLU-style: down_proj(silu(gate_proj(x)) * up_proj(x))
                    gate_out = self.shared_core_layers['gate_proj'](expert_input, exp_id.item())
                    up_out = self.shared_core_layers['up_proj'](expert_input, exp_id.item())
                    activated = torch.nn.functional.silu(gate_out) * up_out

                    if 'down_proj' in self.shared_core_layers:
                        expert_output = self.shared_core_layers['down_proj'](activated, exp_id.item())
                    else:
                        expert_output = activated
                else:
                    # Fallback: assume single projection
                    proj_name = list(self.shared_core_layers.keys())[0]
                    expert_output = self.shared_core_layers[proj_name](expert_input, exp_id.item())

                # Accumulate weighted output
                weights = expert_weight[mask]
                final_output[mask] += expert_output * weights

        # Reshape back
        final_output = final_output.view(batch_size, seq_len, -1)

        return final_output


def load_compressed_model(
    compressed_dir: str,
    original_model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    config_path: Optional[str] = None
) -> nn.Module:
    """
    Load a compressed MoE model from disk.

    This function:
    1. Loads the original model architecture on CPU
    2. Replaces MoE expert layers with compressed versions (on CPU)
    3. Loads compressed weights from disk
    4. Moves the entire model to target device(s)

    Args:
        compressed_dir: Directory containing compressed weights
        original_model_name: Original HuggingFace model name
        device_map: Device mapping for model parallelism ("auto", "cpu", or specific device)
        torch_dtype: Data type for model weights
        config_path: Optional path to compression config, if not provided assumes its located in the parent directory of compressed_dir

    Returns:
        Model with compressed MoE layers
    """
    compressed_path = Path(compressed_dir)

    # Load compression config
    if config_path is None:
        config_path = compressed_path.parent / "compression_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        comp_config = OmegaConf.load(f)

    logger.info(f"Loading compressed model from {compressed_path}")
    logger.info(f"Original model: {original_model_name}")

    # Load the original model architecture on CPU first
    # This avoids loading expert weights to GPU only to immediately discard them
    logger.info("Loading base model architecture on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Get model config
    model_config = model.config

    # Replace expert layers with compressed versions (all on CPU)
    projections = comp_config["projections"]
    rank = comp_config["rank"]

    logger.info(f"searching {compressed_path} for compressed layers...")
    layer_dirs = sorted(compressed_path.glob("layer_*"))
    logger.info(f"Found {len(layer_dirs)} compressed layers")

    for layer_dir in layer_dirs:
        layer_idx = int(layer_dir.name.split("_")[1])
        logger.info(f"Loading compressed layer {layer_idx}...")

        # Load compressed data for each projection
        shared_core_layers = {}

        for projection in projections:
            proj_path = layer_dir / f"{projection}.pt"
            if not proj_path.exists():
                logger.warning(f"Missing {proj_path}, skipping")
                continue

            # Load compressed data (already on CPU)
            logger.info(f"  Loading {projection}...")
            data = torch.load(proj_path, map_location='cpu')

            # Create SharedCoreLayer on CPU
            metadata = data["metadata"]
            shared_layer = SharedCoreLayer(
                num_experts=metadata["num_experts"],
                d_in=metadata["d_in"],
                d_out=metadata["d_out"],
                rank=rank,
                init_core=data["core"]
            )

            # Load wrapper parameters
            for expert_idx, wrapper_data in enumerate(data["wrappers"]):
                expert = shared_layer.get_expert(expert_idx)
                expert.input_wrapper.U.data = wrapper_data["U_in"]
                expert.input_wrapper.V.data = wrapper_data["V_in"]
                expert.output_wrapper.U.data = wrapper_data["U_out"]
                expert.output_wrapper.V.data = wrapper_data["V_out"]

            shared_core_layers[projection] = shared_layer

        # Replace the layer in the model (on CPU)
        try:
            layer = model.model.layers[layer_idx]

            # Try to find the MoE block
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                # Qwen-style
                original_gate = layer.mlp.gate if hasattr(layer.mlp, 'gate') else None
                num_experts = len(layer.mlp.experts)
                num_experts_per_tok = getattr(model_config, 'num_experts_per_tok', 2)

                # Create compressed MoE block (on CPU)
                compressed_block = CompressedMoEBlock(
                    gate=original_gate,
                    shared_core_layers=shared_core_layers,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok
                )

                # Replace the MLP (no need to cast device - everything is on CPU)
                layer.mlp = compressed_block
                logger.info(f"  Replaced layer {layer_idx} MoE block")

            elif hasattr(layer, 'block_sparse_moe'):
                raise NotImplementedError("Mixtral-style BlockSparseMoE replacement not implemented yet")
                # Mixtral-style
                original_gate = layer.block_sparse_moe.gate
                num_experts = len(layer.block_sparse_moe.experts)
                num_experts_per_tok = getattr(model_config, 'num_experts_per_tok', 2)

                compressed_block = CompressedMoEBlock(
                    gate=original_gate,
                    shared_core_layers=shared_core_layers,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok
                )

                layer.block_sparse_moe = compressed_block
                logger.info(f"  Replaced layer {layer_idx} sparse MoE block")

        except Exception as e:
            logger.error(f"Failed to replace layer {layer_idx}: {e}")
            logger.warning(f"Layer {layer_idx} will use original weights")

    # Load non-MoE parameters (attention, norms, embeddings, etc.)
    non_moe_weights_path = compressed_path / "non_moe_weights.pt"
    if non_moe_weights_path.exists():
        logger.info("Loading non-MoE parameters (attention, norms, embeddings, etc.)...")
        non_moe_state_dict = torch.load(non_moe_weights_path, map_location='cpu')

        # Load these parameters into the model
        missing_keys, unexpected_keys = model.load_state_dict(non_moe_state_dict, strict=False)

        # It's expected to have missing keys (the compressed MoE layers)
        # and unexpected keys (if model structure changed slightly)
        if missing_keys:
            logger.info(f"Missing keys (expected for compressed layers): {len(missing_keys)} keys")
            logger.debug(f"Missing: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
            logger.debug(f"Unexpected: {unexpected_keys[:5]}...")

        logger.info(f"Loaded {len(non_moe_state_dict)} non-MoE parameters")
    else:
        logger.warning("No non_moe_weights.pt found - model may be missing attention, norms, etc!")

    # Now move the entire model to the target device(s)
    if device_map != "cpu":
        logger.info(f"Moving model to device_map: {device_map}")
        try:
            from accelerate import dispatch_model, infer_auto_device_map

            # Build no_split_module_classes list
            # Include original model's no-split modules plus our custom CompressedMoEBlock
            no_split_modules = []
            if hasattr(model, '_no_split_modules'):
                no_split_modules.extend(model._no_split_modules)

            # Add our custom CompressedMoEBlock to prevent splitting compressed experts
            no_split_modules.append("CompressedMoEBlock")

            # Remove duplicates while preserving order
            no_split_modules = list(dict.fromkeys(no_split_modules))

            # Infer device map based on available memory
            device_map_dict = infer_auto_device_map(
                model,
                max_memory=None,
                dtype=torch_dtype,
                no_split_module_classes=no_split_modules
            )

            # Dispatch model to devices
            model = dispatch_model(model, device_map=device_map_dict)
            logger.info("Model successfully dispatched to devices")
        except ImportError:
            logger.warning("accelerate library not available, moving to single device")
            # Fallback: move to single device if accelerate is not available
            if device_map == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_map
            model = model.to(device)
            logger.info(f"Model moved to {device}")

    logger.info("Compressed model loaded successfully!")
    return model


def save_compressed_model(
    model: nn.Module,
    output_dir: str,
    tokenizer: Optional[AutoTokenizer] = None
):
    """
    Save a compressed model to disk.

    This saves:
    1. All non-MoE parameters (embeddings, attention, layer norms, etc.) in standard format
    2. Compressed MoE layers in custom format
    3. Model config and tokenizer

    Args:
        model: Compressed model
        output_dir: Directory to save model
        tokenizer: Optional tokenizer to save alongside model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("="*10 + " Saving Compressed Model " + "="*10)
    logger.info(f"Saving compressed model to {output_path}")

    # Save model config
    model.config.save_pretrained(output_path)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    # Collect state dict for non-MoE parameters
    # We'll save everything EXCEPT the compressed MoE expert weights
    non_moe_state_dict = {}
    compressed_layer_indices = []

    for name, param in model.named_parameters():
        # Skip compressed MoE components (they'll be saved separately)
        if 'shared_core_layers' in name or 'CompressedMoE' in name:
            continue
        # Skip original expert weights if they still exist
        if 'experts.' in name and '.mlp.' in name:
            continue

        # Save everything else: embeddings, attention, layer norms, lm_head, etc.
        non_moe_state_dict[name] = param.detach().cpu()

    # Also save buffers (running stats, etc.)
    for name, buffer in model.named_buffers():
        if 'shared_core_layers' not in name:
            non_moe_state_dict[name] = buffer.detach().cpu()
            logger.info(f"Saved to non_moe: {name}")

    # Save non-MoE parameters
    logger.info(f"Saving {len(non_moe_state_dict)} non-MoE parameters...")
    torch.save(non_moe_state_dict, output_path / "non_moe_weights.pt")

    # Extract and save compressed MoE layers
    layer_count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        # Check if this layer has compressed MoE
        moe_block = None
        if isinstance(getattr(layer, 'mlp', None), CompressedMoEBlock):
            moe_block = layer.mlp
        elif isinstance(getattr(layer, 'block_sparse_moe', None), CompressedMoEBlock):
            moe_block = layer.block_sparse_moe

        if moe_block is not None:
            layer_dir = output_path / f"layer_{layer_idx}"
            layer_dir.mkdir(exist_ok=True)
            compressed_layer_indices.append(layer_idx)

            # Save each projection's SharedCoreLayer
            for proj_name, shared_layer in moe_block.shared_core_layers.items():
                # Extract core and wrappers
                core = shared_layer.core.detach().cpu()
                wrappers = []

                for expert in shared_layer.experts:
                    wrappers.append({
                        "U_in": expert.input_wrapper.U.detach().cpu(),
                        "V_in": expert.input_wrapper.V.detach().cpu(),
                        "U_out": expert.output_wrapper.U.detach().cpu(),
                        "V_out": expert.output_wrapper.V.detach().cpu(),
                    })

                # Save
                save_data = {
                    "layer_idx": layer_idx,
                    "projection": proj_name,
                    "core": core,
                    "wrappers": wrappers,
                    "metadata": {
                        "num_experts": shared_layer.num_experts,
                        "d_in": shared_layer.d_in,
                        "d_out": shared_layer.d_out,
                        "rank": shared_layer.rank
                    }
                }

                proj_path = layer_dir / f"{proj_name}.pt"
                torch.save(save_data, proj_path)

            layer_count += 1

    # Save metadata about what was compressed
    save_metadata = {
        "compressed_layer_indices": compressed_layer_indices,
        "num_compressed_layers": layer_count,
        "total_layers": len(model.model.layers) if hasattr(model, 'model') else 0
    }

    with open(output_path / "compressed_metadata.json", 'w') as f:
        json.dump(save_metadata, f, indent=2)

    logger.info(f"Saved {layer_count} compressed MoE layers")
    logger.info(f"Saved {len(non_moe_state_dict)} non-MoE parameters")
    logger.info(f"Model saved to {output_path}")
    logger.info("="*10 + " Save Complete " + "="*10)


def export_to_hf_format(
    compressed_model_path: str,
    original_model_name: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
):
    """
    Export compressed model to standard HuggingFace format for evaluation.

    This materializes the compressed experts as full weight matrices,
    making the model compatible with standard HF tools (including lm_eval).

    Args:
        compressed_model_path: Path to compressed model
        original_model_name: Original model name
        output_path: Where to save the exported model
        device_map: Device mapping
        torch_dtype: Data type

    Returns:
        Path to exported model
    """
    logger.info(f"Exporting compressed model to HuggingFace format...")
    logger.info(f"  Input: {compressed_model_path}")
    logger.info(f"  Output: {output_path}")

    # Load compressed model
    model = load_compressed_model(
        compressed_dir=compressed_model_path,
        original_model_name=original_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype
    )

    # Load original model to get the structure
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # Materialize compressed experts as full weight matrices
    logger.info("Materializing compressed experts...")

    for layer_idx, layer in enumerate(model.model.layers):
        # Check if this layer has compressed MoE
        compressed_block = None
        target_attr = None

        if isinstance(getattr(layer, 'mlp', None), CompressedMoEBlock):
            compressed_block = layer.mlp
            target_attr = 'mlp'
        elif isinstance(getattr(layer, 'block_sparse_moe', None), CompressedMoEBlock):
            compressed_block = layer.block_sparse_moe
            target_attr = 'block_sparse_moe'

        if compressed_block is not None:
            logger.info(f"  Materializing layer {layer_idx}...")

            # Get the original structure
            original_layer = original_model.model.layers[layer_idx]
            original_moe = getattr(original_layer, target_attr)

            # For each expert, materialize the weights
            num_experts = compressed_block.num_experts

            for expert_idx in range(num_experts):
                # Get the original expert structure
                original_expert = original_moe.experts[expert_idx]

                # Materialize weights for each projection
                for proj_name, shared_layer in compressed_block.shared_core_layers.items():
                    # Reconstruct full weight matrix
                    reconstructed_weight = shared_layer.get_expert(expert_idx).reconstruct_weight()

                    # Assign to original expert
                    if hasattr(original_expert, proj_name):
                        original_expert_proj = getattr(original_expert, proj_name)
                        original_expert_proj.weight.data = reconstructed_weight.to(
                            device=original_expert_proj.weight.device,
                            dtype=original_expert_proj.weight.dtype
                        )

            # Replace compressed block with materialized original structure
            setattr(layer, target_attr, original_moe)

    # Copy over all other parameters (attention, norms, etc.) from compressed model
    # These should already be in the original model structure we loaded
    logger.info("Copying non-MoE parameters...")

    compressed_state = model.state_dict()
    original_state = original_model.state_dict()

    for name in original_state.keys():
        # Skip expert weights (already materialized)
        if 'experts.' in name and ('.mlp.' in name or '.block_sparse_moe.' in name):
            continue

        # Copy from compressed model if available
        if name in compressed_state:
            original_state[name] = compressed_state[name]

    original_model.load_state_dict(original_state, strict=False)

    # Save in standard HuggingFace format
    logger.info(f"Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    original_model.save_pretrained(output_path)

    # Also copy tokenizer if it exists
    compressed_path = Path(compressed_model_path)
    if (compressed_path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(compressed_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

    logger.info(f"Export complete! Model saved to {output_path}")
    logger.info("This model can now be used with standard HuggingFace tools and lm_eval")

    return str(output_path)
