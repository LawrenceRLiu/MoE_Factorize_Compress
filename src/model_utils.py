from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import re
from typing import Union, Dict, Optional
from pathlib import Path
import json

from src.compressed_moe import SharedCoreExperts, SharedCoreLayer
from src.utils import clean

PROJECTIONS = [
    "up_proj",
    "down_proj",
    "gate_proj"
]

def get_model(model_name: str):
    
    #regex match for Qwen-3 MoEs are Qwen/Qwen3-*B-A*B

    if re.match(r"Qwen3?/\w+-\d+B-A\d+B", model_name):
        from src.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
        
        config = AutoConfig.from_pretrained(model_name,
                                            dtype=torch.float16,
                                            device_map="cpu",
                                            trust_remote_code=True)
        return Qwen3MoeForCausalLM, config
    
    
@torch.no_grad()
def get_hf_equivalent_model(
    compressed_model_path: str,
    original_model_name: str,
    device_map: Union[str, Dict] = "auto",
    dtype: Optional[torch.dtype] = None
) -> AutoModelForCausalLM:
    """
    Convert a compressed MoE model to its HuggingFace equivalent.

    Args:
        compressed_model_path: Path to the compressed model directory
        original_model_name: Name of the original HuggingFace model
        output_path: Path to save the equivalent model
        device_map: Device map for loading the model
        dtype: Torch data type for the model
    Returns:
        HuggingFace equivalent model
    """
    
    # Load the compressed model
    compressed_model_class, _ = get_model(original_model_name)
    compressed_model = compressed_model_class.from_pretrained(
        compressed_model_path,
        device_map=device_map,
        dtype=dtype
    )

    # Create the equivalent HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        device_map=device_map,
        dtype=dtype
    )

    state_dict = compressed_model.state_dict()
    
    #for now we will only handle Qwen3 MoE models
    assert re.match(r"Qwen3?/\w+-\d+B-A\d+B", original_model_name), "Only Qwen3 MoE models are supported currently"
    # TODO: change this to handle other models as needed in the future
    
    #we will need to modify the state dict keys to match the hf model keys
    
    #first delete all the keys that have 'shared_core' in them
    keys_to_delete = [key for key in state_dict.keys() if 'shared_core' in key]
    for key in keys_to_delete:
        del state_dict[key]
    clean()
    
    #now we need to add the equivalent expert representations as keys 
    for layer_idx, layer in enumerate(compressed_model.model.layers):
        
        experts: SharedCoreExperts = layer.mlp.experts
        
        for name in PROJECTIONS:
            
            if name == "up_proj":
                shared_layer = experts.up_shared_core
            elif name == "down_proj":
                shared_layer = experts.down_shared_core
            elif name == "gate_proj":
                shared_layer = experts.gate_shared_core
                
            equivalent_weights = shared_layer.reconstruct_experts()
            
            key_prefix = f"model.layers.{layer_idx}.mlp.experts"
            for expert_idx, weights in enumerate(equivalent_weights):
                key = f"{key_prefix}.{expert_idx}.{name}.weight"
                state_dict[key] = weights
            
            #if we are doing bias 
            if shared_layer.has_bias:
                raise NotImplementedError("Bias handling not implemented yet.")
                # TODO: implement bias handling if needed
            
    #load the modified state dict into the hf model
    hf_model.load_state_dict(state_dict, strict=True)
    del compressed_model
    del state_dict
    clean()
    
    return hf_model


def load_compressed_model(
    compressed_dir: str,
    original_model_name: str,
    device_map: Union[str, Dict] = "auto",
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True
):
    """
    Load a compressed MoE model from a checkpoint directory.

    Args:
        compressed_dir: Path to the compressed model checkpoint directory
        original_model_name: Name of the original HuggingFace model
        device_map: Device map for loading the model (use None for FSDP)
        dtype: Torch data type for the model
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded compressed model
    """
    compressed_path = Path(compressed_dir)

    # Verify the checkpoint exists
    if not compressed_path.exists():
        raise ValueError(f"Compressed model directory not found: {compressed_dir}")

    # Load the compressed model using the appropriate model class
    compressed_model_class, _ = get_model(original_model_name)

    # Load the model
    model = compressed_model_class.from_pretrained(
        str(compressed_path),
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code
    )

    return model


def save_compressed_model(
    model,
    output_dir: str,
    tokenizer: Optional[AutoTokenizer] = None
):
    """
    Save a compressed MoE model to a directory.

    Args:
        model: The compressed model to save
        output_dir: Directory to save the model
        tokenizer: Optional tokenizer to save alongside the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the model using HuggingFace's save_pretrained
    model.save_pretrained(str(output_path))

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))

    print(f"Compressed model saved to: {output_path}")
