"""
Script to generate non-compressed models with the same weights as compressed models for evaluation.

This script extracts the non-compressed model generation functionality from ParallelCompress.py
and SequentialCompress.py into a standalone utility.
"""

import torch
import torch.nn as nn
import os
import tqdm
import argparse
import yaml
from typing import Dict, List
import copy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


import sys

if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())
from src.utils.model_utils import get_compressed_model_class
from src.compression_parent import CompressedLinear
from src.utils.utils import clean, recursive_find

def find_all_CompressedLinear(model) -> List[str]:
    """recusrively finds all instances of CompressedLinear in model"""
    found = []
    for name, module in model.named_children():
        if isinstance(module, CompressedLinear):
            found.append(name)
        else:
            f = find_all_CompressedLinear(module)
            found.extend([f"{name}.{n}" for n in f])
    return found



@torch.no_grad()
def generate_non_compressed_model(
    model_name: str,
    path:str
):
    """
    Generate a non-compressed model with the same weights as the compressed model.
    
    Args:
        base_model (str): Name/path of the base model
        save_path (str): Path where the compressed model is saved
        config_path (str): Path to the compression config file
        checkpoints_dict (Dict[str, str]): Dictionary mapping layer names to checkpoint paths
        output_dir (str): Name of the output directory within save_path
    """
    print("Creating a non-compressed model with the same weights for evaluation")
    
    
    non_compressed_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
    )
        
    # config = AutoConfig.from_pretrained(model_name)
    # non_compressed_model = AutoModelForCausalLM.from_config(
    #     config,
    #     torch_dtype=torch.bfloat16
    # )
    print("Non-compressed model created")
    compressed_model = get_compressed_model_class(model_name).from_pretrained(
        os.path.join(path, "model"),
        device_map="cpu"
    )
    # print("compressed_model state dict keys:",
    #       compressed_model.state_dict().keys())
    # print("non compressed model keys",
    #       non_compressed_model.state_dict().keys())
    layer_names = find_all_CompressedLinear(compressed_model)
    # print("layer names found:", layer_names)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    non_compressed_state_dict = non_compressed_model.state_dict()
    compressed_state_dict = compressed_model.state_dict()
    new_state_dict = copy.deepcopy(compressed_state_dict)
    
    for name in tqdm.tqdm(layer_names, desc="Processing layers"):
        uncompressed_layer:nn.Linear = recursive_find(non_compressed_model, name)
        compressed_layer: CompressedLinear = recursive_find(compressed_model, name)
        non_com_state_dict_keys = [k for k in non_compressed_state_dict.keys() if k.startswith(name)]
        com_state_dict_keys = [k for k in compressed_state_dict.keys() if k.startswith(name)]
        
        #if we have a bias in the compressed state dict
        #assert that there is a bias key in the compressed state dict
        if f"{name}.bias" in com_state_dict_keys:
            assert f"{name}.bias" in non_com_state_dict_keys
        
        compressed_layer.to(dev)
        
        weight = compressed_layer.reconstruct(denormalize=True).to(
                dtype=torch.bfloat16,
                device=uncompressed_layer.weight.data.device
            )

        compressed_layer.to("cpu")
        
        #remove all the com_state dict keys from the new state dict
        for key in com_state_dict_keys:
            if key != f"{name}.bias":
                #remove the key
                del new_state_dict[key]
        
        #add a key about the weight
        new_state_dict[f"{name}.weight"] = weight


    #save the non-compressed model
    
    non_compressed_model.load_state_dict(new_state_dict, strict=True)
    non_compressed_model.to("cpu")
    non_compressed_model.to(dtype=torch.bfloat16)
    # raise ValueError("stop here")
    output_path = os.path.join(path, "non_compressed_model")
    non_compressed_model.save_pretrained(
        output_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)




def main():
    parser = argparse.ArgumentParser(
        description="Generate non-compressed model from compressed checkpoints"
    )
    parser.add_argument("model_name", help="Base model name/path")
    parser.add_argument("path", help="Path to the directory containing the compressed_model")

    args = parser.parse_args()
    
    
    generate_non_compressed_model(
        model_name=args.model_name,
        path=args.path
    )


if __name__ == "__main__":
    main()