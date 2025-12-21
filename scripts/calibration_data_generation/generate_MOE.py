import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np
import os
import sys

if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())


import yaml
from hydra.utils import instantiate

import src.compression_parent as compression_parent
from src.utils.model_utils import find_layers, load_model, inference_layer
import src.data as data
import src.utils.utils as utils
import copy
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers import AutoModelForCausalLM, AutoTokenizer    
try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def replace_with_compressed(
    layer: nn.Module,
    layer_name: str,
    log_object: str,
    save_weight: bool,
    save_weight_path: str,
    parameters_accounted_for: int,
):
    """Replace a linear layer with a CompressedLinear layer that logs hessian or hessian_diag.

    Args:
        layer (nn.Module): The layer containing the linear layer to replace.
        layer_name (str): The name of the linear layer to replace.
        log_object (str): What to log (hessian or hessian_diag).
        save_weight (bool): Whether to save the original weights.
        save_weight_path (str): Path to save the original weights.

    Returns:
        None
    """
    original_layer: nn.Linear = getattr(layer, layer_name)
    assert isinstance(original_layer, nn.Linear), f"Layer {layer_name} is not a nn.Linear"
    parameters_accounted_for += original_layer.weight.numel()
    if save_weight:
        save_weight_path_use = f"{save_weight_path}{layer_name}.pt"
        os.makedirs(os.path.dirname(save_weight_path_use), exist_ok=True)
        torch.save(
            {"weight": original_layer.weight, "bias": original_layer.bias},
            save_weight_path_use,
        )

    new_layer = compression_parent.CompressedLinear(
        original_layer.weight, original_layer.bias
    )

    if log_object == "hessian":
        new_layer.enable_hessian_logging()
    elif log_object == "hessian_diag":
        new_layer.enable_hessianDiag_logging()
    else:
        raise ValueError(f"Unknown log_object: {log_object}")

    setattr(layer, layer_name, new_layer)
    del original_layer  # Remove the original layer to free memory
    return layer, parameters_accounted_for

def save_compressed_log_object(
    layer: nn.Module,
    layer_name: str,
    log_object: str,
    save_path: str,
):
    """Save the logged calibration data from a CompressedLinear layer.

    Args:
        layer (nn.Module): The layer containing the CompressedLinear layer.
        layer_name (str): The name of the CompressedLinear layer.
        log_object (str): What to save (hessian or hessian_diag).
        save_path (str): Path to save the logged data.
        layer_idx (int): Index of the layer for naming purposes.
    Returns:
        None
    """
    projection = getattr(layer, layer_name)

    save_file_path = f"{save_path}{layer_name}.pt"
    if log_object == "hessian":
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        torch.save({"hessian": projection.hessian}, save_file_path)
    elif log_object == "hessian_diag":
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        #check that its all non-zero
        if torch.any(projection.hessianDiag == 0):
            print("WARNING: Hessian diag is all zero, likely this expert has not seen any tokens")
        # assert torch.all(projection.hessianDiag != 0), "Hessian diag has zero values!"
        torch.save({"hessianDiag": projection.hessianDiag}, save_file_path)

    projection.clean()

    

@torch.no_grad()
def generate_calibration_data(model: AutoModelForCausalLM,
                      tokens: torch.Tensor,
                      attention_mask: Union[torch.Tensor, None],
                      forward_batch_size: int,
                      log_object: str,
                      save_path: str,
                      save_weights: bool,
                      log_save_subdir: str):
    """Calculate and logs 

    Args:
        model (AutoModelForCausalLM): The model to use for inference. 
        tokens (torch.Tensor):
        Tokens to pass into the model of shape (n_samples, seqlen).
        attention_mask (Union[torch.Tensor, None]): Custom attention mask of
            shape (n_samples, seqlen, seqlen) or None if we just use the defualt
            mask 
        forward_batch_size (int): Batch size for forward pass
        log_object (str): What to generate/log (hessian or hessian_diag)
        save_path (str): Path to save everything
        save_weights (bool): Save the original weights
        log_save_subdir (str): Subdirectory to save the logs and calibration data.

    Raises:
        ValueError: If the log_object is not recognized.
    """
    
    model.eval()
    
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters_accounted_for = 0
    
    attention_projections = ["q_proj", "k_proj", "v_proj", "o_proj"]  #hard coded for now, may have to change later
    mlp_projections = ["up_proj", "gate_proj", "down_proj"]
    # Find the layers in the model 
    for i, layer in tqdm.tqdm(enumerate(model.model.layers)):
        # print(layer)
        for name in attention_projections:
            layer.self_attn, parameters_accounted_for = replace_with_compressed(
                layer=layer.self_attn,
                layer_name=name,
                log_object=log_object,
                save_weight=save_weights,
                save_weight_path=os.path.join(
                    save_path,
                    "original_weights",
                    "layer_{i}/self_attn.".format(i=i),
                ),
                parameters_accounted_for=parameters_accounted_for,
            )
        
        #if we are dealing with an MoE mlp 
        if isinstance(layer.mlp, modeling_qwen3_moe.Qwen3MoeSparseMoeBlock):
            for j, expert in enumerate(layer.mlp.experts):
                for name in mlp_projections:
                    layer.mlp.experts[j], parameters_accounted_for = replace_with_compressed(
                        layer=expert,
                        layer_name=name,
                        log_object=log_object,
                        save_weight=save_weights,
                        save_weight_path=os.path.join(
                            save_path,
                            "original_weights",
                            f"layer_{i}/mlp.expert_{j}."
                        ),
                        parameters_accounted_for=parameters_accounted_for,
                    )
        else:
            for name in mlp_projections:
                layer.mlp, parameters_accounted_for = replace_with_compressed(
                    layer=layer.mlp,
                    layer_name=name,
                    log_object=log_object,
                    save_weight=save_weights,
                    save_weight_path=os.path.join(
                        save_path,
                        "original_weights",
                        "layer_{i}/mlp.".format(i=i),
                    ),
                    parameters_accounted_for=parameters_accounted_for,
                )
            
    print(f"Total parameters: {total_parameters/10**9:.2f}B, Parameters accounted for: {parameters_accounted_for/10**9:.2f}B",
            f"Fraction accounted for: {parameters_accounted_for / total_parameters:.4f}")
    utils.clean()
    # forward pass the tokens
    for i in tqdm.tqdm(range(0, tokens.shape[0], forward_batch_size), desc="Generating calibration data"):
        # get the batch of tokens
        batch_tokens = tokens[i:i+forward_batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+forward_batch_size].unsqueeze(1)  # Ensure it's of shape (batch_size, 1, seqlen, seqlen)
        else:
            batch_attention_mask = None
        # forward pass
        model(input_ids=batch_tokens, attention_mask=batch_attention_mask)
        
    #get the logged calibration data from the model 
    save_file_path = os.path.join(
        save_path,
        log_save_subdir,
    )
    for i, l in enumerate(model.model.layers):
        for name in attention_projections:
            save_compressed_log_object(
                layer=l.self_attn,
                layer_name=name,
                log_object=log_object,
                save_path=os.path.join(
                    save_file_path,
                    f"layer_{i}/self_attn."
                ),
            )
        #if we are dealing with an MoE mlp
        if isinstance(l.mlp, modeling_qwen3_moe.Qwen3MoeSparseMoeBlock):
            for j, expert in enumerate(l.mlp.experts):
                for name in mlp_projections:
                    save_compressed_log_object(
                        layer=expert,
                        layer_name=name,
                        log_object=log_object,
                        save_path=os.path.join(
                            save_file_path,
                            f"layer_{i}/mlp.expert_{j}."
                        ),
                    )
        else:
            for name in mlp_projections:
                save_compressed_log_object(
                    layer=l.mlp,
                    layer_name=name,
                    log_object=log_object,
                    save_path=os.path.join(
                        save_file_path,
                        f"layer_{i}/mlp."
                    ),
                )
    
    

def main(model: str, dataset: str, seqlen: int, seed: int = 0, 
         n_samples: int = 128, forward_batch_size: int = 4, 
         log_object: str = "hessian_diag", save_path: str = "./", 
         save_weights: bool = False, save_calibration_data: bool = False,
         overwrite_existing: bool = False):
    """Main function that takes individual parameters instead of args object.
    
    Args:
        model (str): LLM model to load
        dataset (str): Path to the dataset config file
        seqlen (int): Sequence length, -1 for maximum sequence length
        seed (int): Seed for sampling the calibration data
        n_samples (int): Number of calibration data samples
        forward_batch_size (int): Batch size for forward pass
        log_object (str): What to generate/log (hessian or hessian_diag)
        save_path (str): Path to save everything
        save_weights (bool): Save the original weights
        save_calibration_data (bool): Save the calibration data for debugging
    """
    print(f"Model: {model}, Dataset: {dataset}, Seqlen: {seqlen}, Seed: {seed}")
    print(f"N_samples: {n_samples}, Forward_batch_size: {forward_batch_size}")
    print(f"Log_object: {log_object}, Save_path: {save_path}")
    print(f"Save_weights: {save_weights}, Save_calibration_data: {save_calibration_data}")
    
    # Set the seed
    utils.seed(seed, seed_all=True)

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,  # Use float16 for better performance on GPUs
        device_map="auto",  # Automatically map the model to available GPUs
        trust_remote_code=True,  # Trust remote code for models that require it
    )
    model_obj.seqlen = (
        seqlen if seqlen > 0 else model_obj.config.max_position_embeddings
    )
    
    # Path related stuff
    dataset_cfg = instantiate(
        yaml.safe_load(open(dataset, "r")),
        n_samples=n_samples,
        ctx_len=model_obj.seqlen,
    )
    save_path = os.path.join(save_path, model)
    log_save_subdir = os.path.join(
        log_object, data.create_save_dir(
            dataset_cfg, seed)  
        )

    print("seqlen", model_obj.seqlen)
    model_obj.eval()

    utils.seed(seed, seed_all=True)

    if log_object != "ignore":
        os.makedirs(os.path.join(save_path, log_save_subdir), exist_ok=True)
        # Save the raw config, ie the one without the n_samples and ctx_len
        #if the dataset config exists
        if os.path.exists(os.path.join(save_path, log_save_subdir, "dataset_config.yaml")):
            #if its the same we just end the function
            existing_cfg = yaml.safe_load(open(os.path.join(save_path, log_save_subdir,
                                                            "dataset_config.yaml"), "r"))
            new_cfg = yaml.safe_load(open(dataset, "r"))
            if existing_cfg == new_cfg and not overwrite_existing:
                print("Calibration data already exists, skipping generation.")
                return
        with open(os.path.join(save_path, log_save_subdir, "dataset_config.yaml"), "w") as f:
            yaml.dump(yaml.safe_load(open(dataset, "r")), f)
            
    traintokens, attention_mask, tokenizer = data.generate_calibration_data_single_source(
        dataset_cfg,
        model_name=model,
        seed=seed,
        verbose=True
    )
    
    # If we are saving the calibration data, save it now
    if save_calibration_data:
        calibration_data_save_path = os.path.join(
            save_path, log_save_subdir, "raw_data"
        )
        data.decode_calibration_data(
            traintokens, tokenizer, log_dir=calibration_data_save_path, skip_special_tokens=True
        )
    
    tick = time.time()

    generate_calibration_data(model_obj, traintokens, 
                      attention_mask,
                      forward_batch_size, log_object, save_path, save_weights,
                      log_save_subdir=log_save_subdir)
    print("total time taken:", time.time() - tick)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="LLM model to load",
                        required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the dataset config file",
        required=True
    )
    parser.add_argument("--seqlen", type=int,
                        help="Sequence length also refered to as ctx_len in the code. if -1, maximum sequence length is used",
                        required=True)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--n_samples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--forward_batch_size",
        type=int,
        default=4,
        help="Batch size for forward pass, parallel process this many sequences.",
    )
    parser.add_argument(
        "--log_object",
        type=str,
        default="hessian_diag",
        choices=["hessian", "hessian_diag"],
        help="What to generate/log, hessian or hessian_diag. or to just ignore and not log anything.",
    )
    parser.add_argument(
        "--save_path", type=str, default="./", help="Path to save everything"
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Save the original weights of the linear layers before replacing them with CompressedLinear.",
    )
    parser.add_argument(
        "--save_calibration_data",
        action="store_true",
        help="Save the calibration data used for generating the hessians. for debugging purposes.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Overwrite existing calibration data if it exists.",
    )
    args = parser.parse_args()
    
    # Call the main function with individual arguments
    main(
        model=args.model,
        dataset=args.dataset,
        seqlen=args.seqlen,
        seed=args.seed,
        n_samples=args.n_samples,
        forward_batch_size=args.forward_batch_size,
        log_object=args.log_object,
        save_path=args.save_path,
        save_weights=args.save_weights,
        save_calibration_data=args.save_calibration_data,
        overwrite_existing=args.overwrite_existing,
    )