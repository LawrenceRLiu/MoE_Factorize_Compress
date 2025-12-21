
import yaml
import os
import numpy as np
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import utils

import lm_eval
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM
import json


def eval_lm(
        base_model: str,
        model: AutoModelForCausalLM,
        tasks: list[str] = ["winogrande", "piqa", "hellaswag", "arc_easy", "arc_challenge"],
        num_fewshot: int = 0,
        log_wandb: bool = False,
        results_log_txt: str = None,
        apply_chat_template: bool = False,
        HLFM_kwargs: dict = {}):
    
    

    # Wrap in HFLM for compatibility with lm-eval
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    lm = HFLM(
        pretrained=model,  # Pass the preloaded model instance
        tokenizer=tokenizer,
        batch_size=5, #"auto",  # Adjust based on your hardware; use 'auto' for auto-detection
        parallelize=True,  # This is crucial for multi-GPU
        **HLFM_kwargs  # Additional kwargs for HFLM
    )

    # Run evaluation on desired tasks
    results = lm_eval.simple_evaluate(
        model=lm,  # Use the wrapped model
        tasks=tasks,  
        num_fewshot=num_fewshot,  
        batch_size=5,  # Matches the HFLM batch_size
        apply_chat_template=apply_chat_template,  # Use chat template if specified
    )
    
    print(make_table(results))
    if log_wandb:
        wandb.log({"lm_eval/results": make_table(results)})
    if results_log_txt is not None:
        with open(results_log_txt, "a") as f:
            f.write(make_table(results))
            f.write("\n\n")
        print("Results saved to:", results_log_txt)
    #save it as a json
    # if results_log_json is not None:
    #     os.makedirs(os.path.dirname(results_log_json), exist_ok=True)
    #     with open(results_log_json, "w") as f:
    #         json.dump(results, f, indent=4)
    



    # if results_log_yaml is not None:
    #     os.makedirs(os.path.dirname(results_log_yaml), exist_ok=True)
    #     with open(results_log_yaml, "w") as f:
    #         yaml.safe_dump(results, f, default_flow_style=False)
    
    return results

