from transformers import AutoModelForCausalLM, AutoConfig
import torch 

def get_model(model_name: str):
    
    #regex match for Qwen-3 MoEs are Qwen/Qwen3-*B-A*B
    
    import re
    if re.match(r"Qwen3?/\w+-\d+B-A\d+B", model_name):
        from src.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
        
        config = AutoConfig.from_pretrained(model_name,
                                            dtype=torch.float16,
                                            device_map="cpu",
                                            trust_remote_code=True)
        return Qwen3MoeForCausalLM, config