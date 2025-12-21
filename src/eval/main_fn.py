import torch
import os
import yaml
import tqdm
import sys 
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
from src.utils.data_old import get_loaders
from src.eval.ppl import ppl_eval_basic
from src.eval.task_eval import eval_lm
# from src.eval.zero_shot import zero_shot


@torch.no_grad()
def eval(model, cfg):

    # model.eval()
    # ppl eval
    if hasattr(cfg.eval, "ppl_dataset"):
        raise NotImplementedError("PPL evaluation is not implemented yet.")
        if len(cfg.eval.ppl_dataset) == 0:
            print("No ppl datasets specified, skipping ppl eval")
        else:
            for dataset in cfg.eval.ppl_dataset:
                # seed(cfg.seed, seed_all = True)
                testloader = get_loaders(
                    dataset,
                    nsamples=0,
                    seqlen=model.seqlen,
                    model=cfg.base_model,
                    train_test="test",
                )

                ppl_eval_basic(
                    model=model,
                    testenc=testloader,
                    dataset_name=dataset,
                    results_log_txt=os.path.join(cfg.save_path, "results.txt"),
                    log_wandb=cfg.log_wandb,
                )
        with open(os.path.join(cfg.save_path, "results.txt"), "a") as f:
            f.write("PPL evaluation completed.\n\n")
            print("PPL evaluation completed.")

    # zero shot eval
    if hasattr(cfg.eval, "lm_eval_tasks"):
        #we expect the lm_eval tasks
        for task_shot in cfg.eval.lm_eval_tasks:
            tasks = [task_shot["task"]]
            num_fewshot = task_shot["num_fewshot"]
            print(f"Evaluating {tasks} with {num_fewshot} few-shot examples")
            output_path = os.path.join(cfg.save_path, f"results.txt")
            with open(output_path, "a") as f:
                f.write(f"Evaluating {tasks} with {num_fewshot} few-shot examples\n")
            eval_lm(
                base_model=cfg.base_model,
                model=model,
                tasks=tasks,
                num_fewshot=num_fewshot,
                log_wandb=cfg.log_wandb,
                results_log_txt= output_path,
                apply_chat_template=cfg.eval.get("apply_chat_template", False),
            )


if __name__ == "__main__":
    from src.utils.utils import recursive_apply
    from transformers import AutoConfig, AutoModelForCausalLM
    from omegaconf import OmegaConf, DictConfig
    # from src.model.qwen3 import Qwen3ForCausalLM
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a model on various tasks.")
    parser.add_argument("--config", type=str, required=False, help="Path to the evaluation configuration file.",
                        default="config/eval/leaderboard_eval.yaml")
    parser.add_argument("--model_name", type=str, required=False, help="Name of the model to evaluate.",
                        default="google/gemma-2-9b-it")
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the evaluation results.",
                        default="/data/lliu/PermPrune/models/")
    
    args = parser.parse_args()
    model_name = args.model_name
    config_path = args.config
    save_path = args.save_path
    print(f"Loading model {model_name} from {config_path} and saving results to {save_path}")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     # dtype=torch.float16,
    #     device_map="auto"
    # )
    save_path = f"/data/lliu/PermPrune/models/{model_name}/base"

    # from src.model.qwen3 import Qwen3ForCausalLM
    # model = Qwen3ForCausalLM.from_pretrained(
    #     "/data/lliu/PermPrune/models/Qwen/Qwen3-4B/ft/64_24/model/final_model",
    #     device_map="auto")
    # model.to(torch.float16)
    # save_path = f"/data/lliu/PermPrune/models/Qwen/Qwen3-4B/ft/64_24/model"
    # recursive_apply(model, "cache_reconstruct", {"denormalize": True, "offload": True})
    
    # model.seqlen = 4096
    os.makedirs(save_path, exist_ok=True)
    leaderboard_eval = yaml.safe_load(open("config/eval/leaderboard_eval.yaml", "r"))
    leaderboard_eval["apply_chat_template"] = False  
    cfg = DictConfig({"eval": leaderboard_eval,
                        "base_model": model_name,
                        "save_path": save_path,
                        "log_wandb": False})
    
    eval(model_name, cfg)