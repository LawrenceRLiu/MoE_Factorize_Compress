import wandb 
from dataclasses import dataclass, asdict
from typing import Dict
from omegaconf import OmegaConf

# all the stuff we would need to reinitialize a wandb run
@dataclass
class WandBConfig:
    project: str
    run_name: str
    run_id: str
    
    
    def to_dict(self) -> dict:
        return asdict(self)
