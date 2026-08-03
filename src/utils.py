import random
import numpy as np
import torch
import yaml

def load_config(config_path):
    """Loads config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    """Sets seed to guarantee determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
