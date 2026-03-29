import yaml
import os
import argparse
import torch

DEFAULTS = {
    "sorscher_compatible": False,
}

def load_config(config_path: str):
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)
    options = argparse.Namespace(**cfg)
    options.device = torch.device(options.device)
    options.dtype = getattr(torch, options.dtype)
    return options