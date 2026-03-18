import yaml
import os
import argparse

def load_config(config_path: str):
    assert os.path.exists(config_path), f"Config file not found at {config_path}"   
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    options = argparse.Namespace(**cfg)
    return options