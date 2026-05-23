import yaml
import os
from typing import Any, Dict

def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML file safely with UTF-8 encoding.
    """
    # Try finding it relative to project root
    if not os.path.exists(filepath):
        # Try finding relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        filepath = os.path.join(project_root, filepath)
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_base_config() -> Dict[str, Any]:
    return load_yaml("configs/base_config.yaml")

def get_qdrant_config() -> Dict[str, Any]:
    return load_yaml("configs/qdrant_config.yaml")

def get_prompts_config() -> Dict[str, Any]:
    return load_yaml("configs/prompts.yaml")
