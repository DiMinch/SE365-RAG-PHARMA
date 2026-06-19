import yaml
import os
from typing import Any, Dict

def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load a YAML file safely with UTF-8 encoding."""
    if not os.path.exists(filepath):
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
    """
    Load Qdrant config from YAML, then overlay environment variables.

    Priority (highest to lowest):
      1. QDRANT_URL + QDRANT_API_KEY  → Qdrant Cloud (teamwork)
      2. QDRANT_HOST + QDRANT_PORT    → Local Docker
      3. YAML defaults (localhost:6333)
    """
    cfg = load_yaml("configs/qdrant_config.yaml")

    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_host = os.getenv("QDRANT_HOST", "").strip()
    qdrant_port = os.getenv("QDRANT_PORT", "").strip()

    if qdrant_url and qdrant_api_key:
        # ── Qdrant Cloud mode ─────────────────────────────────────────
        cfg["connection"]["url"] = qdrant_url
        cfg["connection"]["api_key"] = qdrant_api_key
        # Remove host/port so QdrantClient uses url instead
        cfg["connection"].pop("host", None)
        cfg["connection"].pop("port", None)
    else:
        # ── Local mode ────────────────────────────────────────────────
        if qdrant_host:
            cfg["connection"]["host"] = qdrant_host
        if qdrant_port:
            cfg["connection"]["port"] = int(qdrant_port)
        cfg["connection"].pop("url", None)
        cfg["connection"].pop("api_key", None)

    return cfg

def get_prompts_config() -> Dict[str, Any]:
    return load_yaml("configs/prompts.yaml")
