"""
utils/config.py
---------------
YAML configuration loader. Returns a nested namespace object supporting
dot-access (cfg.training.lr) and dict-access (cfg['training']['lr']).

Usage:
    from utils.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.training.lr)
"""

import yaml
from types import SimpleNamespace
from typing import Any


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def _namespace_to_dict(ns: Any) -> Any:
    """Recursively convert a SimpleNamespace back to a plain dict."""
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path: str) -> SimpleNamespace:
    """Load a YAML config file and return as a nested SimpleNamespace.

    Args:
        path: Path to the YAML config file.

    Returns:
        Nested SimpleNamespace with all config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML file is empty or malformed.
    """
    if not path:
        raise FileNotFoundError("Config path is empty.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    return _dict_to_namespace(raw)


def config_to_dict(cfg: SimpleNamespace) -> dict:
    """Convert a config namespace back to a plain dict (for saving to JSON/YAML).

    Args:
        cfg: SimpleNamespace config object.

    Returns:
        Plain nested dict.
    """
    return _namespace_to_dict(cfg)
