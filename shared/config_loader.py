from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} placeholders with environment variable values."""
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            result = os.environ.get(var_name, "")
            return result
        return _ENV_VAR_RE.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file and substitute ${ENV_VAR} placeholders
    with the corresponding environment variable values.

    Usage:
        cfg = load_config("config/pipeline.yaml")
        model = cfg["pipeline"]["embedding_model"]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _substitute_env_vars(raw or {})
