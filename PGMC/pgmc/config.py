"""YAML configuration loading for PGMC."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_pgmc_config(path: str, benchmark: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("PGMC config not found: {}".format(config_path))

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    common = raw.get("common", {})
    benchmark_cfg = raw.get("benchmarks", {}).get(benchmark, {})
    if not isinstance(common, dict) or not isinstance(benchmark_cfg, dict):
        raise ValueError("PGMC config sections must be YAML mappings")
    return _deep_merge(common, benchmark_cfg)

