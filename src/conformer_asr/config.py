"""Shared configuration loader."""
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_all(directory: str | Path = "configs") -> dict:
    return {name: load_config(Path(directory) / f"{name}.yaml")
            for name in ("data", "model", "training", "evaluation")}
