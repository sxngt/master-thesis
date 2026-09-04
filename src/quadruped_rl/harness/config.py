"""Hierarchical YAML config composition.

Composition order (later overrides earlier, deep-merged):
    default <- sim <- algorithm <- robot <- terrain <- reward <- coach <- experiment <- CLI

The `sim` group selects the simulation backend — a first-class experiment
axis: the same algorithms are trained in Isaac Gym and cross-validated in
PyBullet/Gazebo (scripts/cross_validate.py).
Every run MUST persist its fully resolved config (reproducibility requirement).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins). Returns a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_inherits(cfg: dict, root: Path) -> dict:
    """Resolve {inherit: <relative path>} nodes by loading and merging the target."""
    if isinstance(cfg, dict):
        if set(cfg) == {"inherit"}:
            return _resolve_inherits(load_yaml(root / cfg["inherit"]), root)
        return {k: _resolve_inherits(v, root) for k, v in cfg.items()}
    return cfg


def compose_config(
    sim: str | None = None,
    algorithm: str | None = None,
    robot: str | None = None,
    terrain: str | None = None,
    reward: str | None = None,
    coach: str | None = None,
    experiment: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_root: str | Path = CONFIG_ROOT,
) -> dict[str, Any]:
    root = Path(config_root)
    cfg = load_yaml(root / "default.yaml")
    for group, name in [
        ("sim", sim),
        ("algorithm", algorithm),
        ("robot", robot),
        ("terrain", terrain),
        ("reward", reward),
        ("coach", coach),
        ("experiment", experiment),
    ]:
        if name:
            cfg = deep_merge(cfg, load_yaml(root / group / f"{name}.yaml"))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return _resolve_inherits(cfg, root)


def save_resolved_config(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out
