"""Configuration loading with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED = ("horizon", "n_folds", "n_calib_folds", "models", "data")


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config {path} is missing keys: {missing}")
    if cfg["n_calib_folds"] >= cfg["n_folds"]:
        raise ValueError("n_calib_folds must be smaller than n_folds (need at least one test fold)")
    if cfg["data"].get("source", "synthetic") not in {"synthetic", "m5"}:
        raise ValueError("data.source must be 'synthetic' or 'm5'")
    cfg.setdefault("seed", 0)
    cfg.setdefault("neural_steps", 100)
    cfg.setdefault("foundation", False)
    cfg.setdefault("decision", {"cu_ratio": 0.4, "co_ratio": 0.1})
    cfg.setdefault("causal", {})
    return cfg
