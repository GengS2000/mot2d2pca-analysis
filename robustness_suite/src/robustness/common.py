from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def suite_root() -> Path:
    return ROOT


def output_root() -> Path:
    """Return the configured runtime output directory.

    Slurm exports an absolute path.  Local/manual runs may use a path relative
    to the suite root.
    """
    value = os.environ.get("ROBUSTNESS_OUTPUT_ROOT", "outputs")
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def numeric_time(labels) -> np.ndarray:
    values = []
    for label in labels:
        text = str(label)
        if text == "0":
            values.append(0.0)
        elif "-" in text:
            left, right = text.split("-", 1)
            values.append((float(left) + float(right)) / 2)
        else:
            values.append(float(text))
    return np.asarray(values, dtype=float)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    rmse = math.sqrt(float(np.mean((a - b) ** 2)))
    scale = float(np.std(a))
    return rmse / scale if scale > 0 else rmse


def atomic_json(path: Path, payload: dict) -> None:
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.floating) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
