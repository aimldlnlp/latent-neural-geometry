from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_flat_metrics(path_csv: str | Path, path_json: str | Path, payload: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for group, metrics in payload.items():
        for name, value in metrics.items():
            rows.append({"group": group, "metric": name, "value": float(value)})
    frame = pd.DataFrame(rows).sort_values(["group", "metric"]).reset_index(drop=True)
    frame.to_csv(path_csv, index=False)
    save_json(path_json, payload)


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std

