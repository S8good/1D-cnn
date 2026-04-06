from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def compute_mvr_from_predictions(true_values: Sequence[float], pred_values: Sequence[float]) -> float:
    true_arr = np.asarray(list(true_values), dtype=np.float32).reshape(-1)
    pred_arr = np.asarray(list(pred_values), dtype=np.float32).reshape(-1)
    if true_arr.size != pred_arr.size:
        raise ValueError("true_values and pred_values must have the same length")
    if true_arr.size < 2:
        return 0.0
    # Use a stable sort so repeated concentrations keep their original order.
    order = np.argsort(true_arr, kind="mergesort")
    sorted_pred = pred_arr[order]
    violations = sorted_pred[:-1] > sorted_pred[1:]
    return float(np.mean(violations.astype(np.float32)))


def summarise_values(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    return float(arr.mean()), float(arr.std(ddof=0))


def build_model_summary(rows: Sequence[dict], model_order: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if "model" not in df.columns:
        raise KeyError("rows must include a 'model' field")
    order_map = {name: idx for idx, name in enumerate(model_order)}
    df["_order"] = df["model"].map(order_map).fillna(len(order_map)).astype(int)
    df = df.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)
    return df
