import os
import re
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.ai_engine import get_ai_engine


@dataclass
class SampleResult:
    file: str
    true_ng_ml: float
    pred_ng_ml: float
    abs_err_ng_ml: float
    rel_err_pct: float
    spec_mae: float
    spec_rmse: float
    spec_corr: float


def _read_spectrum(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=0)
    elif ext in [".txt", ".tsv"]:
        df = pd.read_csv(file_path, sep=None, engine="python")
    else:
        df = pd.read_csv(file_path)

    preferred_cols = {"absorbance", "intensity", "signal", "y"}
    for col in df.columns:
        if str(col).strip().lower() in preferred_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if vals.size > 0:
                return vals.astype(np.float32)

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
    if not valid_cols:
        raise ValueError(f"No numeric spectrum data in file: {file_path}")
    vals = numeric_df[valid_cols[-1]].dropna().values
    if vals.size == 0:
        raise ValueError(f"Empty numeric spectrum column in file: {file_path}")
    return vals.astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.sqrt(np.sum(a0 ** 2) * np.sum(b0 ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a0 * b0) / denom)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    if sst <= 0:
        return float("nan")
    return float(1.0 - (sse / sst))


def _parse_true_concentration(file_name: str) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*-\s*cea", file_name.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*ng/ml", file_name.lower())
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse concentration from file name: {file_name}")


def evaluate(test_dir: str) -> pd.DataFrame:
    engine = get_ai_engine()
    if not engine.is_loaded:
        raise RuntimeError("AI engine failed to load models.")

    rows: List[SampleResult] = []
    for name in sorted(os.listdir(test_dir)):
        if not name.lower().endswith((".xlsx", ".xls", ".csv", ".txt", ".tsv")):
            continue

        true_c = _parse_true_concentration(name)
        spectrum = _read_spectrum(os.path.join(test_dir, name))
        pred = engine.predict_spectrum_from_spectrum(spectrum)

        pred_c = float(pred["pred_concentration"])
        y_in = np.asarray(pred["input_resampled"], dtype=np.float64)
        y_hat = np.asarray(pred["pred_spectrum"], dtype=np.float64)

        abs_err = abs(pred_c - true_c)
        rel_err = (abs_err / true_c * 100.0) if true_c > 0 else float("nan")
        spec_mae = float(np.mean(np.abs(y_in - y_hat)))
        spec_rmse = float(np.sqrt(np.mean((y_in - y_hat) ** 2)))
        spec_corr = _corr(y_in, y_hat)

        rows.append(
            SampleResult(
                file=name,
                true_ng_ml=true_c,
                pred_ng_ml=pred_c,
                abs_err_ng_ml=abs_err,
                rel_err_pct=rel_err,
                spec_mae=spec_mae,
                spec_rmse=spec_rmse,
                spec_corr=spec_corr,
            )
        )

    if not rows:
        raise RuntimeError(f"No valid files found under: {test_dir}")

    return pd.DataFrame([r.__dict__ for r in rows]).sort_values("true_ng_ml").reset_index(drop=True)


def make_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 2, 10, 50, float("inf")]
    labels = ["(0,2]", "(2,10]", "(10,50]", "(50,+inf)"]
    seg = df.copy()
    seg["segment"] = pd.cut(seg["true_ng_ml"], bins=bins, labels=labels, include_lowest=True, right=True)
    out = (
        seg.groupby("segment", observed=True)
        .agg(
            count=("file", "count"),
            mae_ng_ml=("abs_err_ng_ml", "mean"),
            mape_pct=("rel_err_pct", "mean"),
            median_abs_err=("abs_err_ng_ml", "median"),
            median_rel_err_pct=("rel_err_pct", "median"),
        )
        .reset_index()
    )
    return out


def save_bland_altman(df: pd.DataFrame, out_path: str) -> None:
    y_true = df["true_ng_ml"].to_numpy(dtype=np.float64)
    y_pred = df["pred_ng_ml"].to_numpy(dtype=np.float64)
    mean_vals = (y_true + y_pred) / 2.0
    diff_vals = y_pred - y_true
    bias = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    plt.figure(figsize=(8, 5))
    plt.scatter(mean_vals, diff_vals, s=45, alpha=0.85)
    plt.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias={bias:.3f}")
    plt.axhline(loa_upper, color="gray", linestyle=":", linewidth=1.5, label=f"+1.96SD={loa_upper:.3f}")
    plt.axhline(loa_lower, color="gray", linestyle=":", linewidth=1.5, label=f"-1.96SD={loa_lower:.3f}")
    plt.title("Bland-Altman: Predicted - True Concentration")
    plt.xlabel("Mean of (True, Predicted) [ng/ml]")
    plt.ylabel("Difference (Predicted - True) [ng/ml]")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_true_vs_pred(df: pd.DataFrame, out_path: str) -> None:
    y_true = df["true_ng_ml"].to_numpy(dtype=np.float64)
    y_pred = df["pred_ng_ml"].to_numpy(dtype=np.float64)
    xy_min = min(np.min(y_true), np.min(y_pred))
    xy_max = max(np.max(y_true), np.max(y_pred))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=55, alpha=0.9)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "k--", linewidth=1.2, label="Ideal y=x")
    for _, row in df.iterrows():
        plt.annotate(
            row["file"].replace(".xlsx", ""),
            (row["true_ng_ml"], row["pred_ng_ml"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.title("True vs Predicted Concentration")
    plt.xlabel("True [ng/ml]")
    plt.ylabel("Predicted [ng/ml]")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_metrics_report(df: pd.DataFrame, seg_df: pd.DataFrame, out_path: str) -> None:
    y_true = df["true_ng_ml"].to_numpy(dtype=np.float64)
    y_pred = df["pred_ng_ml"].to_numpy(dtype=np.float64)
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / y_true * 100.0

    report_lines = [
        "Evaluation Report: test-predict",
        "",
        f"Samples: {len(df)}",
        f"MAE (ng/ml): {np.mean(abs_err):.6f}",
        f"RMSE (ng/ml): {np.sqrt(np.mean((y_pred - y_true) ** 2)):.6f}",
        f"Median AE (ng/ml): {np.median(abs_err):.6f}",
        f"MAPE (%): {np.mean(rel_err):.6f}",
        f"Median APE (%): {np.median(rel_err):.6f}",
        f"R2: {_r2_score(y_true, y_pred):.6f}",
        f"Pearson r: {np.corrcoef(y_true, y_pred)[0,1]:.6f}",
        "",
        "Segment Errors (by true concentration):",
        seg_df.to_string(index=False),
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_dir = os.path.join(root, "data", "test-predict")
    out_dir = os.path.join(root, "outputs", "eval_test_predict")
    os.makedirs(out_dir, exist_ok=True)

    df = evaluate(test_dir)
    seg_df = make_segment_table(df)

    detail_csv = os.path.join(out_dir, "detail_metrics.csv")
    seg_csv = os.path.join(out_dir, "segment_metrics.csv")
    report_txt = os.path.join(out_dir, "summary_report.txt")
    bland_png = os.path.join(out_dir, "bland_altman.png")
    scatter_png = os.path.join(out_dir, "true_vs_pred.png")

    df.to_csv(detail_csv, index=False)
    seg_df.to_csv(seg_csv, index=False)
    save_metrics_report(df, seg_df, report_txt)
    save_bland_altman(df, bland_png)
    save_true_vs_pred(df, scatter_png)

    print(f"Saved detail metrics: {detail_csv}")
    print(f"Saved segment metrics: {seg_csv}")
    print(f"Saved report:         {report_txt}")
    print(f"Saved Bland-Altman:   {bland_png}")
    print(f"Saved scatter plot:   {scatter_png}")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
