from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.paper_figure_utils import build_model_summary, compute_mvr_from_predictions, summarise_values


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

STAGE2_DIR = OUTPUTS_DIR / "stage2_regressor_3seed_20260326_143033"
SEEDS = [20260325, 20260331, 20260407]

A_METRICS = OUTPUTS_DIR / "stage0" / "split_test_metrics_predictor_v1.csv"
A_PREDS = OUTPUTS_DIR / "stage0" / "split_test_predictions_predictor_v1.csv"
B_METRICS = OUTPUTS_DIR / "stage0" / "split_test_metrics_predictor_v2.csv"
B_PREDS = OUTPUTS_DIR / "stage0" / "split_test_predictions_predictor_v2.csv"

OUT_MVR_CSV = OUTPUTS_DIR / "mvr_summary.csv"
OUT_MVR_FIG = OUTPUTS_DIR / "mvr_comparison_figure.png"
OUT_CH_SUMMARY = OUTPUTS_DIR / "c_hill_comparison_summary.csv"
OUT_CH_DETAIL = OUTPUTS_DIR / "c_hill_seed_detail.csv"
OUT_CH_FIG = OUTPUTS_DIR / "c_hill_comparison_figure.png"
OUT_CH_3SEED_SUMMARY = OUTPUTS_DIR / "c_hill_3seed_summary.csv"

MODEL_COLORS = {
    "Model A": "#7B7B7B",
    "Model B": "#4C72B0",
    "Model C": "#DD8452",
    "C+Cycle": "#8172B2",
    "C+Hill": "#C49A44",
    "Model D": "#55A868",
}

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def _read_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _single_seed_row(model: str, metrics_path: Path, preds_path: Path) -> dict:
    metrics = _read_metrics(metrics_path).iloc[0]
    preds = _read_predictions(preds_path)
    mvr = compute_mvr_from_predictions(preds["true_ng_ml"], preds["pred_ng_ml"])
    row = {
        "model": model,
        "seed": int(metrics.get("seed", 0) or 0),
        "mae_ng_ml": float(metrics["mae_ng_ml"]),
        "rmse_ng_ml": float(metrics["rmse_ng_ml"]),
        "mape_pct": float(metrics["mape_pct"]),
        "r2": float(metrics["r2"]),
        "mvr": float(mvr),
        "hill_mae_nm": float(metrics["hill_consistency_mae_nm"]) if "hill_consistency_mae_nm" in metrics.index else math.nan,
    }
    return row


def _stage2_seed_paths(stage_name: str, seed: int) -> tuple[Path, Path]:
    seed_dir = STAGE2_DIR / stage_name / f"seed_{seed}"
    return seed_dir / "metrics.csv", seed_dir / "predictions.csv"


def _stage3_seed_paths(run_tag: str, seed: int) -> tuple[Path, Path]:
    run_dir = OUTPUTS_DIR / f"{run_tag}_seed{seed}"
    metric_name = f"split_test_metrics_predictor_v2_{run_tag}.csv"
    pred_name = f"split_test_predictions_predictor_v2_{run_tag}.csv"
    return run_dir / metric_name, run_dir / pred_name


def _collect_seed_rows() -> list[dict]:
    rows = [
        _single_seed_row("Model A", A_METRICS, A_PREDS),
        _single_seed_row("Model B", B_METRICS, B_PREDS),
    ]
    for seed in SEEDS:
        c_metrics, c_preds = _stage2_seed_paths("stage1", seed)
        rows.append(_single_seed_row("Model C", c_metrics, c_preds))
        cycle_metrics, cycle_preds = _stage2_seed_paths("stage2", seed)
        rows.append(_single_seed_row("C+Cycle", cycle_metrics, cycle_preds))
        ch_metrics, ch_preds = _stage3_seed_paths("stage3_ch_fixed_regressor", seed)
        rows.append(_single_seed_row("C+Hill", ch_metrics, ch_preds))
        d_metrics, d_preds = _stage3_seed_paths("stage3_3c_learnable_regressor", seed)
        rows.append(_single_seed_row("Model D", d_metrics, d_preds))
    return rows


def _summarise_model(rows: list[dict], model: str, seeds_label: str) -> dict:
    model_rows = [row for row in rows if row["model"] == model]
    metrics = {}
    for key in ["mae_ng_ml", "rmse_ng_ml", "mape_pct", "r2", "mvr", "hill_mae_nm"]:
        values = [row[key] for row in model_rows if not math.isnan(row[key])]
        if values:
            mean, std = summarise_values(values)
        else:
            mean, std = math.nan, math.nan
        metrics[f"{key}_mean"] = mean
        metrics[f"{key}_std"] = std
    metrics["model"] = model
    metrics["seeds"] = seeds_label
    return metrics


def _write_mvr_outputs(summary_df: pd.DataFrame):
    model_order = ["Model A", "Model B", "Model C", "C+Cycle", "C+Hill", "Model D"]
    out_df = build_model_summary(
        [
            {
                "model": row["model"],
                "seeds": row["seeds"],
                "mvr_mean": row["mvr_mean"],
                "mvr_std": row["mvr_std"],
            }
            for _, row in summary_df.iterrows()
        ],
        model_order,
    )
    out_df.to_csv(OUT_MVR_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(out_df))
    means = out_df["mvr_mean"].to_numpy(dtype=float)
    stds = np.nan_to_num(out_df["mvr_std"].to_numpy(dtype=float), nan=0.0)
    colors = [MODEL_COLORS[name] for name in out_df["model"]]
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
    for idx, mean in enumerate(means):
        ax.text(idx, mean + max(means) * 0.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    best_idx = int(np.argmin(means))
    ax.text(best_idx, means[best_idx] + stds[best_idx] + max(means) * 0.1, "★", ha="center", color="#C0392B", fontsize=14)
    ax.set_title("Monotonicity Violation Rate Across Ablation Models", fontweight="bold")
    ax.set_ylabel("MVR")
    ax.set_xticks(x)
    ax.set_xticklabels(out_df["model"], rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_MVR_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_c_hill_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame):
    focus_order = ["Model C", "C+Cycle", "C+Hill", "Model D"]
    focus_df = build_model_summary(
        [row for row in summary_df.to_dict(orient="records") if row["model"] in focus_order],
        focus_order,
    )
    focus_df.to_csv(OUT_CH_SUMMARY, index=False, encoding="utf-8-sig")
    detail_df[detail_df["model"].isin(focus_order)].to_csv(OUT_CH_DETAIL, index=False, encoding="utf-8-sig")
    focus_df[focus_df["model"] == "C+Hill"].to_csv(OUT_CH_3SEED_SUMMARY, index=False, encoding="utf-8-sig")

    metrics = [
        ("mae_ng_ml_mean", "MAE (ng/mL)", True),
        ("rmse_ng_ml_mean", "RMSE (ng/mL)", True),
        ("mape_pct_mean", "MAPE (%)", True),
        ("r2_mean", "R²", False),
        ("mvr_mean", "MVR", True),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8))
    x = np.arange(len(focus_df))
    labels = focus_df["model"].tolist()

    for ax, (col, title, low_better) in zip(axes, metrics):
        std_col = col.replace("_mean", "_std")
        means = focus_df[col].to_numpy(dtype=float)
        stds = np.nan_to_num(focus_df[std_col].to_numpy(dtype=float), nan=0.0)
        colors = [MODEL_COLORS[name] for name in labels]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
        for idx, mean in enumerate(means):
            ax.text(idx, mean + max(means) * 0.03 if max(means) > 0 else mean + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5)
        best_idx = int(np.argmin(means) if low_better else np.argmax(means))
        ax.text(best_idx, means[best_idx] + stds[best_idx] + max(means) * 0.1 if max(means) > 0 else means[best_idx] + 0.03, "★", ha="center", color="#C0392B", fontsize=13)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Extended Ablation: Effect of Cycle and Hill Constraints", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_CH_FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    detail_rows = _collect_seed_rows()
    detail_df = pd.DataFrame(detail_rows)
    summary_rows = [
        _summarise_model(detail_rows, "Model A", "20260325"),
        _summarise_model(detail_rows, "Model B", "20260325"),
        _summarise_model(detail_rows, "Model C", ",".join(str(seed) for seed in SEEDS)),
        _summarise_model(detail_rows, "C+Cycle", ",".join(str(seed) for seed in SEEDS)),
        _summarise_model(detail_rows, "C+Hill", ",".join(str(seed) for seed in SEEDS)),
        _summarise_model(detail_rows, "Model D", ",".join(str(seed) for seed in SEEDS)),
    ]
    summary_df = build_model_summary(
        summary_rows,
        ["Model A", "Model B", "Model C", "C+Cycle", "C+Hill", "Model D"],
    )
    _write_mvr_outputs(summary_df)
    _write_c_hill_outputs(summary_df, detail_df)
    print(f"Saved {OUT_MVR_CSV}")
    print(f"Saved {OUT_MVR_FIG}")
    print(f"Saved {OUT_CH_SUMMARY}")
    print(f"Saved {OUT_CH_DETAIL}")
    print(f"Saved {OUT_CH_3SEED_SUMMARY}")
    print(f"Saved {OUT_CH_FIG}")


if __name__ == "__main__":
    main()
