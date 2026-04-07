import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SOURCE_OUTPUTS = Path("C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master/outputs")
DEFAULT_WORKTREE_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def _profile_tag(profile: str) -> str:
    return profile.lower().replace(".", "p")


def _std(values):
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _seed_to_str(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def collect_model_c_summary(source_outputs: Path):
    stage2_dir = source_outputs / "stage2_regressor_3seed_20260326_143033"
    summary_df = pd.read_csv(stage2_dir / "summary_mean_std_with_mape.csv")
    paired_df = pd.read_csv(stage2_dir / "paired_seed_metrics.csv")

    row = summary_df.loc[summary_df["model"] == "Model C"].iloc[0]
    best_row = paired_df.sort_values(["model_c_rmse", "model_c_mae", "model_c_r2"], ascending=[True, True, False]).iloc[0]
    return {
        "label": "Model C",
        "mean": {"mae": float(row["mae_mean"]), "rmse": float(row["rmse_mean"]), "r2": float(row["r2_mean"])},
        "std": {"mae": float(row["mae_std"]), "rmse": float(row["rmse_std"]), "r2": float(row["r2_std"])},
        "best": {
            "seed": _seed_to_str(best_row["seed"]),
            "mae": float(best_row["model_c_mae"]),
            "rmse": float(best_row["model_c_rmse"]),
            "r2": float(best_row["model_c_r2"]),
        },
    }


def collect_stage25_profile_summary(worktree_outputs: Path, profile: str):
    tag = _profile_tag(profile)
    pattern = f"stage25_{tag}_seed*"
    rows = []
    for run_dir in sorted(worktree_outputs.glob(pattern)):
        if not re.fullmatch(rf"stage25_{tag}_seed\d{{8}}", run_dir.name):
            continue
        csv_path = run_dir / f"split_test_metrics_predictor_v2_stage25_{tag}.csv"
        if csv_path.exists():
            row = pd.read_csv(csv_path).iloc[0].to_dict()
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No Stage 2.5 outputs found for {profile} under {worktree_outputs}")

    mae = [float(r["mae_ng_ml"]) for r in rows]
    rmse = [float(r["rmse_ng_ml"]) for r in rows]
    r2 = [float(r["r2"]) for r in rows]
    best_row = sorted(rows, key=lambda r: (float(r["rmse_ng_ml"]), float(r["mae_ng_ml"]), -float(r["r2"])))[0]
    return {
        "label": f"Stage {profile}",
        "mean": {"mae": sum(mae) / len(mae), "rmse": sum(rmse) / len(rmse), "r2": sum(r2) / len(r2)},
        "std": {"mae": _std(mae), "rmse": _std(rmse), "r2": _std(r2)},
        "best": {
            "seed": _seed_to_str(best_row["seed"]),
            "mae": float(best_row["mae_ng_ml"]),
            "rmse": float(best_row["rmse_ng_ml"]),
            "r2": float(best_row["r2"]),
        },
    }


def plot_stage25_comparison(summary_rows, out_path: Path):
    labels = [row["label"] for row in summary_rows]
    metrics = [("mae", "MAE (ng/ml)", False), ("rmse", "RMSE (ng/ml)", False), ("r2", "R2", True)]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 11))
    fig.suptitle("Model C vs Stage 2.5 Comparison", fontsize=15, fontweight="bold")

    x = list(range(len(labels)))
    for row_index, (metric_key, ylabel, higher_is_better) in enumerate(metrics):
        best_ax = axes[row_index, 0]
        mean_ax = axes[row_index, 1]

        best_values = [row["best"][metric_key] for row in summary_rows]
        mean_values = [row["mean"][metric_key] for row in summary_rows]
        std_values = [row["std"][metric_key] for row in summary_rows]

        best_ax.bar(x, best_values, color=["#2f5d8a", "#d17c2f", "#2f8a5d", "#8a2f5d"])
        best_ax.set_title(f"Best Single Run by {ylabel}")
        best_ax.set_ylabel(ylabel)
        best_ax.set_xticks(x, labels, rotation=15)
        best_ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        mean_ax.errorbar(x, mean_values, yerr=std_values, fmt="o", capsize=5, color="#1f1f1f", ecolor="#4d4d4d")
        mean_ax.set_title(f"3-seed Mean ± Std by {ylabel}")
        mean_ax.set_ylabel(ylabel)
        mean_ax.set_xticks(x, labels, rotation=15)
        mean_ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        if higher_is_better:
            best_ax.set_ylim(bottom=min(best_values + mean_values) - 0.02, top=max(best_values + mean_values) + 0.02)
            mean_ax.set_ylim(bottom=min(best_values + mean_values) - 0.02, top=max(best_values + mean_values) + 0.02)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Model C vs Stage 2.5 comparison.")
    parser.add_argument("--source-outputs", type=Path, default=DEFAULT_SOURCE_OUTPUTS)
    parser.add_argument("--worktree-outputs", type=Path, default=DEFAULT_WORKTREE_OUTPUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_SOURCE_OUTPUTS / "compare_modelc_stage25_abc.png")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_rows = [
        collect_model_c_summary(args.source_outputs),
        collect_stage25_profile_summary(args.worktree_outputs, "2.5A"),
        collect_stage25_profile_summary(args.worktree_outputs, "2.5B"),
        collect_stage25_profile_summary(args.worktree_outputs, "2.5C"),
    ]
    plot_stage25_comparison(summary_rows, args.out)
    print(f"Saved comparison plot: {args.out}")


if __name__ == "__main__":
    main()
