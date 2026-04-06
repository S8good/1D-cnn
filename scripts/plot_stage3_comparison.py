"""
Stage 3 Formal Comparison Figures & Paper Table
================================================
Outputs:
  outputs/stage3_comparison_figure.png   -- Figure 1: 1×5 grouped bar (3A/3B/3C)
  outputs/stage3_hilmae_figure.png       -- Figure 1b: Hill-MAE standalone
  outputs/stage3_paper_table.csv         -- Table 1: mean±std, best marked *
  outputs/stage3_seed_detail_figure.png  -- Figure 2: seed detail scatter (if --seed-detail)

Usage:
  python scripts/plot_stage3_comparison.py
  python scripts/plot_stage3_comparison.py --seed-detail
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR  = os.path.join(ROOT_DIR, "outputs")
SUMMARY_CSV  = os.path.join(OUTPUTS_DIR, "stage3_3seed_summary_20260331.csv")
DETAIL_CSV   = os.path.join(OUTPUTS_DIR, "stage3_3seed_detail_20260331.csv")
FIG1_PATH    = os.path.join(OUTPUTS_DIR, "stage3_comparison_figure.png")
FIG1B_PATH   = os.path.join(OUTPUTS_DIR, "stage3_hilmae_figure.png")
TABLE_PATH   = os.path.join(OUTPUTS_DIR, "stage3_paper_table.csv")
FIG2_PATH    = os.path.join(OUTPUTS_DIR, "stage3_seed_detail_figure.png")

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "3A": "#4C72B0",  # steel blue
    "3B": "#DD8452",  # warm orange
    "3C": "#55A868",  # forest green
}
MODEL_LABELS = {
    "3A": "3A (fixed-frozen)",
    "3B": "3B (fixed-regressor)",
    "3C": "3C (learnable-regressor)",
}
MODELS = ["3A", "3B", "3C"]

# Metrics to plot: (col_prefix, display_label, units, lower_is_better)
METRICS = [
    ("mae",      "MAE",      "ng/mL", True),
    ("rmse",     "RMSE",     "ng/mL", True),
    ("mape",     "MAPE",     "%",     True),
    ("r2",       "R²",       "",      False),
    ("hill_mae", "Hill-MAE", "nm",    True),
]

matplotlib.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      100,
})


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names: strip quotes / whitespace
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["model"] = df["model"].str.strip().str.strip('"')
    # rename hill_mae columns if needed
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if "hill" in lc and "mean" in lc:
            rename_map[c] = "hill_mae_mean"
        elif "hill" in lc and "std" in lc:
            rename_map[c] = "hill_mae_std"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_detail(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["model"] = df["model"].str.strip().str.strip('"')
    # hill column
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if "hill" in lc and "consist" in lc:
            rename_map[c] = "hill_mae_nm"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — grouped bar chart, 1×5
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure1(summary: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    fig.suptitle("Stage 3 Model Comparison (3-Seed, mean ± std)", fontsize=13, fontweight="bold", y=1.01)

    bar_width = 0.22
    x_positions = np.arange(len(MODELS))

    for ax_idx, (prefix, label, unit, low_better) in enumerate(METRICS):
        ax = axes[ax_idx]

        # Hill-MAE sub-plot gets grey background
        if prefix == "hill_mae":
            ax.set_facecolor("#F5F5F5")

        means, stds = [], []
        for m in MODELS:
            row = summary[summary["model"] == m].iloc[0]
            means.append(float(row[f"{prefix}_mean"]))
            stds.append(float(row[f"{prefix}_std"]))

        for i, (m, mean, std) in enumerate(zip(MODELS, means, stds)):
            bar = ax.bar(
                i, mean, bar_width * 2.4,
                color=MODEL_COLORS[m],
                edgecolor="white", linewidth=0.8,
                yerr=std, capsize=5,
                error_kw=dict(elinewidth=1.2, ecolor="#333333", capthick=1.2),
                zorder=3,
            )
            # Value label on top of bar
            label_y = mean + std + (max(means) * 0.02)
            ax.text(i, label_y, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="semibold", color="#222222")

        # Best-value marker (star on best bar)
        best_idx = int(np.argmin(means) if low_better else np.argmax(means))
        star_y = means[best_idx] + stds[best_idx] + (max(means) * 0.09)
        ax.text(best_idx, star_y, "★", ha="center", va="bottom",
                fontsize=13, color="#C0392B")

        ymin = 0 if prefix != "r2" else min(means) * 0.92
        ymax_raw = max(means) + max(stds)
        ypad = ymax_raw * 0.22
        ax.set_ylim(ymin, ymax_raw + ypad)

        ylabel = f"{label} ({unit})" if unit else label
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["3A", "3B", "3C"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if prefix == "hill_mae":
            ax.set_xlabel("*Physical consistency (Δλ vs. Hill curve)", fontsize=8,
                          labelpad=4, color="#555555")

    # Shared legend
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.0, 1.01),
               frameon=False)
    # Star annotation
    fig.text(0.01, -0.02, "★ = best value per metric", fontsize=9, color="#C0392B")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 1 ] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1b — Hill-MAE standalone
# ──────────────────────────────────────────────────────────────────────────────
def plot_figure1b(summary: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_facecolor("#F5F5F5")

    means, stds = [], []
    for m in MODELS:
        row = summary[summary["model"] == m].iloc[0]
        means.append(float(row["hill_mae_mean"]))
        stds.append(float(row["hill_mae_std"]))

    bar_width = 0.45
    for i, (m, mean, std) in enumerate(zip(MODELS, means, stds)):
        ax.bar(
            i, mean, bar_width,
            color=MODEL_COLORS[m], edgecolor="white", linewidth=0.8,
            label=MODEL_LABELS[m],
            yerr=std, capsize=6,
            error_kw=dict(elinewidth=1.5, ecolor="#333333", capthick=1.5),
            zorder=3,
        )
        label_y = mean + std + max(means) * 0.025
        ax.text(i, label_y, f"{mean:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="semibold")

    # Best marker
    best_idx = int(np.argmin(means))
    star_y = means[best_idx] + stds[best_idx] + max(means) * 0.10
    ax.text(best_idx, star_y, "★", ha="center", va="bottom",
            fontsize=16, color="#C0392B")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(["3A\n(fixed-frozen)", "3B\n(fixed-regressor)", "3C\n(learnable-reg.)"])
    ax.set_ylabel("Hill-MAE (nm)")
    ax.set_title("Hill Consistency Error — Δλ vs. Hill Curve\n(3-Seed, mean ± std)",
                 fontweight="bold")
    ymax_raw = max(means) + max(stds)
    ax.set_ylim(0, ymax_raw * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.12, "★ = best (lowest Hill-MAE)", transform=ax.transAxes,
            ha="center", fontsize=9, color="#C0392B")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 1b] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Seed detail scatter (3 rows × 5 cols)
# ──────────────────────────────────────────────────────────────────────────────
DETAIL_METRIC_MAP = [
    ("mae_ng_ml",   "MAE",      "ng/mL", True),
    ("rmse_ng_ml",  "RMSE",     "ng/mL", True),
    ("mape_pct",    "MAPE",     "%",     True),
    ("r2",          "R²",       "",      False),
    ("hill_mae_nm", "Hill-MAE", "nm",    True),
]

SEED_MARKERS = ["o", "s", "^"]
SEED_COLORS  = ["#2E86AB", "#E84855", "#3BB273"]


def plot_figure2(detail: pd.DataFrame, save_path: str):
    seeds = sorted(detail["seed"].astype(str).unique())
    nrows, ncols = len(MODELS), len(DETAIL_METRIC_MAP)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8),
                             sharey=False, sharex=False)
    fig.suptitle("Stage 3 Seed Detail — Individual Runs (n=3)", fontsize=13,
                 fontweight="bold", y=1.01)

    for row_idx, model in enumerate(MODELS):
        mdf = detail[detail["model"] == model]
        for col_idx, (col, label, unit, low_better) in enumerate(DETAIL_METRIC_MAP):
            ax = axes[row_idx][col_idx]
            values = []
            for si, (seed, mk, sc) in enumerate(zip(seeds, SEED_MARKERS, SEED_COLORS)):
                srow = mdf[mdf["seed"].astype(str) == str(seed)]
                if srow.empty:
                    continue
                val = float(srow.iloc[0][col])
                values.append(val)
                ax.scatter(si, val, marker=mk, color=sc, s=80, zorder=4,
                           label=f"seed {seed}" if row_idx == 0 and col_idx == 0 else None)

            if values:
                mean_v = np.mean(values)
                ax.axhline(mean_v, color=MODEL_COLORS[model], linewidth=1.8,
                           linestyle="--", alpha=0.85, zorder=3)
                # Shade between min/max
                ax.fill_between([-0.5, len(seeds) - 0.5],
                                min(values), max(values),
                                color=MODEL_COLORS[model], alpha=0.08)

            # Col header (top row only)
            if row_idx == 0:
                title = f"{label} ({unit})" if unit else label
                if col == "hill_mae_nm":
                    ax.set_facecolor("#F5F5F5")
                    title += "\n[Hill-MAE]"
                ax.set_title(title, fontsize=10)

            if col == "hill_mae_nm":
                ax.set_facecolor("#F5F5F5")

            # Row label (left col only)
            if col_idx == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9, rotation=90,
                              labelpad=4)

            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([f"s{s[-4:]}" for s in seeds], fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Seed legend
    seed_patches = [
        mpatches.Patch(color=sc, label=f"seed {s}")
        for s, sc in zip(seeds, SEED_COLORS)
    ]
    fig.legend(handles=seed_patches, loc="lower center",
               ncol=len(seeds), bbox_to_anchor=(0.5, -0.04), frameon=False,
               fontsize=10)
    fig.text(0.01, -0.06, "-- = mean across seeds", fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 2 ] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Table 1 — paper summary CSV
# ──────────────────────────────────────────────────────────────────────────────
TABLE_METRICS = [
    ("mae",      "MAE↓ (ng/mL)", True),
    ("rmse",     "RMSE↓ (ng/mL)", True),
    ("mape",     "MAPE↓ (%)", True),
    ("r2",       "R²↑", False),
    ("mvr",      "MVR", True),
    ("hill_mae", "Hill-MAE↓ (nm)", True),
]

PROFILE_NAMES = {
    "3A": "fixed-frozen",
    "3B": "fixed-regressor",
    "3C": "learnable-regressor",
}


def make_paper_table(summary: pd.DataFrame, save_path: str):
    rows = []
    # Collect all metric means to determine best
    metric_means = {prefix: [] for prefix, _, _ in TABLE_METRICS}
    for m in MODELS:
        row = summary[summary["model"] == m].iloc[0]
        for prefix, _, _ in TABLE_METRICS:
            mean_col = f"{prefix}_mean"
            metric_means[prefix].append(float(row[mean_col]) if mean_col in row else float("nan"))

    best_idxs = {}
    for prefix, _, low_better in TABLE_METRICS:
        vals = metric_means[prefix]
        if low_better:
            best_idxs[prefix] = int(np.nanargmin(vals))
        else:
            best_idxs[prefix] = int(np.nanargmax(vals))

    for mi, m in enumerate(MODELS):
        srow = summary[summary["model"] == m].iloc[0]
        record = {
            "Model": m,
            "Profile": PROFILE_NAMES[m],
            "Seeds": "20260325, 20260331, 20260407",
        }
        for prefix, col_label, _ in TABLE_METRICS:
            mean_col = f"{prefix}_mean"
            std_col  = f"{prefix}_std"
            mean_v = float(srow[mean_col]) if mean_col in srow else float("nan")
            std_v  = float(srow[std_col])  if std_col  in srow else float("nan")

            # Format
            if prefix == "r2":
                cell = f"{mean_v:.4f}±{std_v:.4f}"
            elif prefix in ("mvr",):
                cell = f"{mean_v:.3f}±{std_v:.3f}"
            elif prefix == "mape":
                cell = f"{mean_v:.2f}±{std_v:.2f}"
            elif prefix == "hill_mae":
                cell = f"{mean_v:.4f}±{std_v:.4f}"
            else:
                cell = f"{mean_v:.4f}±{std_v:.4f}"

            # Mark best with *
            if best_idxs[prefix] == mi:
                cell = f"*{cell}"

            record[col_label] = cell

        rows.append(record)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[Table 1  ] Saved → {save_path}")
    print(df_out.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3 comparison figures & paper table")
    parser.add_argument("--seed-detail", action="store_true",
                        help="Also generate Figure 2: per-seed scatter plot")
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading data …")
    summary = load_summary(SUMMARY_CSV)
    detail  = load_detail(DETAIL_CSV)

    print(f"Summary shape: {summary.shape}, models: {summary['model'].tolist()}")
    print(f"Detail  shape: {detail.shape}, models: {detail['model'].unique().tolist()}")

    plot_figure1(summary, FIG1_PATH)
    plot_figure1b(summary, FIG1B_PATH)
    make_paper_table(summary, TABLE_PATH)

    if args.seed_detail:
        plot_figure2(detail, FIG2_PATH)
    else:
        print("[Figure 2 ] Skipped (pass --seed-detail to generate)")

    print("\nDone. All outputs in:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
