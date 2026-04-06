"""
Supplementary Figures: Bland-Altman + Segment Statistics
=========================================================
Outputs:
  outputs/bland_altman_3c_figure.png   -- Figure S1: B-A plot (3C, per-seed + combined)
  outputs/segment_stats_table.csv      -- Table S1: per-model per-segment MAE
  outputs/segment_stats_figure.png     -- Figure S2: grouped bar per segment × model

Usage:
  python scripts/plot_supplementary.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Prediction files ─────────────────────────────────────────────────────────────
A_PREDS  = os.path.join(OUTPUTS_DIR, "stage0", "split_test_predictions_predictor_v1.csv")
B_PREDS  = os.path.join(OUTPUTS_DIR, "stage0", "split_test_predictions_predictor_v2.csv")
C_PREDS  = os.path.join(OUTPUTS_DIR, "stage1", "split_test_predictions_predictor_v2_fusion.csv")
D_SEED_DIRS = [
    os.path.join(OUTPUTS_DIR, "stage3_3c_learnable_regressor_seed20260325",
                 "split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv"),
    os.path.join(OUTPUTS_DIR, "stage3_3c_learnable_regressor_seed20260331",
                 "split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv"),
    os.path.join(OUTPUTS_DIR, "stage3_3c_learnable_regressor_seed20260407",
                 "split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv"),
]

OUT_BA_FIG   = os.path.join(OUTPUTS_DIR, "bland_altman_3c_figure.png")
OUT_SEG_CSV  = os.path.join(OUTPUTS_DIR, "segment_stats_table.csv")
OUT_SEG_FIG  = os.path.join(OUTPUTS_DIR, "segment_stats_figure.png")

# Style ────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
MODEL_COLORS = {
    "A": "#7B7B7B",
    "B": "#4C72B0",
    "C": "#DD8452",
    "D": "#55A868",
}
MODEL_LABELS = {
    "A": "Model A (V1)",
    "B": "Model B (V2)",
    "C": "Model C (Fusion)",
    "D": "Model D (3C+Hill)",
}
SEED_COLORS = ["#1565C0", "#B71C1C", "#1B5E20"]
SEED_MARKS  = ["o", "s", "^"]
SEEDS       = ["s0325", "s0331", "s0407"]

# Concentration segments (ng/mL)
SEG_LOW  = (0,    5)    # low
SEG_MID  = (5,   30)    # mid
SEG_HIGH = (30,  200)   # high


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_preds(path):
    df = pd.read_csv(path)
    return df["true_ng_ml"].values, df["pred_ng_ml"].values


def load_D_all():
    frames = [pd.read_csv(p) for p in D_SEED_DIRS if os.path.exists(p)]
    if not frames:
        return np.array([]), np.array([])
    df = pd.concat(frames, ignore_index=True)
    return df["true_ng_ml"].values, df["pred_ng_ml"].values


def load_D_by_seed():
    result = []
    for p in D_SEED_DIRS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            result.append((df["true_ng_ml"].values, df["pred_ng_ml"].values))
    return result   # list of (true, pred) per seed


def mae_in_segment(true, pred, lo, hi):
    mask = (true >= lo) & (true < hi)
    if mask.sum() == 0:
        return float("nan"), int(mask.sum())
    return float(np.mean(np.abs(true[mask] - pred[mask]))), int(mask.sum())


# ── Figure S1: Bland-Altman (3C per seed + combined) ─────────────────────────
def plot_bland_altman(save_path):
    seed_data = load_D_by_seed()
    all_true, all_pred = load_D_all()

    n_cols = len(seed_data) + 1   # 3 seeds + combined
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5), sharey=False)
    fig.suptitle(
        "Bland–Altman Analysis — Model D (3C, learnable-regressor)\n"
        "Difference (Pred − True) vs. Mean (Pred + True) / 2",
        fontweight="bold", fontsize=13, y=1.02)

    def _ba_ax(ax, true_v, pred_v, color, title, extra_label=""):
        mean_ = (true_v + pred_v) / 2.0
        diff_ = pred_v - true_v
        md    = np.mean(diff_)
        sd    = np.std(diff_, ddof=1)
        loa_u = md + 1.96 * sd
        loa_l = md - 1.96 * sd

        ax.scatter(mean_, diff_, c=color, alpha=0.55, s=40,
                   edgecolors="none", zorder=3)
        ax.axhline(md,    color="#222222", lw=1.8, linestyle="-", zorder=4)
        ax.axhline(loa_u, color="#C0392B", lw=1.4, linestyle="--", zorder=4)
        ax.axhline(loa_l, color="#C0392B", lw=1.4, linestyle="--", zorder=4)
        ax.axhline(0,     color="#999999", lw=0.9, linestyle=":", zorder=2)

        # Labels on lines
        xmax = np.nanmax(mean_) if len(mean_) else 1
        ax.text(xmax * 1.01, md,    f"Mean\n{md:+.2f}", va="center",
                fontsize=8, color="#222222", clip_on=False)
        ax.text(xmax * 1.01, loa_u, f"+1.96SD\n{loa_u:+.2f}", va="center",
                fontsize=8, color="#C0392B", clip_on=False)
        ax.text(xmax * 1.01, loa_l, f"−1.96SD\n{loa_l:+.2f}", va="center",
                fontsize=8, color="#C0392B", clip_on=False)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean of Pred & True (ng/mL)")
        if ax == axes[0]:
            ax.set_ylabel("Difference  Pred − True (ng/mL)")
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        n = len(true_v)
        ax.text(0.03, 0.97, f"n={n}", transform=ax.transAxes,
                va="top", fontsize=9, color="#555555")

    for i, ((trv, prv), sc, mk, sl) in enumerate(
            zip(seed_data, SEED_COLORS, SEED_MARKS, SEEDS)):
        _ba_ax(axes[i], trv, prv, sc, f"Seed {sl}")

    _ba_ax(axes[-1], all_true, all_pred, "#8E44AD",
           "Combined (3 seeds)", "combined")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure S1] Saved → {save_path}")


# ── Figure S2 + Table S1: Segment statistics (A/B/C/D) ───────────────────────
SEGMENTS = [
    ("Low\n(<5 ng/mL)",   SEG_LOW),
    ("Mid\n(5–30 ng/mL)", SEG_MID),
    ("High\n(>30 ng/mL)", SEG_HIGH),
]


def compute_segment_stats():
    preds_map = {
        "A": load_preds(A_PREDS),
        "B": load_preds(B_PREDS),
        "C": load_preds(C_PREDS),
    }
    dt, dp = load_D_all()
    preds_map["D"] = (dt, dp)

    rows = []
    for m, (true, pred) in preds_map.items():
        for seg_label, (lo, hi) in SEGMENTS:
            mae_v, n = mae_in_segment(true, pred, lo, hi)
            rows.append({
                "Model": m,
                "Segment": seg_label.replace("\n", " "),
                "Lo_ng_ml": lo,
                "Hi_ng_ml": hi,
                "N": n,
                "MAE_ng_ml": round(mae_v, 4) if not np.isnan(mae_v) else float("nan"),
            })
    return pd.DataFrame(rows)


def save_segment_table(df, save_path):
    pivot = df.pivot_table(index="Model", columns="Segment", values="MAE_ng_ml")
    # reorder columns
    ordered = [s.replace("\n", " ") for _, (lo, hi) in SEGMENTS
               for s in [f"Low (<5 ng/mL)", f"Mid (5-30 ng/mL)", f"High (>30 ng/mL)"]]
    pivot = pivot.reindex(columns=[c for c in pivot.columns])
    pivot.to_csv(save_path, encoding="utf-8-sig")
    print(f"[Table  S1] Saved → {save_path}")
    print(pivot.to_string())


def plot_segment_figure(df, save_path):
    models = ["A", "B", "C", "D"]
    seg_labels = [seg for seg, _ in SEGMENTS]
    n_segs = len(SEGMENTS)
    x = np.arange(len(models))
    bar_w = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Segmented MAE by Concentration Range — Ablation Models",
                 fontweight="bold", fontsize=13)

    seg_colors = ["#5DADE2", "#F39C12", "#E74C3C"]   # low=blue, mid=amber, high=red

    for si, ((seg_label, _), sc) in enumerate(zip(SEGMENTS, seg_colors)):
        offset = (si - 1) * bar_w
        heights = []
        for mi, m in enumerate(models):
            sub = df[(df["Model"] == m) &
                     (df["Segment"] == seg_label.replace("\n", " "))]
            mae_v = float(sub["MAE_ng_ml"].values[0]) if len(sub) else float("nan")
            heights.append(mae_v)
            bar = ax.bar(mi + offset, mae_v, bar_w, color=sc,
                         edgecolor="white", linewidth=0.6, zorder=3,
                         label=seg_label.replace("\n", " ") if mi == 0 else "")
            if not np.isnan(mae_v):
                ax.text(mi + offset, mae_v + 0.15, f"{mae_v:.2f}",
                        ha="center", fontsize=7.5, fontweight="semibold")

    # Model x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{MODEL_LABELS[m].split('(')[1].rstrip(')')}"
                        for m in models], fontsize=10)
    ax.set_ylabel("MAE (ng/mL)")
    ax.set_ylim(0, df["MAE_ng_ml"].max() * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend for segments
    seg_patches = [mpatches.Patch(color=sc, label=sl.replace("\n", " "))
                   for sc, (sl, _) in zip(seg_colors, SEGMENTS)]
    ax.legend(handles=seg_patches, title="Concentration Range",
              loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure S2] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=== Bland-Altman (3C) ===")
    plot_bland_altman(OUT_BA_FIG)

    print("\n=== Segment Statistics (A/B/C/D) ===")
    seg_df = compute_segment_stats()
    save_segment_table(seg_df, OUT_SEG_CSV)
    plot_segment_figure(seg_df, OUT_SEG_FIG)

    print("\nDone. Outputs in:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
