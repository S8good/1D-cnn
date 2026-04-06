"""
Full Ablation Comparison + Auxiliary Figures
=============================================
Outputs:
  outputs/ablation_summary.csv              -- Table: A/B/C/D mean metrics
  outputs/ablation_comparison_figure.png    -- Figure 3: 4-model ablation bar chart
  outputs/true_vs_pred_3c_figure.png        -- Figure 4: True-vs-Pred scatter (3C, seed-mean)
  outputs/hill_consistency_figure.png       -- Figure 5: Hill curve + 3A/3B/3C Δλ scatter

Usage:
  python scripts/plot_ablation_full.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Stage 3 summary (for D = 3C mean)
S3_SUMMARY = os.path.join(OUTPUTS_DIR, "stage3_3seed_summary_20260331.csv")

# Single-seed metrics for A / B / C
A_METRICS = os.path.join(OUTPUTS_DIR, "stage0", "split_test_metrics_predictor_v1.csv")
B_METRICS = os.path.join(OUTPUTS_DIR, "stage0", "split_test_metrics_predictor_v2.csv")
C_METRICS = os.path.join(OUTPUTS_DIR, "stage1", "split_test_metrics_predictor_v2_fusion.csv")

# Single-seed predictions for True-vs-Pred (3C seed20260325 = best representative)
C_PREDS_PATH = os.path.join(OUTPUTS_DIR, "stage1", "split_test_predictions_predictor_v2_fusion.csv")
D_PREDS_DIR  = os.path.join(OUTPUTS_DIR)  # stage3_3c_learnable_regressor_seedXXXXXX/

# Stage3 Hill params pth
HILL_PARAMS_PTH = os.path.join(ROOT_DIR, "models", "pretrained", "stage3_hill_params.pth")

# Stage3 detail for Hill consistency scatter
S3_DETAIL = os.path.join(OUTPUTS_DIR, "stage3_3seed_detail_20260331.csv")

# Output files
OUT_ABLATION_CSV = os.path.join(OUTPUTS_DIR, "ablation_summary.csv")
OUT_ABLATION_FIG = os.path.join(OUTPUTS_DIR, "ablation_comparison_figure.png")
OUT_TVP_FIG      = os.path.join(OUTPUTS_DIR, "true_vs_pred_3c_figure.png")
OUT_HILL_FIG     = os.path.join(OUTPUTS_DIR, "hill_consistency_figure.png")

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})

MODEL_COLORS = {
    "A": "#7B7B7B",  # grey - baseline CNN
    "B": "#4C72B0",  # blue - V2
    "C": "#DD8452",  # orange - V2_Fusion
    "D": "#55A868",  # green - 3C (Model D)
}
MODEL_DISPLAY = {
    "A": "Model A\n(V1, 1D-CNN)",
    "B": "Model B\n(V2, dual-ch.)",
    "C": "Model C\n(V2-Fusion)",
    "D": "Model D\n(3C, +Hill)",
}
S3_COLORS = {"3A": "#4C72B0", "3B": "#DD8452", "3C": "#55A868"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_single_metrics(path):
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        "mae": float(row["mae_ng_ml"]),
        "rmse": float(row["rmse_ng_ml"]),
        "mape": float(row["mape_pct"]),
        "r2": float(row["r2"]),
    }


def load_stage3_mean(path, model_key="3C"):
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["model"] = df["model"].str.strip().str.strip('"')
    row = df[df["model"] == model_key].iloc[0]
    return {
        "mae": float(row["mae_mean"]),
        "rmse": float(row["rmse_mean"]),
        "mape": float(row["mape_mean"]),
        "r2": float(row["r2_mean"]),
    }


def hill_curve_numpy(conc, dl_max, k_half, hill_n):
    conc = np.clip(conc, 0.0, None)
    return dl_max * (conc ** hill_n) / (k_half ** hill_n + conc ** hill_n + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Ablation comparison bar chart (A / B / C / D)
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_METRICS = [
    ("mae",  "MAE (ng/mL)",  True),
    ("rmse", "RMSE (ng/mL)", True),
    ("mape", "MAPE (%)",     True),
    ("r2",   "R²",           False),
]

def plot_ablation_figure(metric_dict, save_path):
    """metric_dict: {"A": {mae, rmse, mape, r2}, "B": ..., "C": ..., "D": ...}"""
    models = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Ablation Study — Model A / B / C / D (Test Set)", fontsize=13,
                 fontweight="bold", y=1.01)

    bar_w = 0.55
    x = np.arange(len(models))

    for ax_idx, (key, ylabel, low_better) in enumerate(ABLATION_METRICS):
        ax = axes[ax_idx]
        vals = [metric_dict[m][key] for m in models]
        best_idx = int(np.argmin(vals) if low_better else np.argmax(vals))

        for i, (m, v) in enumerate(zip(models, vals)):
            c = MODEL_COLORS[m]
            bar = ax.bar(i, v, bar_w, color=c, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(i, v + max(vals) * 0.01, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="semibold")

        # Best marker
        star_y = vals[best_idx] + max(vals) * 0.08
        ax.text(best_idx, star_y, "★", ha="center", fontsize=14, color="#C0392B")

        ymin = 0 if key != "r2" else min(vals) * 0.92
        ax.set_ylim(ymin, max(vals) * 1.22)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in models], fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Color legend
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_DISPLAY[m].replace("\n", " "))
               for m in models]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.0, 1.01),
               ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, -0.02, "★ = best value per metric", fontsize=9, color="#C0392B")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 3] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table — ablation_summary.csv
# ─────────────────────────────────────────────────────────────────────────────
def save_ablation_table(metric_dict, save_path):
    models = ["A", "B", "C", "D"]
    keys_info = [
        ("mae",  "MAE↓ (ng/mL)", True),
        ("rmse", "RMSE↓ (ng/mL)", True),
        ("mape", "MAPE↓ (%)", True),
        ("r2",   "R²↑", False),
    ]
    # find best per metric
    best = {}
    for k, _, low in keys_info:
        vals = [metric_dict[m][k] for m in models]
        best[k] = int(np.argmin(vals) if low else np.argmax(vals))

    rows = []
    for mi, m in enumerate(models):
        row = {"Model": m, "Name": MODEL_DISPLAY[m].replace("\n", " ")}
        for k, col, low in keys_info:
            v = metric_dict[m][k]
            cell = f"*{v:.4f}" if best[k] == mi else f"{v:.4f}"
            row[col] = cell
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[Table  3] Saved → {save_path}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — True vs Pred scatter (3C, all 3 seeds stacked)
# ─────────────────────────────────────────────────────────────────────────────
def load_3c_predictions():
    """Load and stack predictions from all 3 seeds for 3C."""
    seed_dirs = [
        "stage3_3c_learnable_regressor_seed20260325",
        "stage3_3c_learnable_regressor_seed20260331",
        "stage3_3c_learnable_regressor_seed20260407",
    ]
    frames = []
    for d in seed_dirs:
        p = os.path.join(OUTPUTS_DIR, d, f"split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        # fallback: single seed at root
        p = os.path.join(OUTPUTS_DIR, "split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_true_vs_pred(preds_df, save_path, model_label="Model D (3C, learnable-reg.)", color="#55A868"):
    if preds_df.empty:
        print(f"[Figure 4] Skipped — no prediction data found.")
        return

    true_v = preds_df["true_ng_ml"].values
    pred_v = preds_df["pred_ng_ml"].values
    mae  = np.mean(np.abs(true_v - pred_v))
    rmse = np.sqrt(np.mean((true_v - pred_v) ** 2))
    r, _ = pearsonr(true_v, pred_v)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(true_v, pred_v, c=color, alpha=0.45, s=30, edgecolors="none", zorder=3,
               label=f"n={len(true_v)} (3 seeds × test set)")

    # Identity line
    vmax = max(true_v.max(), pred_v.max()) * 1.05
    vmin = min(0, true_v.min(), pred_v.min())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2, alpha=0.6, label="Ideal (y=x)")

    # Annotation box
    txt = f"MAE={mae:.3f} ng/mL\nRMSE={rmse:.3f} ng/mL\nPearson r={r:.4f}"
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xlabel("True Concentration (ng/mL)")
    ax.set_ylabel("Predicted Concentration (ng/mL)")
    ax.set_title(f"True vs. Predicted — {model_label}\n(3-Seed, concatenated test set)",
                 fontweight="bold")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 4] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Hill consistency curve + 3A/3B/3C scatter
# ─────────────────────────────────────────────────────────────────────────────
def load_hill_params():
    """Try to load Hill params from .pth; fallback to reasonable defaults."""
    try:
        import torch
        params = torch.load(HILL_PARAMS_PTH, map_location="cpu", weights_only=True)
        dl_max  = float(params["delta_lambda_max"])
        k_half  = float(params["k_half"])
        hill_n  = float(params["hill_n"])
        print(f"[Hill params] Loaded from {HILL_PARAMS_PTH}: dl_max={dl_max:.4f}, k_half={k_half:.4f}, n={hill_n:.4f}")
        return dl_max, k_half, hill_n
    except Exception as e:
        print(f"[Hill params] Could not load .pth ({e}); using fallback defaults.")
        return 6.5, 12.0, 1.0  # typical LSPR Hill-fit defaults


def load_3model_pred_delta(outputs_dir):
    """
    Load per-model prediction data to get (predicted_conc, hill_mae_per_seed) for scatter.
    We materialise concentration predictions → compute expected Δλ from Hill → scatter.
    """
    data = {}
    for m_key, subdir_prefix in [("3A", "stage3_3a_fixed_frozen_seed"),
                                  ("3B", "stage3_3b_fixed_regressor_seed"),
                                  ("3C", "stage3_3c_learnable_regressor_seed")]:
        for sfx, fn_key in [("20260325", "3a_fixed_frozen"), ("20260331", "3a_fixed_frozen"),
                             ("20260407", "3a_fixed_frozen"),
                             ("20260325", "3b_fixed_regressor"), ("20260331", "3b_fixed_regressor"),
                             ("20260407", "3b_fixed_regressor"),
                             ("20260325", "3c_learnable_regressor"), ("20260331", "3c_learnable_regressor"),
                             ("20260407", "3c_learnable_regressor")]:
            pass  # just placeholder; load per-model below

    # Simpler: load each seed predictions file for each model
    model_concs = {m: [] for m in ["3A", "3B", "3C"]}
    model_preds  = {m: [] for m in ["3A", "3B", "3C"]}

    for m_key, folder_pfx, fn_part in [
        ("3A", "stage3_3a_fixed_frozen_seed",        "split_test_predictions_predictor_v2_stage3_3a_fixed_frozen.csv"),
        ("3B", "stage3_3b_fixed_regressor_seed",     "split_test_predictions_predictor_v2_stage3_3b_fixed_regressor.csv"),
        ("3C", "stage3_3c_learnable_regressor_seed", "split_test_predictions_predictor_v2_stage3_3c_learnable_regressor.csv"),
    ]:
        for seed in ["20260325", "20260331", "20260407"]:
            p = os.path.join(outputs_dir, f"{folder_pfx}{seed}", fn_part)
            if os.path.exists(p):
                df = pd.read_csv(p)
                model_concs[m_key].extend(df["true_ng_ml"].tolist())
                model_preds[m_key].extend(df["pred_ng_ml"].tolist())

    return model_concs, model_preds


def plot_hill_consistency(save_path):
    dl_max, k_half, hill_n = load_hill_params()

    # Hill curve (theory)
    conc_line = np.linspace(0.3, 115, 600)
    dl_theory = hill_curve_numpy(conc_line, dl_max, k_half, hill_n)

    model_concs_true, model_preds = load_3model_pred_delta(OUTPUTS_DIR)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # ── Reference: Hill(true_conc) = ideal target (light grey, plotted first)
    concs_all = np.array(model_concs_true["3C"])
    if len(concs_all):
        dl_ideal = hill_curve_numpy(concs_all, dl_max, k_half, hill_n)
        ax.scatter(concs_all, dl_ideal, c="#BBBBBB", alpha=0.30, s=28,
                   edgecolors="none", zorder=2, label="Hill(true_conc)  [ideal target]")

    # ── Model scatter — bold, high-contrast, distinct marker shapes
    scatter_styles = {
        "3A": dict(c="#1565C0", alpha=0.70, s=60, marker="^",
                   edgecolors="#0D47A1", linewidths=0.5),   # deep blue triangle
        "3B": dict(c="#E65100", alpha=0.70, s=60, marker="s",
                   edgecolors="#BF360C", linewidths=0.5),   # deep orange square
        "3C": dict(c="#1B5E20", alpha=0.75, s=60, marker="o",
                   edgecolors="#004D00", linewidths=0.5),   # deep green circle
    }
    for m_key in ["3A", "3B", "3C"]:
        concs_true = np.array(model_concs_true[m_key])
        preds_     = np.array(model_preds[m_key])
        if len(concs_true) == 0:
            continue
        dl_from_pred = hill_curve_numpy(preds_, dl_max, k_half, hill_n)
        ax.scatter(concs_true, dl_from_pred, zorder=4,
                   label=f"{m_key}:  Hill(pred_conc)",
                   **scatter_styles[m_key])

    # ── Theory curve on top
    ax.plot(conc_line, dl_theory, color="#111111", lw=2.8, zorder=6,
            label="Hill curve  (theory)")

    ax.set_xlabel("True Concentration (ng/mL)", fontsize=12)
    ax.set_ylabel("Expected Δλ from Hill model (nm)", fontsize=12)
    ax.set_title(
        "Hill Physical Consistency\n"
        r"$\Delta\lambda$ = Hill(pred_conc) vs. True Concentration (3-Seed)",
        fontweight="bold", fontsize=13)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 140)
    ax.set_ylim(-0.15, dl_max * 1.18)
    ax.grid(linestyle=":", linewidth=0.8, alpha=0.45)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.92,
              edgecolor="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation outside axes (below xlabel)
    fig.text(0.5, -0.03,
             "Note: scatter closer to the black curve → better physical consistency",
             ha="center", fontsize=9.5, color="#444444", style="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 5] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Load metrics ──────────────────────────────────────────────────────────
    print("Loading metrics …")
    metrics = {
        "A": load_single_metrics(A_METRICS),
        "B": load_single_metrics(B_METRICS),
        "C": load_single_metrics(C_METRICS),
        "D": load_stage3_mean(S3_SUMMARY, "3C"),
    }
    for m, v in metrics.items():
        print(f"  Model {m}: MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}  "
              f"MAPE={v['mape']:.2f}%  R²={v['r2']:.4f}")

    # ── Figure 3 & Table ──────────────────────────────────────────────────────
    plot_ablation_figure(metrics, OUT_ABLATION_FIG)
    save_ablation_table(metrics, OUT_ABLATION_CSV)

    # ── Figure 4 — True vs Pred ───────────────────────────────────────────────
    print("Loading 3C predictions for True-vs-Pred …")
    preds_3c = load_3c_predictions()
    plot_true_vs_pred(preds_3c, OUT_TVP_FIG)

    # ── Figure 5 — Hill consistency ───────────────────────────────────────────
    print("Loading stage3 predictions for Hill consistency …")
    plot_hill_consistency(OUT_HILL_FIG)

    print("\nDone. All outputs in:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
