import argparse
import os
import re
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.full_spectrum_models import SpectralPredictorV2_Fusion


def _prepare_model_dirs(project_root: str):
    models_root = os.path.join(project_root, "models")
    checkpoints_dir = os.path.join(models_root, "checkpoints")
    pretrained_dir = os.path.join(models_root, "pretrained")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)
    return models_root, checkpoints_dir, pretrained_dir


def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_bsa_feature_lookup(features_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(features_path, sheet_name="sheet1")
    required = ["spectrum_col", "stage", "peak_wavelength_nm", "peak_intensity_au", "fwhm_nm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in sheet1: {missing}")

    df = df.copy()
    df["stage"] = df["stage"].astype(str).str.upper()
    bsa_df = df[df["stage"] == "BSA"].copy()
    if bsa_df.empty:
        raise RuntimeError("No BSA rows found in features sheet1.")

    lookup: Dict[str, np.ndarray] = {}
    for _, row in bsa_df.iterrows():
        key = str(row["spectrum_col"])
        vals = np.array(
            [
                float(row["peak_wavelength_nm"]),
                float(row["peak_intensity_au"]),
                float(row["fwhm_nm"]),
            ],
            dtype=np.float32,
        )
        if np.all(np.isfinite(vals)):
            lookup[key] = vals
    return lookup


def derive_bsa_col(ag_col: str) -> str:
    return re.sub(r"-Ag-", "-BSA-", ag_col, flags=re.IGNORECASE)


def parse_training_data_fusion(
    data_path: str,
    bsa_lookup: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _load_table(data_path)
    if "Wavelength" not in df.columns:
        raise ValueError("Input data must contain 'Wavelength' column.")

    wl = pd.to_numeric(df["Wavelength"], errors="coerce").dropna().values.astype(np.float32)
    spectra = []
    physics = []
    concs = []

    wl_raw = df["Wavelength"].values.astype(np.float32)
    for col in df.columns:
        if col == "Wavelength" or "Ag" not in str(col):
            continue
        m = re.search(r"([0-9.]+)\s*ng/ml", str(col), flags=re.IGNORECASE)
        if not m:
            continue
        conc = float(m.group(1))

        bsa_col = derive_bsa_col(str(col))
        if bsa_col not in bsa_lookup:
            continue

        spec = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        valid = np.isfinite(spec) & np.isfinite(wl_raw)
        if np.count_nonzero(valid) < int(0.9 * len(wl)):
            continue
        spec_valid = spec[valid]
        wl_valid = wl_raw[valid]
        if spec_valid.size != wl.size:
            spec_valid = np.interp(wl, wl_valid, spec_valid).astype(np.float32)

        spectra.append(spec_valid)
        physics.append(bsa_lookup[bsa_col])
        concs.append(np.log10(conc + 1e-3))

    if not spectra:
        raise RuntimeError(f"No valid Ag spectra with BSA physics were parsed from: {data_path}")

    return (
        np.stack(spectra).astype(np.float32),
        np.stack(physics).astype(np.float32),
        np.asarray(concs, dtype=np.float32),
        wl,
    )


def compute_robust_stats(train_specs: np.ndarray, train_physics: np.ndarray) -> Dict[str, np.ndarray]:
    diff_specs = np.gradient(train_specs, axis=1)
    raw_med = np.median(train_specs, axis=0)
    raw_iqr = np.percentile(train_specs, 75, axis=0) - np.percentile(train_specs, 25, axis=0)
    raw_iqr = np.where(raw_iqr < 1e-6, 1.0, raw_iqr)

    diff_med = np.median(diff_specs, axis=0)
    diff_iqr = np.percentile(diff_specs, 75, axis=0) - np.percentile(diff_specs, 25, axis=0)
    diff_iqr = np.where(diff_iqr < 1e-6, 1.0, diff_iqr)

    phy_med = np.median(train_physics, axis=0)
    phy_iqr = np.percentile(train_physics, 75, axis=0) - np.percentile(train_physics, 25, axis=0)
    phy_iqr = np.where(phy_iqr < 1e-6, 1.0, phy_iqr)

    return {
        "raw_med": raw_med.astype(np.float32),
        "raw_iqr": raw_iqr.astype(np.float32),
        "diff_med": diff_med.astype(np.float32),
        "diff_iqr": diff_iqr.astype(np.float32),
        "phy_med": phy_med.astype(np.float32),
        "phy_iqr": phy_iqr.astype(np.float32),
    }


def build_input_channels(specs: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    diff_specs = np.gradient(specs, axis=1)
    raw_norm = (specs - stats["raw_med"][None, :]) / (stats["raw_iqr"][None, :] + 1e-8)
    diff_norm = (diff_specs - stats["diff_med"][None, :]) / (stats["diff_iqr"][None, :] + 1e-8)
    return np.stack([raw_norm, diff_norm], axis=1).astype(np.float32)


def build_physics_inputs(physics: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((physics - stats["phy_med"][None, :]) / (stats["phy_iqr"][None, :] + 1e-8)).astype(np.float32)


def monotonic_penalty(pred_logc: torch.Tensor, true_logc: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(true_logc.squeeze(1))
    sorted_pred = pred_logc[order].squeeze(1)
    violations = torch.relu(sorted_pred[:-1] - sorted_pred[1:])
    return violations.mean()


def evaluate_regression(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true = np.power(10.0, y_true_log) - 1e-3
    y_pred = np.power(10.0, y_pred_log) - 1e-3
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(abs_err / np.maximum(y_true, 1e-8)) * 100.0)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_pred - y_true) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"mae_ng_ml": mae, "rmse_ng_ml": rmse, "mape_pct": mape, "r2": r2}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_split_dir = os.path.join(root, "data", "processed", "splits_reconstructed")
    parser = argparse.ArgumentParser(description="Train SpectralPredictorV2_Fusion with fixed split.")
    parser.add_argument(
        "--train",
        type=str,
        default=os.path.join(default_split_dir, "train_preprocessed_pairs.xlsx"),
        help="Path to train split file (.xlsx/.csv)",
    )
    parser.add_argument(
        "--val",
        type=str,
        default=os.path.join(default_split_dir, "val_preprocessed_pairs.xlsx"),
        help="Path to validation split file (.xlsx/.csv)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.path.join(default_split_dir, "test_preprocessed_pairs.xlsx"),
        help="Path to test split file (.xlsx/.csv)",
    )
    parser.add_argument(
        "--features-xlsx",
        type=str,
        default=os.path.join(root, "data", "processed", "Reconstructed_Preprocessed_Features_and_Delta.xlsx"),
        help="Path to feature excel (sheet1 includes BSA peak features).",
    )
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=20260325, help="Random seed")
    parser.add_argument("--mono-weight", type=float, default=0.05, help="Monotonic penalty weight")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _models_root, checkpoints_dir, pretrained_dir = _prepare_model_dirs(root)
    train_path = os.path.abspath(args.train)
    val_path = os.path.abspath(args.val)
    test_path = os.path.abspath(args.test)
    features_path = os.path.abspath(args.features_xlsx)

    print(f"Train split: {train_path}")
    print(f"Val split:   {val_path}")
    print(f"Test split:  {test_path}")
    print(f"Features:    {features_path}")

    bsa_lookup = load_bsa_feature_lookup(features_path)
    specs_train, phy_train, y_train, wavelengths_train = parse_training_data_fusion(train_path, bsa_lookup)
    specs_val, phy_val, y_val, wavelengths_val = parse_training_data_fusion(val_path, bsa_lookup)
    specs_test, phy_test, y_test, wavelengths_test = parse_training_data_fusion(test_path, bsa_lookup)

    if len(wavelengths_train) != len(wavelengths_val) or len(wavelengths_train) != len(wavelengths_test):
        raise RuntimeError("Train/val/test wavelength lengths are inconsistent.")
    if not (np.allclose(wavelengths_train, wavelengths_val) and np.allclose(wavelengths_train, wavelengths_test)):
        raise RuntimeError("Train/val/test wavelength grids are not identical.")
    wavelengths = wavelengths_train

    stats = compute_robust_stats(specs_train, phy_train)
    x_train = build_input_channels(specs_train, stats)
    x_val = build_input_channels(specs_val, stats)
    x_test = build_input_channels(specs_test, stats)
    p_train = build_physics_inputs(phy_train, stats)
    p_val = build_physics_inputs(phy_val, stats)
    p_test = build_physics_inputs(phy_test, stats)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(p_train),
        torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(p_val),
        torch.from_numpy(y_val.reshape(-1, 1).astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(p_test),
        torch.from_numpy(y_test.reshape(-1, 1).astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SpectralPredictorV2_Fusion(seq_len=x_train.shape[-1]).to(device)
    criterion = nn.HuberLoss(delta=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    checkpoint_interval = 20

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for xb, pb, yb in train_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, pb)
            loss_main = criterion(pred, yb)
            loss_mono = monotonic_penalty(pred, yb)
            loss = loss_main + float(args.mono_weight) * loss_mono
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, pb, yb in val_loader:
                xb = xb.to(device)
                pb = pb.to(device)
                yb = yb.to(device)
                pred = model(xb, pb)
                loss_main = nn.functional.mse_loss(pred, yb)
                loss_mono = monotonic_penalty(pred, yb)
                val_losses.append((loss_main + float(args.mono_weight) * loss_mono).item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        scheduler.step(mean_val)

        if mean_val < best_val:
            best_val = mean_val
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"predictor_v2_fusion_epoch_{epoch + 1:04d}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "stats": stats,
                    "wavelengths": wavelengths.astype(np.float32),
                    "train_loss": mean_train,
                    "val_loss": mean_val,
                },
                ckpt_path,
            )

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | train={mean_train:.5f} | val={mean_val:.5f} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_test = []
    true_test = []
    with torch.no_grad():
        for xb, pb, yb in test_loader:
            xb = xb.to(device)
            pb = pb.to(device)
            pred = model(xb, pb).cpu().numpy().reshape(-1)
            preds_test.append(pred)
            true_test.append(yb.numpy().reshape(-1))
    y_pred_test = np.concatenate(preds_test, axis=0)
    y_true_test = np.concatenate(true_test, axis=0)
    test_metrics = evaluate_regression(y_true_test, y_pred_test)

    best_ckpt_path = os.path.join(checkpoints_dir, "predictor_v2_fusion_best.pth")
    if best_state is not None:
        torch.save(
            {
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
                "model_state_dict": best_state,
                "stats": stats,
                "wavelengths": wavelengths.astype(np.float32),
            },
            best_ckpt_path,
        )

    model_path = os.path.join(pretrained_dir, "spectral_predictor_v2_fusion.pth")
    params_path = os.path.join(pretrained_dir, "predictor_v2_fusion_norm_params.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(
        {
            "raw_med": stats["raw_med"],
            "raw_iqr": stats["raw_iqr"],
            "diff_med": stats["diff_med"],
            "diff_iqr": stats["diff_iqr"],
            "phy_med": stats["phy_med"],
            "phy_iqr": stats["phy_iqr"],
            "wavelengths": wavelengths.astype(np.float32),
        },
        params_path,
    )

    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "split_test_metrics_predictor_v2_fusion.csv")
    detail_path = os.path.join(out_dir, "split_test_predictions_predictor_v2_fusion.csv")

    pd.DataFrame(
        [
            {
                "train_file": train_path,
                "val_file": val_path,
                "test_file": test_path,
                "features_file": features_path,
                "samples_train_ag": int(len(y_train)),
                "samples_val_ag": int(len(y_val)),
                "samples_test_ag": int(len(y_test)),
                "seed": int(args.seed),
                "mono_weight": float(args.mono_weight),
                **test_metrics,
            }
        ]
    ).to_csv(metrics_path, index=False)

    pd.DataFrame(
        {
            "true_log10_conc": y_true_test,
            "pred_log10_conc": y_pred_test,
            "true_ng_ml": np.clip(np.power(10.0, y_true_test) - 1e-3, a_min=0.0, a_max=None),
            "pred_ng_ml": np.clip(np.power(10.0, y_pred_test) - 1e-3, a_min=0.0, a_max=None),
        }
    ).to_csv(detail_path, index=False)

    print(f"Saved: {model_path}")
    print(f"Saved: {params_path}")
    print("Test metrics:")
    print(
        f"  MAE={test_metrics['mae_ng_ml']:.6f} ng/ml | "
        f"RMSE={test_metrics['rmse_ng_ml']:.6f} ng/ml | "
        f"MAPE={test_metrics['mape_pct']:.3f}% | "
        f"R2={test_metrics['r2']:.6f}"
    )
    print(f"Saved test metrics: {metrics_path}")
    print(f"Saved test details: {detail_path}")


if __name__ == "__main__":
    train()
