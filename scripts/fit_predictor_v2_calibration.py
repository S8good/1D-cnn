import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.full_spectrum_models import SpectralPredictorV2


def _load_torch(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _resolve_artifact_path(models_root: str, file_name: str) -> str:
    candidates = [
        os.path.join(models_root, "pretrained", file_name),
        os.path.join(models_root, file_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing artifact: {file_name} (searched: {candidates})")


def parse_training_data(xlsx_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_excel(xlsx_path, sheet_name=0)
    if "Wavelength" not in df.columns:
        raise ValueError("Training Excel must contain 'Wavelength' column.")

    wl = pd.to_numeric(df["Wavelength"], errors="coerce").dropna().values.astype(np.float32)
    spectra: List[np.ndarray] = []
    log_concs: List[float] = []
    true_concs: List[float] = []

    for col in df.columns:
        if col == "Wavelength" or "Ag" not in str(col):
            continue
        m = re.search(r"([0-9.]+)\s*ng/ml", str(col), flags=re.IGNORECASE)
        if not m:
            continue
        c = float(m.group(1))
        spec = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        valid = np.isfinite(spec) & np.isfinite(df["Wavelength"].values.astype(np.float32))
        if np.count_nonzero(valid) < int(0.9 * len(wl)):
            continue
        spec_valid = spec[valid]
        wl_valid = df["Wavelength"].values.astype(np.float32)[valid]
        if spec_valid.size != wl.size:
            spec_valid = np.interp(wl, wl_valid, spec_valid).astype(np.float32)
        spectra.append(spec_valid)
        true_concs.append(c)
        log_concs.append(np.log10(c + 1e-3))

    return np.stack(spectra), np.asarray(log_concs, dtype=np.float32), np.asarray(true_concs, dtype=np.float32)


def build_input_channels(specs: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    diff = np.gradient(specs, axis=1)
    raw = (specs - params["raw_med"][None, :]) / (params["raw_iqr"][None, :] + 1e-8)
    d = (diff - params["diff_med"][None, :]) / (params["diff_iqr"][None, :] + 1e-8)
    return np.stack([raw, d], axis=1).astype(np.float32)


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root, "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    pretrained_dir = os.path.join(models_dir, "pretrained")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)
    data_path = os.path.join(root, "data", "processed", "All_Absorbance_Spectra_Preprocessed.xlsx")

    model_path = _resolve_artifact_path(models_dir, "spectral_predictor_v2.pth")
    norm_path = _resolve_artifact_path(models_dir, "predictor_v2_norm_params.pth")
    out_path = os.path.join(pretrained_dir, "predictor_v2_calibration.pth")

    params = _load_torch(norm_path)
    specs, y_true_log, y_true_conc = parse_training_data(data_path)
    x = build_input_channels(specs, params)

    model = SpectralPredictorV2(seq_len=x.shape[-1])
    model.load_state_dict(_load_torch(model_path))
    model.eval()

    with torch.no_grad():
        pred_log = model(torch.from_numpy(x)).squeeze(1).cpu().numpy()

    # Robust anchor fitting: median prediction per true concentration.
    anchors = []
    for c in sorted(np.unique(y_true_conc)):
        mask = y_true_conc == c
        anchors.append((float(np.median(pred_log[mask])), float(np.log10(c + 1e-3))))
    anchors = sorted(anchors, key=lambda t: t[0])

    x_anchor = np.asarray([a[0] for a in anchors], dtype=np.float64)
    y_anchor = np.asarray([a[1] for a in anchors], dtype=np.float64)

    uniq_x, uniq_idx = np.unique(x_anchor, return_index=True)
    uniq_y = y_anchor[uniq_idx]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(uniq_x, uniq_y)

    cal_x = iso.X_thresholds_.astype(np.float32)
    cal_y = iso.y_thresholds_.astype(np.float32)
    torch.save({"x_thresholds": cal_x, "y_thresholds": cal_y}, out_path)
    torch.save(
        {
            "source_model": model_path,
            "source_norm": norm_path,
            "x_thresholds": cal_x,
            "y_thresholds": cal_y,
        },
        os.path.join(checkpoints_dir, "predictor_v2_calibration_latest.pth"),
    )

    pred_log_cal = np.interp(pred_log, cal_x, cal_y, left=cal_y[0], right=cal_y[-1])
    pred_conc_raw = np.maximum(0.0, (10 ** pred_log) - 1e-3)
    pred_conc_cal = np.maximum(0.0, (10 ** pred_log_cal) - 1e-3)

    mape_raw = float(np.mean(np.abs(pred_conc_raw - y_true_conc) / y_true_conc * 100.0))
    mape_cal = float(np.mean(np.abs(pred_conc_cal - y_true_conc) / y_true_conc * 100.0))
    mae_raw = float(np.mean(np.abs(pred_conc_raw - y_true_conc)))
    mae_cal = float(np.mean(np.abs(pred_conc_cal - y_true_conc)))

    print(f"Loaded predictor from: {model_path}")
    print(f"Loaded norm params from: {norm_path}")
    print(f"Saved calibration: {out_path}")
    print(f"Train-set MAE raw -> cal:  {mae_raw:.6f} -> {mae_cal:.6f}")
    print(f"Train-set MAPE raw -> cal: {mape_raw:.6f}% -> {mape_cal:.6f}%")
    print(f"Threshold count: {len(cal_x)}")


if __name__ == "__main__":
    main()
