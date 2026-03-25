from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


FeatureExtractor = Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]]


@dataclass
class TrainingSamples:
    features: np.ndarray
    targets: np.ndarray
    baseline: Dict[str, float]
    wavelengths: np.ndarray
    source_path: str


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    if ext in [".csv", ".txt", ".tsv"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {ext}")


def find_training_source(base_dir: str) -> Tuple[str, Tuple[str, ...]]:
    split_train_xlsx = os.path.join(
        base_dir, "data", "processed", "splits_reconstructed", "train_preprocessed_pairs.xlsx"
    )
    split_train_csv = os.path.join(
        base_dir, "data", "processed", "splits_reconstructed", "train_preprocessed_pairs.csv"
    )
    filtered_path = os.path.join(base_dir, "data", "filtered", "cea_training_data_20pct.csv")
    spectra_path = os.path.join(base_dir, "data", "processed", "Reconstructed_Preprocessed_Spectra.csv")
    paired_path = os.path.join(base_dir, "data", "processed", "All_Absorbance_Spectra_Preprocessed.xlsx")

    if os.path.exists(split_train_xlsx):
        return "paired", (split_train_xlsx,)
    if os.path.exists(split_train_csv):
        return "paired", (split_train_csv,)
    if os.path.exists(filtered_path) and os.path.exists(spectra_path):
        return "legacy", (filtered_path, spectra_path)
    if os.path.exists(paired_path):
        return "paired", (paired_path,)

    raise FileNotFoundError(
        "No usable training source found: missing split train file, legacy CEA CSV, and "
        "All_Absorbance_Spectra_Preprocessed.xlsx."
    )


def _assert_columns(df: pd.DataFrame, columns: Tuple[str, ...], source_path: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in file: {source_path}")


def build_training_samples_from_paired_file(
    data_path: str,
    feature_extractor: FeatureExtractor,
) -> TrainingSamples:
    df = read_table(data_path)
    _assert_columns(df, ("Wavelength",), data_path)

    wl_series = pd.to_numeric(df["Wavelength"], errors="coerce")
    wl = wl_series.dropna().values.astype(np.float32)
    wl_raw = wl_series.values
    if wl.size == 0:
        raise ValueError(f"No valid numeric wavelengths in file: {data_path}")

    pattern = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)ng/ml-(Ag|BSA)-(.+)\s*$", re.IGNORECASE)
    pairs: Dict[Tuple[float, str], Dict[str, str]] = {}
    for col in df.columns:
        if col == "Wavelength":
            continue
        m = pattern.match(str(col))
        if not m:
            continue
        conc = float(m.group(1))
        phase = m.group(2).lower()
        rep = m.group(3).strip()
        pairs.setdefault((conc, rep), {})[phase] = col

    x_rows = []
    y_rows = []
    bsa_feature_rows = []
    for (conc, _rep), cols in pairs.items():
        if "bsa" not in cols or "ag" not in cols:
            continue

        pre_raw = pd.to_numeric(df[cols["bsa"]], errors="coerce").values
        post_raw = pd.to_numeric(df[cols["ag"]], errors="coerce").values
        valid = np.isfinite(pre_raw) & np.isfinite(post_raw) & np.isfinite(wl_raw)
        if np.count_nonzero(valid) < 10:
            continue

        pre = pre_raw[valid].astype(np.float32)
        post = post_raw[valid].astype(np.float32)
        wl_valid = wl_raw[valid].astype(np.float32)

        lambda_pre, a_pre, fwhm_pre = feature_extractor(wl_valid, pre)
        lambda_post, a_post, _fwhm_post = feature_extractor(wl_valid, post)

        x_rows.append([np.log10(conc + 1e-3), lambda_pre, a_pre, fwhm_pre])
        y_rows.append([lambda_post - lambda_pre, a_post - a_pre])
        bsa_feature_rows.append((lambda_pre, a_pre, fwhm_pre))

    if not x_rows:
        raise ValueError(f"No complete Ag/BSA column pairs found in file: {data_path}")

    if bsa_feature_rows:
        baseline = {
            "lambda": float(np.median([r[0] for r in bsa_feature_rows])),
            "A": float(np.median([r[1] for r in bsa_feature_rows])),
            "fwhm": float(np.median([r[2] for r in bsa_feature_rows])),
        }
    else:
        baseline = {"lambda": float(np.median(wl)), "A": 0.05, "fwhm": 80.0}

    return TrainingSamples(
        features=np.asarray(x_rows, dtype=np.float32),
        targets=np.asarray(y_rows, dtype=np.float32),
        baseline=baseline,
        wavelengths=wl,
        source_path=data_path,
    )


def build_training_samples_from_legacy_files(filtered_path: str, spectra_path: str) -> TrainingSamples:
    df_feat = pd.read_csv(filtered_path)
    df_spec = pd.read_csv(spectra_path)
    _assert_columns(
        df_feat,
        ("c_ng_ml", "lambda_peak_nm_pre", "Apeak_pre", "fwhm_nm_pre", "lambda_peak_nm_post", "Apeak_post"),
        filtered_path,
    )
    _assert_columns(df_spec, ("Wavelength",), spectra_path)

    wavelengths = pd.to_numeric(df_spec["Wavelength"], errors="coerce").dropna().values.astype(np.float32)
    if wavelengths.size == 0:
        raise ValueError(f"No valid wavelengths in file: {spectra_path}")

    x = np.column_stack(
        [
            np.log10(df_feat["c_ng_ml"].values.astype(np.float32) + 1e-3),
            df_feat["lambda_peak_nm_pre"].values.astype(np.float32),
            df_feat["Apeak_pre"].values.astype(np.float32),
            df_feat["fwhm_nm_pre"].values.astype(np.float32),
        ]
    )
    y = np.column_stack(
        [
            (df_feat["lambda_peak_nm_post"] - df_feat["lambda_peak_nm_pre"]).values.astype(np.float32),
            (df_feat["Apeak_post"] - df_feat["Apeak_pre"]).values.astype(np.float32),
        ]
    )
    baseline = {
        "lambda": float(df_feat.iloc[0]["lambda_peak_nm_pre"]),
        "A": float(df_feat.iloc[0]["Apeak_pre"]),
        "fwhm": float(df_feat.iloc[0]["fwhm_nm_pre"]),
    }
    return TrainingSamples(
        features=x.astype(np.float32),
        targets=y.astype(np.float32),
        baseline=baseline,
        wavelengths=wavelengths,
        source_path=f"{filtered_path};{spectra_path}",
    )


def read_spectrum_file(file_name: str) -> np.ndarray:
    ext = os.path.splitext(file_name)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_name, sheet_name=0)
    elif ext in [".txt", ".tsv"]:
        df = pd.read_csv(file_name, sep=None, engine="python")
    elif ext == ".csv":
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(file_name, sep=None, engine="python")

    if df.empty:
        raise ValueError("File is empty.")

    preferred_cols = ["absorbance", "intensity", "signal", "y"]
    for col in df.columns:
        if str(col).strip().lower() in preferred_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if vals.size > 0:
                return vals.astype(np.float32)

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
    if not valid_cols:
        raise ValueError("No numeric spectrum column found.")

    vals = numeric_df[valid_cols[-1]].dropna().values
    if vals.size == 0:
        raise ValueError("No valid numeric values found in spectrum column.")
    return vals.astype(np.float32)
