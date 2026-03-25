import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


COL_PATTERN = re.compile(
    r"^(?P<conc>\d+(?:\.\d+)?)ng/ml-(?P<stage>BSA|Ag)-(?P<sid>.+)$",
    re.IGNORECASE,
)


def parse_column(col: str) -> Tuple[float, str, str]:
    m = COL_PATTERN.match(col)
    if not m:
        raise ValueError(f"Invalid column format: {col}")
    return float(m.group("conc")), m.group("stage").upper(), m.group("sid")


def build_pair_df_from_spectra(df: pd.DataFrame) -> pd.DataFrame:
    all_cols = [c for c in df.columns if c != "Wavelength"]
    pair_map: Dict[Tuple[float, str], Dict[str, str]] = {}
    for col in all_cols:
        conc, stage, sid = parse_column(col)
        key = (conc, sid)
        if key not in pair_map:
            pair_map[key] = {}
        pair_map[key][stage] = col

    pair_rows: List[Tuple[float, str, str, str]] = []
    for (conc, sid), stages in pair_map.items():
        if "BSA" not in stages or "AG" not in stages:
            continue
        pair_rows.append((conc, sid, stages["BSA"], stages["AG"]))

    if not pair_rows:
        raise RuntimeError("No complete BSA/Ag pairs found in input CSV.")

    pair_df = pd.DataFrame(pair_rows, columns=["conc", "sid", "bsa_col", "ag_col"]).sort_values(
        ["conc", "sid"]
    )
    pair_df["sid"] = pair_df["sid"].astype(str)
    return pair_df


def load_quality_sheet(features_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(features_path, sheet_name="sheet2")
    required_cols = ["concentration_ng_ml", "sample_id", "bsa_col", "ag_col", "within_cv10"]
    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing columns in sheet2: {missing}")

    meta = meta[required_cols].copy()
    meta["conc"] = meta["concentration_ng_ml"].astype(float)
    meta["sid"] = meta["sample_id"].astype(str)
    meta["within_cv10"] = meta["within_cv10"].astype(bool)
    return meta


def split_and_save(
    pair_df: pd.DataFrame,
    spectra_df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = pair_df.index.to_list()
    labels = pair_df["conc"].astype(str).to_list()

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        stratify=labels,
    )
    temp_df = pair_df.loc[temp_idx].copy()
    val_frac_in_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_df.index.to_list(),
        test_size=(1.0 - val_frac_in_temp),
        random_state=random_state,
        stratify=temp_df["conc"].astype(str).to_list(),
    )

    split_map = {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }
    assignment_rows = []
    for split_name, split_indices in split_map.items():
        sub = pair_df.loc[split_indices]
        keep_cols = ["Wavelength"]
        for _, row in sub.iterrows():
            keep_cols.extend([row["bsa_col"], row["ag_col"]])
            assignment_rows.append(
                {
                    "split": split_name,
                    "concentration_ng_ml": row["conc"],
                    "sid": row["sid"],
                    "bsa_column": row["bsa_col"],
                    "ag_column": row["ag_col"],
                }
            )

        out_xlsx = out_dir / f"{split_name}_preprocessed_pairs.xlsx"
        out_csv = out_dir / f"{split_name}_preprocessed_pairs.csv"
        df_split = spectra_df[keep_cols].copy()
        df_split.to_excel(out_xlsx, index=False)
        df_split.to_csv(out_csv, index=False)

    assignment = pd.DataFrame(assignment_rows).sort_values(["split", "concentration_ng_ml", "sid"])
    assignment_file = out_dir / "split_assignment.csv"
    assignment.to_csv(assignment_file, index=False)

    print(f"Output dir: {out_dir}")
    print(f"Total pairs: {len(pair_df)}")
    for split_name, split_indices in split_map.items():
        sub = pair_df.loc[split_indices]
        counts = sub.groupby("conc").size().to_dict()
        print(f"{split_name}: {len(sub)} pairs ({len(sub) * 2} spectra)")
        print(f"  per concentration: {counts}")
    print(f"Saved split assignment: {assignment_file}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_csv = root / "data" / "processed" / "Reconstructed_Preprocessed_Spectra.csv"
    features_xlsx = root / "data" / "processed" / "Reconstructed_Preprocessed_Features_and_Delta.xlsx"
    out_dir_all = root / "data" / "processed" / "splits_reconstructed_all"
    out_dir_cv10 = root / "data" / "processed" / "splits_reconstructed_cv10"
    out_dir_legacy = root / "data" / "processed" / "splits_reconstructed"

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    random_state = 20260324

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")

    spectra_df = pd.read_csv(input_csv)
    if "Wavelength" not in spectra_df.columns:
        raise ValueError("Input CSV must contain 'Wavelength' column.")

    print(f"Input file: {input_csv}")
    print(f"Features file: {features_xlsx}")

    pair_df = build_pair_df_from_spectra(spectra_df)
    quality_df = load_quality_sheet(features_xlsx)

    merged = pair_df.merge(
        quality_df[["conc", "sid", "within_cv10", "bsa_col", "ag_col"]],
        on=["conc", "sid"],
        how="inner",
        suffixes=("", "_meta"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping pairs between spectra and features metadata.")

    # Use metadata column mapping as the canonical source for split assignment.
    merged["bsa_col"] = merged["bsa_col_meta"]
    merged["ag_col"] = merged["ag_col_meta"]
    merged = merged.drop(columns=["bsa_col_meta", "ag_col_meta"]).sort_values(["conc", "sid"])

    pair_df_all = merged[["conc", "sid", "bsa_col", "ag_col", "within_cv10"]].copy()
    pair_df_cv10 = pair_df_all[pair_df_all["within_cv10"]].copy()
    if pair_df_cv10.empty:
        raise RuntimeError("No samples available after within_cv10 filtering.")

    print("\n=== Split set: all pairs ===")
    split_and_save(
        pair_df=pair_df_all[["conc", "sid", "bsa_col", "ag_col"]],
        spectra_df=spectra_df,
        out_dir=out_dir_all,
        random_state=random_state,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print("\n=== Split set: within_cv10 pairs ===")
    split_and_save(
        pair_df=pair_df_cv10[["conc", "sid", "bsa_col", "ag_col"]],
        spectra_df=spectra_df,
        out_dir=out_dir_cv10,
        random_state=random_state,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Keep backward compatibility for existing training scripts.
    split_and_save(
        pair_df=pair_df_cv10[["conc", "sid", "bsa_col", "ag_col"]],
        spectra_df=spectra_df,
        out_dir=out_dir_legacy,
        random_state=random_state,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


if __name__ == "__main__":
    main()
