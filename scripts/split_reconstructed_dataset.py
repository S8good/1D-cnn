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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_csv = root.parents[1] / "CEA_build" / "Reconstructed_Preprocessed_Spectra.csv"
    out_dir = root / "data" / "processed" / "splits_reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    random_state = 20260324

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")

    df = pd.read_csv(input_csv)
    if "Wavelength" not in df.columns:
        raise ValueError("Input CSV must contain 'Wavelength' column.")

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

    # Pair-wise split with concentration stratification.
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
            keep_cols.append(row["bsa_col"])
            keep_cols.append(row["ag_col"])

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

        df_split = df[keep_cols].copy()
        df_split.to_excel(out_xlsx, index=False)
        df_split.to_csv(out_csv, index=False)

    assignment = pd.DataFrame(assignment_rows).sort_values(
        ["split", "concentration_ng_ml", "sid"]
    )
    assignment_file = out_dir / "split_assignment.csv"
    assignment.to_csv(assignment_file, index=False)

    # Summary output
    print(f"Input file: {input_csv}")
    print(f"Output dir: {out_dir}")
    print(f"Total pairs: {len(pair_df)}")
    for split_name, split_indices in split_map.items():
        sub = pair_df.loc[split_indices]
        print(f"{split_name}: {len(sub)} pairs ({len(sub) * 2} spectra)")
        counts = sub.groupby("conc").size().to_dict()
        print(f"  per concentration: {counts}")
    print(f"Saved split assignment: {assignment_file}")


if __name__ == "__main__":
    main()
