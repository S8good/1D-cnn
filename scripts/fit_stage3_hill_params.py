import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.stage3_hill import build_delta_lambda_table


def fit_fixed_hill_params(delta_df: pd.DataFrame):
    grouped = delta_df.groupby("concentration_ng_ml", as_index=False)["delta_lambda_nm"].median()
    conc = torch.tensor(grouped["concentration_ng_ml"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    target = torch.tensor(grouped["delta_lambda_nm"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    raw = torch.nn.Parameter(torch.tensor([5.0, 10.0, 0.5], dtype=torch.float32))
    opt = torch.optim.Adam([raw], lr=0.05)

    for _ in range(1500):
        opt.zero_grad()
        delta_max = torch.nn.functional.softplus(raw[0])
        k_half = torch.nn.functional.softplus(raw[1])
        hill_n = torch.nn.functional.softplus(raw[2])
        pred = delta_max * conc.pow(hill_n) / (k_half.pow(hill_n) + conc.pow(hill_n) + 1e-8)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()

    return {
        "delta_lambda_max": float(delta_max.detach()),
        "k_half": float(k_half.detach()),
        "hill_n": float(hill_n.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-xlsx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_excel(args.features_xlsx, sheet_name="sheet1")
    params = fit_fixed_hill_params(build_delta_lambda_table(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, args.out)


if __name__ == "__main__":
    main()
