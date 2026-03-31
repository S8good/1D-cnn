from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn


def hill_delta_lambda(
    concentration_ng_ml: torch.Tensor,
    delta_lambda_max: float | torch.Tensor,
    k_half: float | torch.Tensor,
    hill_n: float | torch.Tensor,
) -> torch.Tensor:
    conc = torch.clamp(concentration_ng_ml, min=0.0)
    dl_max = torch.as_tensor(delta_lambda_max, dtype=conc.dtype, device=conc.device)
    k = torch.as_tensor(k_half, dtype=conc.dtype, device=conc.device)
    n = torch.as_tensor(hill_n, dtype=conc.dtype, device=conc.device)
    conc_pow = torch.pow(conc, n)
    k_pow = torch.pow(k, n)
    return dl_max * conc_pow / (k_pow + conc_pow + 1e-12)


def build_delta_lambda_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "stage", "concentration_ng_ml", "peak_wavelength_nm"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    table = df.copy()
    table["stage"] = table["stage"].astype(str).str.upper()
    bsa = table[table["stage"] == "BSA"][["sample_id", "concentration_ng_ml", "peak_wavelength_nm"]].rename(
        columns={"peak_wavelength_nm": "lambda_bsa_nm"}
    )
    ag = table[table["stage"] == "AG"][["sample_id", "concentration_ng_ml", "peak_wavelength_nm"]].rename(
        columns={"peak_wavelength_nm": "lambda_ag_nm"}
    )
    merged = bsa.merge(ag, on=["sample_id", "concentration_ng_ml"], how="inner")
    merged["delta_lambda_nm"] = merged["lambda_ag_nm"] - merged["lambda_bsa_nm"]
    return merged[
        [
            "sample_id",
            "concentration_ng_ml",
            "lambda_bsa_nm",
            "lambda_ag_nm",
            "delta_lambda_nm",
        ]
    ].sort_values(["concentration_ng_ml", "sample_id"]).reset_index(drop=True)


def soft_argmax_peak_nm(
    spectra: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    window_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if spectra.ndim == 1:
        spectra = spectra.unsqueeze(0)
    if spectra.ndim != 2:
        raise ValueError("spectra must be shape [batch, length] or [length]")
    if wavelengths_nm.ndim != 1 or wavelengths_nm.shape[0] != spectra.shape[1]:
        raise ValueError("wavelengths_nm must be shape [length] matching spectra")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = spectra
    if window_mask is not None:
        if window_mask.ndim != 1 or window_mask.shape[0] != spectra.shape[1]:
            raise ValueError("window_mask must be shape [length] matching spectra")
        if not bool(window_mask.any()):
            raise ValueError("window_mask must include at least one True value")
        fill = torch.finfo(logits.dtype).min
        logits = torch.where(window_mask.unsqueeze(0), logits, torch.full_like(logits, fill))

    weights = torch.softmax(logits / temperature, dim=1)
    return (weights * wavelengths_nm.unsqueeze(0)).sum(dim=1, keepdim=True)


class LearnableHillCurve(nn.Module):
    def __init__(self, delta_lambda_max: float, k_half: float, hill_n: float):
        super().__init__()
        self.delta_lambda_max_raw = nn.Parameter(self._inverse_softplus(float(delta_lambda_max)))
        self.k_half_raw = nn.Parameter(self._inverse_softplus(float(k_half)))
        self.hill_n_raw = nn.Parameter(self._inverse_softplus(float(hill_n)))
        self.register_buffer("delta_lambda_max_init", torch.tensor(float(delta_lambda_max), dtype=torch.float32))
        self.register_buffer("k_half_init", torch.tensor(float(k_half), dtype=torch.float32))
        self.register_buffer("hill_n_init", torch.tensor(float(hill_n), dtype=torch.float32))

    def _positive_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_lambda_max = torch.nn.functional.softplus(self.delta_lambda_max_raw)
        k_half = torch.nn.functional.softplus(self.k_half_raw)
        hill_n = torch.nn.functional.softplus(self.hill_n_raw)
        return delta_lambda_max, k_half, hill_n

    def constrained_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._positive_params()

    @staticmethod
    def _inverse_softplus(value: float) -> torch.Tensor:
        return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))

    def forward(self, concentration_ng_ml: torch.Tensor) -> torch.Tensor:
        delta_lambda_max, k_half, hill_n = self.constrained_parameters()
        return hill_delta_lambda(concentration_ng_ml, delta_lambda_max, k_half, hill_n)

    def regularization_loss(self) -> torch.Tensor:
        delta_lambda_max, k_half, hill_n = self.constrained_parameters()
        loss = (
            (delta_lambda_max - self.delta_lambda_max_init) ** 2
            + (k_half - self.k_half_init) ** 2
            + (hill_n - self.hill_n_init) ** 2
        )
        return torch.where(loss < 1e-12, torch.zeros_like(loss), loss)


class FixedHillCurve(nn.Module):
    def __init__(self, delta_lambda_max: float, k_half: float, hill_n: float):
        super().__init__()
        self.register_buffer("delta_lambda_max", torch.tensor(float(delta_lambda_max)))
        self.register_buffer("k_half", torch.tensor(float(k_half)))
        self.register_buffer("hill_n", torch.tensor(float(hill_n)))

    def forward(self, concentration_ng_ml: torch.Tensor) -> torch.Tensor:
        return hill_delta_lambda(concentration_ng_ml, self.delta_lambda_max, self.k_half, self.hill_n)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=self.delta_lambda_max.dtype, device=self.delta_lambda_max.device)
