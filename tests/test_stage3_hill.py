import math

import pandas as pd
import pytest
import torch

from src.core.stage3_hill import (
    FixedHillCurve,
    LearnableHillCurve,
    build_delta_lambda_table,
    hill_delta_lambda,
    soft_argmax_peak_nm,
)


def test_hill_delta_lambda_matches_expected_scalar_value():
    conc = torch.tensor([[10.0]], dtype=torch.float32)
    result = hill_delta_lambda(concentration_ng_ml=conc, delta_lambda_max=8.0, k_half=10.0, hill_n=1.0)
    assert torch.allclose(result, torch.tensor([[4.0]], dtype=torch.float32), atol=1e-6)


def test_hill_delta_lambda_clamps_negative_concentration_to_zero():
    conc = torch.tensor([[-5.0]], dtype=torch.float32)
    result = hill_delta_lambda(concentration_ng_ml=conc, delta_lambda_max=8.0, k_half=10.0, hill_n=1.0)
    assert torch.allclose(result, torch.tensor([[0.0]], dtype=torch.float32), atol=1e-6)


def test_build_delta_lambda_table_pairs_bsa_and_ag_rows():
    df = pd.DataFrame(
        [
            {"sample_id": "S2", "stage": "Ag", "concentration_ng_ml": 2.0, "peak_wavelength_nm": 606.0},
            {"sample_id": "S2", "stage": "BSA", "concentration_ng_ml": 2.0, "peak_wavelength_nm": 603.0},
            {"sample_id": "S1", "stage": "BSA", "concentration_ng_ml": 1.0, "peak_wavelength_nm": 602.0},
            {"sample_id": "S1", "stage": "Ag", "concentration_ng_ml": 1.0, "peak_wavelength_nm": 604.5},
        ]
    )
    paired = build_delta_lambda_table(df)
    assert list(paired.columns) == [
        "sample_id",
        "concentration_ng_ml",
        "lambda_bsa_nm",
        "lambda_ag_nm",
        "delta_lambda_nm",
    ]
    assert paired.iloc[0]["delta_lambda_nm"] == 2.5
    assert list(paired["sample_id"]) == ["S1", "S2"]


def test_build_delta_lambda_table_requires_expected_columns():
    with pytest.raises(KeyError):
        build_delta_lambda_table(pd.DataFrame([{"sample_id": "S1"}]))


def test_soft_argmax_peak_nm_recovers_single_peak_inside_window():
    wavelengths = torch.tensor([600.0, 601.0, 602.0, 603.0, 604.0], dtype=torch.float32)
    spectra = torch.tensor([[0.0, 0.0, 1.0, 4.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([False, True, True, True, False])
    peak_nm = soft_argmax_peak_nm(spectra, wavelengths, window_mask=mask, temperature=0.25)
    assert 602.4 <= float(peak_nm.item()) <= 603.2


def test_learnable_hill_curve_stays_positive_and_has_regularization():
    module = LearnableHillCurve(delta_lambda_max=8.0, k_half=10.0, hill_n=1.4)
    delta_lambda_max, k_half, hill_n = module.constrained_parameters()
    out = module(torch.tensor([[5.0]], dtype=torch.float32))
    assert float(delta_lambda_max.item()) > 0.0
    assert float(k_half.item()) > 0.0
    assert float(hill_n.item()) > 0.0
    assert float(out.item()) > 0.0
    assert float(module.regularization_loss().item()) == 0.0


def test_fixed_hill_curve_matches_closed_form():
    module = FixedHillCurve(delta_lambda_max=8.0, k_half=10.0, hill_n=1.0)
    out = module(torch.tensor([[10.0]], dtype=torch.float32))
    assert math.isclose(float(out.item()), 4.0, rel_tol=1e-6)
    assert float(module.regularization_loss().item()) == 0.0
