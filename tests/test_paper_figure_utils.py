import math

from src.core.paper_figure_utils import build_model_summary, compute_mvr_from_predictions


def test_compute_mvr_from_predictions_counts_adjacent_descents():
    true_values = [0.5, 1.0, 5.0, 10.0]
    pred_values = [0.5, 1.2, 0.9, 2.0]
    rate = compute_mvr_from_predictions(true_values, pred_values)
    assert math.isclose(rate, 1.0 / 3.0, rel_tol=1e-6)


def test_build_model_summary_respects_requested_order():
    rows = [
        {"model": "Model D", "mae_mean": 6.5},
        {"model": "Model B", "mae_mean": 6.4},
        {"model": "Model C", "mae_mean": 5.6},
    ]
    summary = build_model_summary(rows, ["Model B", "Model C", "Model D"])
    assert summary["model"].tolist() == ["Model B", "Model C", "Model D"]
