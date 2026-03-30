import math

import pytest

from src.core.stage25_config import Stage3GateDecision, build_stage25_profile, evaluate_stage3_gate


def test_build_stage25_profile_a_has_expected_aggressive_defaults():
    profile = build_stage25_profile("2.5A")

    assert profile.name == "2.5A"
    assert profile.update_strategy == "alternating"
    assert profile.predictor_train_mode == "tail"
    assert profile.p_steps == 2
    assert profile.g_steps == 1
    assert math.isclose(profile.predictor_lr, 2e-4)
    assert math.isclose(profile.generator_lr, 5e-4)
    assert math.isclose(profile.w_cycle, 0.01)


def test_evaluate_stage3_gate_returns_route_b_for_non_degrading_candidate():
    decision = evaluate_stage3_gate(
        baseline={"mae_ng_ml": 6.0, "rmse_ng_ml": 12.0, "r2": 0.80},
        candidate={"mae_ng_ml": 6.08, "rmse_ng_ml": 12.18, "r2": 0.795},
        paired_mae_wins=3,
        seed_count=3,
        std_ratio=0.95,
        training_success_rate=1.0,
        generator_collapse=False,
    )

    assert isinstance(decision, Stage3GateDecision)
    assert decision.route == "B"
    assert decision.can_enter_stage3 is True
    assert "稳定" in decision.narrative


def test_build_stage25_profile_raises_for_invalid_profile():
    with pytest.raises(ValueError):
        build_stage25_profile("2.5Z")


def test_evaluate_stage3_gate_blocks_zero_baseline_degradation():
    decision = evaluate_stage3_gate(
        baseline={"mae_ng_ml": 0.0, "rmse_ng_ml": 0.0, "r2": 0.80},
        candidate={"mae_ng_ml": 0.1, "rmse_ng_ml": 0.2, "r2": 0.80},
        paired_mae_wins=3,
        seed_count=3,
        std_ratio=0.90,
        training_success_rate=1.0,
        generator_collapse=False,
    )

    assert decision.can_enter_stage3 is False
    assert decision.route is None


def test_evaluate_stage3_gate_uses_seed_count_for_route_a_threshold():
    decision = evaluate_stage3_gate(
        baseline={"mae_ng_ml": 10.0, "rmse_ng_ml": 20.0, "r2": 0.80},
        candidate={"mae_ng_ml": 9.0, "rmse_ng_ml": 18.0, "r2": 0.80},
        paired_mae_wins=2,
        seed_count=5,
        std_ratio=1.2,
        training_success_rate=1.0,
        generator_collapse=False,
    )

    assert decision.route != "A"
