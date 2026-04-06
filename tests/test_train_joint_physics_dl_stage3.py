from argparse import Namespace

import math
from pathlib import Path
import torch

from scripts import train_joint_physics_dl as joint_script
from src.core.stage3_hill import FixedHillCurve, LearnableHillCurve
from src.core.stage3_config import build_stage3_profile


def test_apply_stage3_profile_overrides_training_args():
    args = Namespace(stage3_profile=None, predictor_train_mode="all", hill_mode="off", w_hill=0.0)
    profile = build_stage3_profile("3B-fixed-regressor")
    joint_script.apply_stage3_profile(args, profile)
    assert args.stage3_profile == "3B-fixed-regressor"
    assert args.predictor_train_mode == "regressor"
    assert args.hill_mode == "fixed"
    assert args.w_hill == 0.1


def test_build_output_tag_prefers_stage3_profile_name():
    assert joint_script.build_output_tag(stage25_profile=None, stage3_profile="3A-fixed-frozen") == "stage3_3a_fixed_frozen"


def test_run_joint_training_epoch_uses_stage3_runner_when_hill_enabled():
    calls = []

    def fake_stage3_runner(**kwargs):
        calls.append(kwargs)
        return {
            "predictor_steps": 0,
            "generator_steps": 1,
            "predictor_loss": 0.0,
            "generator_loss": 0.1,
            "generator_loss_hill": 0.02,
        }

    stats = joint_script.run_joint_training_epoch(
        predictor="predictor",
        generator="generator",
        train_loader=["batch"],
        predictor_optimizer=None,
        generator_optimizer="g-opt",
        update_strategy="alternating",
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.05,
        p_steps=0,
        g_steps=1,
        hill_weight=0.1,
        stage3_runner=fake_stage3_runner,
        hill_context={"enabled": True},
    )
    assert stats["generator_loss_hill"] == 0.02
    assert calls


def test_monotonicity_violation_rate_counts_adjacent_inversions():
    y_true_log = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32).numpy()
    y_pred_log = torch.tensor([0.0, 0.2, 0.1, 0.4], dtype=torch.float32).numpy()
    rate = joint_script.monotonicity_violation_rate(y_true_log, y_pred_log)
    assert math.isclose(rate, 1.0 / 3.0, rel_tol=1e-6)


def test_evaluate_hill_consistency_returns_small_error_for_matching_generator():
    class MatchingGenerator(torch.nn.Module):
        def forward(self, y_log):
            low_peak = torch.tensor([0.0, 6.0, 6.0, 0.0, 0.0], dtype=y_log.dtype, device=y_log.device)
            high_peak = torch.tensor([0.0, 0.0, 0.0, 0.0, 12.0], dtype=y_log.dtype, device=y_log.device)
            rows = []
            for value in y_log.squeeze(1):
                rows.append(low_peak if float(value.item()) < 0.5 else high_peak)
            return torch.stack(rows, dim=0).unsqueeze(1)

    wavelengths_nm = torch.tensor([601.0, 602.0, 603.0, 604.0, 605.0], dtype=torch.float32)
    physics_raw = torch.tensor([[600.0, 0.1, 0.2], [600.0, 0.1, 0.2]], dtype=torch.float32).numpy()
    y_true_log = torch.tensor(
        [
            [math.log10(1.0 + 1e-3)],
            [math.log10(10.0 + 1e-3)],
        ],
        dtype=torch.float32,
    ).numpy().reshape(-1)
    hill_curve = FixedHillCurve(delta_lambda_max=5.0, k_half=1.0, hill_n=10.0)

    metrics = joint_script.evaluate_hill_consistency(
        generator=MatchingGenerator(),
        y_true_log=y_true_log,
        physics_raw=physics_raw,
        wavelengths_nm=wavelengths_nm.numpy(),
        hill_curve=hill_curve,
        hill_window_center_nm=605.0,
        hill_window_half_width_nm=5.0,
        hill_temperature=0.01,
        device=torch.device("cpu"),
    )

    assert metrics["hill_consistency_mae_nm"] < 0.1


def test_build_joint_optimizer_includes_learnable_hill_curve_parameters():
    predictor = torch.nn.Linear(2, 1)
    generator = torch.nn.Linear(1, 2)
    hill_curve = joint_script.LearnableHillCurve(delta_lambda_max=5.0, k_half=1.0, hill_n=2.0)

    optimizer = joint_script.build_joint_optimizer(
        trainable_predictor_params=list(predictor.parameters()),
        generator=generator,
        predictor_lr=1e-4,
        generator_lr=2e-4,
        hill_lr=5e-4,
        hill_curve=hill_curve,
    )

    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert id(hill_curve.delta_lambda_max_raw) in optimizer_param_ids
    assert id(hill_curve.k_half_raw) in optimizer_param_ids
    assert id(hill_curve.hill_n_raw) in optimizer_param_ids
    hill_group = next(group for group in optimizer.param_groups if id(hill_curve.delta_lambda_max_raw) in {id(param) for param in group["params"]})
    assert math.isclose(hill_group["lr"], 5e-4)


def test_hill_curve_parameter_snapshot_returns_positive_learnable_parameters():
    hill_curve = LearnableHillCurve(delta_lambda_max=5.0, k_half=1.0, hill_n=2.0)
    snapshot = joint_script.hill_curve_parameter_snapshot(hill_curve)
    assert snapshot["delta_lambda_max"] > 0.0
    assert snapshot["k_half"] > 0.0
    assert snapshot["hill_n"] > 0.0


def test_resolve_generator_init_weights_prefers_explicit_override(tmp_path: Path):
    explicit = tmp_path / "stage3_3a_generator.pth"
    args = Namespace(stage3_profile="3C-learnable-regressor", generator_init_weights=str(explicit))
    resolved = joint_script.resolve_generator_init_weights(args)
    assert resolved == str(explicit)
