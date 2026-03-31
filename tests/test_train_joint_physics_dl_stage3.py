from argparse import Namespace

from scripts import train_joint_physics_dl as joint_script
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
