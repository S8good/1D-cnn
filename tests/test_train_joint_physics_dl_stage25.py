from argparse import Namespace

from src.core.stage25_config import Stage25Profile
from scripts import train_joint_physics_dl as joint_script


def test_apply_stage25_profile_overrides_training_args():
    args = Namespace(
        stage25_profile=None,
        update_strategy="joint",
        predictor_train_mode="all",
        predictor_lr=2e-4,
        generator_lr=1e-3,
        p_steps=1,
        g_steps=1,
        w_cycle=0.1,
        w_mono=0.05,
        w_recon=0.05,
    )
    profile = Stage25Profile("2.5B", "alternating", "regressor", 1e-4, 2e-4, 1, 1, 0.005, 0.05, 0.03)

    joint_script.apply_stage25_profile(args, profile)

    assert args.stage25_profile == "2.5B"
    assert args.update_strategy == "alternating"
    assert args.predictor_train_mode == "regressor"
    assert args.predictor_lr == 1e-4
    assert args.generator_lr == 2e-4
    assert args.w_cycle == 0.005
    assert args.w_recon == 0.03


def test_run_joint_training_epoch_uses_alternating_strategy():
    calls = []

    def fake_run_alternating_epoch(**kwargs):
        calls.append(kwargs)
        return {"predictor_steps": 2, "generator_steps": 1, "predictor_loss": 0.2, "generator_loss": 0.3}

    stats = joint_script.run_joint_training_epoch(
        predictor="predictor",
        generator="generator",
        train_loader=["batch-a"],
        predictor_optimizer="p-opt",
        generator_optimizer="g-opt",
        update_strategy="alternating",
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.03,
        alternating_runner=fake_run_alternating_epoch,
        p_steps=2,
        g_steps=1,
    )

    assert stats["loss"] == 0.25
    assert stats["predictor_steps"] == 2
    assert stats["generator_steps"] == 1
    assert calls[0]["p_steps"] == 2
    assert calls[0]["g_steps"] == 1


def test_build_output_tag_prefers_stage25_profile_name():
    assert joint_script.build_output_tag(None) == "cycle"
    assert joint_script.build_output_tag("2.5A") == "stage25_2p5a"
