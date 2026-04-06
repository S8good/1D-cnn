import math
from argparse import Namespace

from src.core.stage3_config import apply_stage3_profile_overrides, build_stage3_profile


def test_build_stage3_profile_3a_uses_frozen_predictor_and_fixed_hill():
    profile = build_stage3_profile("3A-fixed-frozen")
    assert profile.base_stage25_profile == "2.5C"
    assert profile.predictor_train_mode == "frozen"
    assert profile.hill_mode == "fixed"
    assert math.isclose(profile.w_hill, 0.05)
    assert math.isclose(profile.hill_lr, 1e-4)


def test_build_stage3_profile_3c_uses_learnable_hill_and_regressor():
    profile = build_stage3_profile("3C-learnable-regressor")
    assert profile.predictor_train_mode == "regressor"
    assert profile.hill_mode == "learnable_kn"
    assert math.isclose(profile.w_hill, 0.2)
    assert math.isclose(profile.hill_reg_weight, 1e-4)
    assert math.isclose(profile.hill_lr, 1e-4)
    assert profile.g_steps == 2
    assert math.isclose(profile.w_cycle, 0.001)
    assert math.isclose(profile.w_recon, 0.01)
    assert profile.generator_init_profile == "3A-fixed-frozen"
    assert profile.generator_warmup_epochs == 10


def test_apply_stage3_profile_overrides_sets_args():
    args = Namespace(stage3_profile=None, predictor_train_mode="all", w_hill=0.0, hill_mode="off", hill_lr=0.0)
    profile = build_stage3_profile("3B-fixed-regressor")
    apply_stage3_profile_overrides(args, profile)
    assert args.stage3_profile == "3B-fixed-regressor"
    assert args.predictor_train_mode == "regressor"
    assert args.hill_mode == "fixed"
    assert args.hill_lr == profile.hill_lr


def test_build_stage3_profile_ch_disables_cycle_but_keeps_fixed_hill():
    profile = build_stage3_profile("CH-fixed-regressor")
    assert profile.predictor_train_mode == "regressor"
    assert profile.hill_mode == "fixed"
    assert math.isclose(profile.w_cycle, 0.0)
    assert profile.w_hill > 0.0
