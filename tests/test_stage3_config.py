import math
from argparse import Namespace

from src.core.stage3_config import apply_stage3_profile_overrides, build_stage3_profile


def test_build_stage3_profile_3a_uses_frozen_predictor_and_fixed_hill():
    profile = build_stage3_profile("3A-fixed-frozen")
    assert profile.base_stage25_profile == "2.5C"
    assert profile.predictor_train_mode == "frozen"
    assert profile.hill_mode == "fixed"
    assert math.isclose(profile.w_hill, 0.05)


def test_build_stage3_profile_3c_uses_learnable_hill_and_regressor():
    profile = build_stage3_profile("3C-learnable-regressor")
    assert profile.predictor_train_mode == "regressor"
    assert profile.hill_mode == "learnable_kn"


def test_apply_stage3_profile_overrides_sets_args():
    args = Namespace(stage3_profile=None, predictor_train_mode="all", w_hill=0.0, hill_mode="off")
    profile = build_stage3_profile("3B-fixed-regressor")
    apply_stage3_profile_overrides(args, profile)
    assert args.stage3_profile == "3B-fixed-regressor"
    assert args.predictor_train_mode == "regressor"
    assert args.hill_mode == "fixed"
