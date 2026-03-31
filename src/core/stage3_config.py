from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3Profile:
    name: str
    base_stage25_profile: str
    predictor_train_mode: str
    predictor_lr: float
    generator_lr: float
    p_steps: int
    g_steps: int
    w_cycle: float
    w_mono: float
    w_recon: float
    w_hill: float
    hill_mode: str
    hill_reg_weight: float


_PROFILE_MAP = {
    "3A-FIXED-FROZEN": Stage3Profile("3A-fixed-frozen", "2.5C", "frozen", 0.0, 1e-4, 0, 1, 0.003, 0.05, 0.02, 0.05, "fixed", 0.0),
    "3B-FIXED-REGRESSOR": Stage3Profile(
        "3B-fixed-regressor", "2.5C", "regressor", 1e-4, 1e-4, 1, 1, 0.003, 0.05, 0.02, 0.1, "fixed", 0.0
    ),
    "3C-LEARNABLE-REGRESSOR": Stage3Profile(
        "3C-learnable-regressor", "2.5C", "regressor", 1e-4, 1e-4, 1, 1, 0.003, 0.05, 0.02, 0.1, "learnable_kn", 1e-3
    ),
}


def build_stage3_profile(name: str) -> Stage3Profile:
    key = name.strip().upper()
    if key not in _PROFILE_MAP:
        raise ValueError(f"Unsupported Stage 3 profile: {name}")
    return _PROFILE_MAP[key]


def apply_stage3_profile_overrides(args, profile: Stage3Profile):
    args.stage3_profile = profile.name
    args.predictor_train_mode = profile.predictor_train_mode
    args.predictor_lr = profile.predictor_lr
    args.generator_lr = profile.generator_lr
    args.p_steps = profile.p_steps
    args.g_steps = profile.g_steps
    args.w_cycle = profile.w_cycle
    args.w_mono = profile.w_mono
    args.w_recon = profile.w_recon
    args.w_hill = profile.w_hill
    args.hill_mode = profile.hill_mode
    args.hill_reg_weight = profile.hill_reg_weight
    return args
