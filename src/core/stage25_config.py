from dataclasses import dataclass


@dataclass(frozen=True)
class Stage25Profile:
    name: str
    update_strategy: str
    predictor_train_mode: str
    predictor_lr: float
    generator_lr: float
    p_steps: int
    g_steps: int
    w_cycle: float
    w_mono: float
    w_recon: float


@dataclass(frozen=True)
class Stage3GateDecision:
    route: str | None
    can_enter_stage3: bool
    narrative: str
    reason: str


_PROFILE_MAP = {
    "2.5A": Stage25Profile("2.5A", "alternating", "tail", 2e-4, 5e-4, 2, 1, 0.01, 0.05, 0.05),
    "2.5B": Stage25Profile("2.5B", "alternating", "regressor", 1e-4, 2e-4, 1, 1, 0.005, 0.05, 0.03),
    "2.5C": Stage25Profile("2.5C", "alternating", "frozen", 0.0, 1e-4, 0, 1, 0.003, 0.05, 0.02),
}


def build_stage25_profile(name: str) -> Stage25Profile:
    key = name.strip().upper()
    if key not in _PROFILE_MAP:
        raise ValueError(f"Unsupported Stage 2.5 profile: {name}")
    return _PROFILE_MAP[key]


def apply_profile_overrides(args, profile: Stage25Profile):
    args.stage25_profile = profile.name
    args.update_strategy = profile.update_strategy
    args.predictor_train_mode = profile.predictor_train_mode
    args.predictor_lr = profile.predictor_lr
    args.generator_lr = profile.generator_lr
    args.p_steps = profile.p_steps
    args.g_steps = profile.g_steps
    args.w_cycle = profile.w_cycle
    args.w_mono = profile.w_mono
    args.w_recon = profile.w_recon
    return args


def _pct_worse(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline


def evaluate_stage3_gate(
    baseline: dict,
    candidate: dict,
    paired_mae_wins: int,
    seed_count: int,
    std_ratio: float,
    training_success_rate: float,
    generator_collapse: bool,
) -> Stage3GateDecision:
    if generator_collapse:
        return Stage3GateDecision(None, False, "禁止进入阶段3", "Generator 输出塌陷。")
    if training_success_rate < 1.0:
        return Stage3GateDecision(None, False, "禁止进入阶段3", "存在训练失败 seed。")

    mae_worse = _pct_worse(candidate["mae_ng_ml"], baseline["mae_ng_ml"])
    rmse_worse = _pct_worse(candidate["rmse_ng_ml"], baseline["rmse_ng_ml"])
    r2_drop = baseline["r2"] - candidate["r2"]

    if ((mae_worse <= -0.03) or (rmse_worse <= -0.03)) and r2_drop <= 0.01 and paired_mae_wins >= 2:
        return Stage3GateDecision("A", True, "可以进入阶段3", "2.5A 已取得明确总体收益。")
    if mae_worse <= 0.02 and rmse_worse <= 0.02 and r2_drop <= 0.01 and std_ratio <= 1.0:
        return Stage3GateDecision("B", True, "稳定进入阶段3", "2.5B 总体不退化且稳定性可接受。")
    if mae_worse <= 0.05 and rmse_worse <= 0.05 and r2_drop <= 0.02:
        return Stage3GateDecision("C", True, "谨慎进入阶段3", "2.5C 仅满足保守放行条件。")
    return Stage3GateDecision(None, False, "禁止进入阶段3", "阶段2.5 仍存在明显退化。")
