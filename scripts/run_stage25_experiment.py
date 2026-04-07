import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.stage25_config import build_stage25_profile


DEFAULT_SOURCE_ROOT = Path("C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master")


def build_stage25_run_paths(source_root: Path, worktree_root: Path):
    split_root = source_root / "data" / "processed" / "splits_reconstructed"
    pretrained_root = source_root / "models" / "pretrained"
    return {
        "train": split_root / "train_preprocessed_pairs.xlsx",
        "val": split_root / "val_preprocessed_pairs.xlsx",
        "test": split_root / "test_preprocessed_pairs.xlsx",
        "features_xlsx": source_root / "data" / "processed" / "Reconstructed_Preprocessed_Features_and_Delta.xlsx",
        "fusion_weights": pretrained_root / "spectral_predictor_v2_fusion.pth",
        "fusion_norm": pretrained_root / "predictor_v2_fusion_norm_params.pth",
        "generator_weights": pretrained_root / "spectral_generator_cycle.pth",
        "output_root": worktree_root / "outputs",
    }


def build_run_name(profile: str, seed: int, suffix: str = "") -> str:
    run_name = f"stage25_{profile.lower().replace('.', 'p')}_seed{seed}"
    if suffix:
        return f"{run_name}_{suffix}"
    return run_name


def build_output_tag(profile: str) -> str:
    return f"stage25_{profile.lower().replace('.', 'p')}"


def build_train_command(
    worktree_root: Path,
    paths,
    profile: str,
    seed: int,
    joint_epochs: int,
    pretrain_gen_epochs: int,
    extra_args=None,
):
    command = [
        "python",
        "scripts/train_joint_physics_dl.py",
        "--seed",
        str(seed),
        "--joint-epochs",
        str(joint_epochs),
        "--pretrain-gen-epochs",
        str(pretrain_gen_epochs),
        "--train",
        str(paths["train"]),
        "--val",
        str(paths["val"]),
        "--test",
        str(paths["test"]),
        "--features-xlsx",
        str(paths["features_xlsx"]),
        "--fusion-weights",
        str(paths["fusion_weights"]),
        "--fusion-norm",
        str(paths["fusion_norm"]),
        "--generator-weights",
        str(paths["generator_weights"]),
    ]
    if extra_args:
        profile_cfg = build_stage25_profile(profile)
        command.extend(
            [
                "--update-strategy",
                profile_cfg.update_strategy,
                "--predictor-train-mode",
                profile_cfg.predictor_train_mode,
                "--predictor-lr",
                str(profile_cfg.predictor_lr),
                "--generator-lr",
                str(profile_cfg.generator_lr),
                "--p-steps",
                str(profile_cfg.p_steps),
                "--g-steps",
                str(profile_cfg.g_steps),
                "--w-cycle",
                str(profile_cfg.w_cycle),
                "--w-mono",
                str(profile_cfg.w_mono),
                "--w-recon",
                str(profile_cfg.w_recon),
            ]
        )
    else:
        command.extend(["--stage25-profile", profile])
    if extra_args:
        command.extend(extra_args)
    return command


def snapshot_run_artifacts(worktree_root: Path, profile: str, seed: int, suffix: str = "") -> Path:
    output_tag = build_output_tag(profile)
    run_dir = worktree_root / "outputs" / build_run_name(profile, seed, suffix)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        worktree_root / "outputs" / f"split_test_metrics_predictor_v2_{output_tag}.csv",
        worktree_root / "outputs" / f"split_test_predictions_predictor_v2_{output_tag}.csv",
        worktree_root / "models" / "pretrained" / f"spectral_predictor_v2_{output_tag}.pth",
        worktree_root / "models" / "pretrained" / f"spectral_generator_{output_tag}.pth",
        worktree_root / "models" / "pretrained" / f"predictor_v2_{output_tag}_norm_params.pth",
        worktree_root / "models" / "checkpoints" / f"joint_{output_tag}_best.pth",
    ]
    for path in artifact_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected artifact missing after run: {path}")
        shutil.copy2(path, run_dir / path.name)
    return run_dir


def _validate_paths(paths) -> None:
    missing = [str(path) for key, path in paths.items() if key != "output_root" and not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage 2.5 experiment with source-repo assets.")
    parser.add_argument("--profile", required=True, choices=["2.5A", "2.5B", "2.5C"])
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--joint-epochs", type=int, default=60)
    parser.add_argument("--pretrain-gen-epochs", type=int, default=5)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--worktree-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_stage25_run_paths(args.source_root, args.worktree_root)
    _validate_paths(paths)

    run_name = build_run_name(args.profile, args.seed, args.suffix)
    command = build_train_command(
        worktree_root=args.worktree_root,
        paths=paths,
        profile=args.profile,
        seed=args.seed,
        joint_epochs=args.joint_epochs,
        pretrain_gen_epochs=args.pretrain_gen_epochs,
        extra_args=args.extra_args,
    )

    print(f"Run name: {run_name}")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(args.worktree_root) if not existing_pythonpath else str(args.worktree_root) + os.pathsep + existing_pythonpath
    completed = subprocess.run(command, cwd=args.worktree_root, env=env)
    if completed.returncode != 0:
        return int(completed.returncode)
    run_dir = snapshot_run_artifacts(args.worktree_root, args.profile, args.seed, args.suffix)
    print(f"Snapshot dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
