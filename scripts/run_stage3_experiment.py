import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master")


def build_run_name(profile: str, seed: int, suffix: str = "") -> str:
    base = "stage3_" + profile.lower().replace("-", "_") + f"_seed{seed}"
    return base if not suffix else f"{base}_{suffix}"


def build_stage3_run_paths(source_root: Path):
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
        "hill_params": pretrained_root / "stage3_hill_params.pth",
    }


def build_train_command(paths, profile: str, seed: int, joint_epochs: int, pretrain_gen_epochs: int):
    return [
        sys.executable,
        "scripts/train_joint_physics_dl.py",
        "--stage3-profile",
        profile,
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
        "--hill-params-path",
        str(paths["hill_params"]),
    ]


def snapshot_stage3_outputs(source_root: Path, run_name: str):
    output_root = source_root / "outputs"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stage3_tag = run_name.rsplit("_seed", 1)[0]
    for candidate in output_root.glob(f"*{stage3_tag}*.csv"):
        shutil.copy2(candidate, run_dir / candidate.name)
    for candidate in (source_root / "models" / "checkpoints").glob(f"*{stage3_tag}*.pth"):
        shutil.copy2(candidate, run_dir / candidate.name)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--joint-epochs", type=int, default=60)
    parser.add_argument("--pretrain-gen-epochs", type=int, default=5)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    run_name = build_run_name(args.profile, args.seed)
    paths = build_stage3_run_paths(source_root)
    command = build_train_command(paths, args.profile, args.seed, args.joint_epochs, args.pretrain_gen_epochs)

    print(run_name)
    print(" ".join(command))
    if args.dry_run:
        return

    subprocess.run(command, cwd=source_root, check=True)
    run_dir = snapshot_stage3_outputs(source_root, run_name)
    print(f"Snapshot directory: {run_dir}")


if __name__ == "__main__":
    main()
