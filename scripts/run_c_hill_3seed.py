import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("C:/Users/Spc/Desktop/3.LSPR-code/LSPR_code/DeepLearning/LSPR_Spectra_Master")
DEFAULT_SEEDS = [20260325, 20260331, 20260407]


def main():
    parser = argparse.ArgumentParser(description="Run 3-seed C+Hill experiments.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--joint-epochs", type=int, default=60)
    parser.add_argument("--pretrain-gen-epochs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    for seed in DEFAULT_SEEDS:
        command = [
            sys.executable,
            "scripts/run_stage3_experiment.py",
            "--profile",
            "CH-fixed-regressor",
            "--seed",
            str(seed),
            "--joint-epochs",
            str(args.joint_epochs),
            "--pretrain-gen-epochs",
            str(args.pretrain_gen_epochs),
            "--source-root",
            str(source_root),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=source_root, check=True)


if __name__ == "__main__":
    main()
