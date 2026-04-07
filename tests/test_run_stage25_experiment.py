from pathlib import Path

from scripts import run_stage25_experiment as run_script


def test_build_stage25_run_paths_points_to_source_repo_assets(tmp_path: Path):
    source_root = tmp_path / "source"
    worktree_root = tmp_path / "worktree"
    (source_root / "data/processed/splits_reconstructed").mkdir(parents=True)
    (source_root / "models/pretrained").mkdir(parents=True)
    (source_root / "data/processed/Reconstructed_Preprocessed_Features_and_Delta.xlsx").write_text("x")
    for name in ("train_preprocessed_pairs.xlsx", "val_preprocessed_pairs.xlsx", "test_preprocessed_pairs.xlsx"):
        (source_root / "data/processed/splits_reconstructed" / name).write_text("x")
    for name in (
        "spectral_predictor_v2_fusion.pth",
        "predictor_v2_fusion_norm_params.pth",
        "spectral_generator_cycle.pth",
    ):
        (source_root / "models/pretrained" / name).write_text("x")

    paths = run_script.build_stage25_run_paths(source_root, worktree_root)

    assert paths["train"].name == "train_preprocessed_pairs.xlsx"
    assert paths["fusion_weights"].name == "spectral_predictor_v2_fusion.pth"
    assert paths["output_root"] == worktree_root / "outputs"


def test_build_train_command_includes_stage25_profile_and_seed(tmp_path: Path):
    worktree_root = tmp_path / "worktree"
    source_root = tmp_path / "source"
    paths = {
        "train": source_root / "train.xlsx",
        "val": source_root / "val.xlsx",
        "test": source_root / "test.xlsx",
        "features_xlsx": source_root / "features.xlsx",
        "fusion_weights": source_root / "fusion.pth",
        "fusion_norm": source_root / "norm.pth",
        "generator_weights": source_root / "generator.pth",
        "output_root": worktree_root / "outputs",
    }

    command = run_script.build_train_command(
        worktree_root=worktree_root,
        paths=paths,
        profile="2.5A",
        seed=20260325,
        joint_epochs=60,
        pretrain_gen_epochs=5,
    )

    assert command[:2] == ["python", "scripts/train_joint_physics_dl.py"]
    assert "--stage25-profile" in command
    assert "2.5A" in command
    assert "--seed" in command
    assert "20260325" in command
    assert "--joint-epochs" in command
    assert "--generator-weights" in command


def test_build_run_name_uses_profile_and_seed():
    assert run_script.build_run_name("2.5A", 20260325) == "stage25_2p5a_seed20260325"


def test_build_run_name_appends_suffix_when_present():
    assert run_script.build_run_name("2.5B", 20260325, "lr1e4") == "stage25_2p5b_seed20260325_lr1e4"


def test_build_train_command_appends_override_args(tmp_path: Path):
    worktree_root = tmp_path / "worktree"
    source_root = tmp_path / "source"
    paths = {
        "train": source_root / "train.xlsx",
        "val": source_root / "val.xlsx",
        "test": source_root / "test.xlsx",
        "features_xlsx": source_root / "features.xlsx",
        "fusion_weights": source_root / "fusion.pth",
        "fusion_norm": source_root / "norm.pth",
        "generator_weights": source_root / "generator.pth",
        "output_root": worktree_root / "outputs",
    }

    command = run_script.build_train_command(
        worktree_root=worktree_root,
        paths=paths,
        profile="2.5B",
        seed=20260325,
        joint_epochs=80,
        pretrain_gen_epochs=5,
        extra_args=["--generator-lr", "1e-4", "--w-cycle", "0.004"],
    )

    assert "--stage25-profile" not in command
    assert "--update-strategy" in command
    assert "--predictor-train-mode" in command
    assert "--generator-lr" in command
    assert command[-4:] == ["--generator-lr", "1e-4", "--w-cycle", "0.004"]


def test_snapshot_run_artifacts_copies_outputs_into_seed_specific_directory(tmp_path: Path):
    worktree_root = tmp_path / "worktree"
    (worktree_root / "outputs").mkdir(parents=True)
    (worktree_root / "models" / "pretrained").mkdir(parents=True)
    (worktree_root / "models" / "checkpoints").mkdir(parents=True)

    for rel in (
        "outputs/split_test_metrics_predictor_v2_stage25_2p5a.csv",
        "outputs/split_test_predictions_predictor_v2_stage25_2p5a.csv",
        "models/pretrained/spectral_predictor_v2_stage25_2p5a.pth",
        "models/pretrained/spectral_generator_stage25_2p5a.pth",
        "models/pretrained/predictor_v2_stage25_2p5a_norm_params.pth",
        "models/checkpoints/joint_stage25_2p5a_best.pth",
    ):
        path = worktree_root / rel
        path.write_text(rel)

    run_dir = run_script.snapshot_run_artifacts(worktree_root, "2.5A", 20260325)

    assert run_dir.name == "stage25_2p5a_seed20260325"
    assert (run_dir / "split_test_metrics_predictor_v2_stage25_2p5a.csv").exists()
    assert (run_dir / "joint_stage25_2p5a_best.pth").exists()
