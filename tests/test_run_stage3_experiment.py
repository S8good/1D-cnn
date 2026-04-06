from pathlib import Path

from scripts import run_stage3_experiment as run_script


def test_build_run_name_uses_stage3_profile_and_seed():
    assert run_script.build_run_name("3A-fixed-frozen", 20260325) == "stage3_3a_fixed_frozen_seed20260325"


def test_build_train_command_includes_stage3_profile(tmp_path: Path):
    paths = {
        "train": tmp_path / "train.xlsx",
        "val": tmp_path / "val.xlsx",
        "test": tmp_path / "test.xlsx",
        "features_xlsx": tmp_path / "features.xlsx",
        "fusion_weights": tmp_path / "fusion.pth",
        "fusion_norm": tmp_path / "norm.pth",
        "generator_weights": tmp_path / "generator.pth",
        "hill_params": tmp_path / "hill.pth",
    }
    command = run_script.build_train_command(
        paths=paths,
        profile="3A-fixed-frozen",
        seed=20260325,
        joint_epochs=60,
        pretrain_gen_epochs=5,
    )
    assert "--stage3-profile" in command
    assert "3A-fixed-frozen" in command
    assert "--hill-params-path" in command


def test_build_train_command_for_3c_uses_3a_generator_init(tmp_path: Path):
    paths = {
        "train": tmp_path / "train.xlsx",
        "val": tmp_path / "val.xlsx",
        "test": tmp_path / "test.xlsx",
        "features_xlsx": tmp_path / "features.xlsx",
        "fusion_weights": tmp_path / "fusion.pth",
        "fusion_norm": tmp_path / "norm.pth",
        "generator_weights": tmp_path / "generator.pth",
        "hill_params": tmp_path / "hill.pth",
        "stage3_3a_generator": tmp_path / "stage3_3a_generator.pth",
    }
    command = run_script.build_train_command(
        paths=paths,
        profile="3C-learnable-regressor",
        seed=20260325,
        joint_epochs=60,
        pretrain_gen_epochs=5,
    )
    assert "--generator-init-weights" in command
    assert str(paths["stage3_3a_generator"]) in command
    assert "--generator-warmup-epochs" in command
    assert "10" in command


def test_build_train_command_for_ch_uses_3a_generator_init(tmp_path: Path):
    paths = {
        "train": tmp_path / "train.xlsx",
        "val": tmp_path / "val.xlsx",
        "test": tmp_path / "test.xlsx",
        "features_xlsx": tmp_path / "features.xlsx",
        "fusion_weights": tmp_path / "fusion.pth",
        "fusion_norm": tmp_path / "norm.pth",
        "generator_weights": tmp_path / "generator.pth",
        "hill_params": tmp_path / "hill.pth",
        "stage3_3a_generator": tmp_path / "stage3_3a_generator.pth",
    }
    command = run_script.build_train_command(
        paths=paths,
        profile="CH-fixed-regressor",
        seed=20260325,
        joint_epochs=60,
        pretrain_gen_epochs=5,
    )
    assert "--generator-init-weights" in command
    assert str(paths["stage3_3a_generator"]) in command
