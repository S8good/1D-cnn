from pathlib import Path

from scripts import plot_stage25_summary as summary_script


def test_collect_model_c_summary_reads_mean_and_best_seed(tmp_path: Path):
    outputs = tmp_path / "outputs"
    stage2 = outputs / "stage2_regressor_3seed_20260326_143033"
    stage2.mkdir(parents=True)
    (stage2 / "summary_mean_std_with_mape.csv").write_text(
        "model,mae_mean,mae_std,rmse_mean,rmse_std,mape_mean,mape_std,r2_mean,r2_std\n"
        "Model C,6.48,0.86,13.25,2.36,33.68,4.65,0.7595,0.0833\n",
        encoding="utf-8",
    )
    (stage2 / "paired_seed_metrics.csv").write_text(
        "seed,model_c_mae,model_c_rmse,model_c_r2,c_cycle_mae,c_cycle_rmse,c_cycle_r2,delta_mae,delta_rmse,delta_r2\n"
        "20260325,5.58,10.82,0.8430,0,0,0,0,0,0\n"
        "20260331,7.30,15.53,0.6763,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    result = summary_script.collect_model_c_summary(outputs)

    assert result["label"] == "Model C"
    assert result["mean"]["mae"] == 6.48
    assert result["best"]["seed"] == "20260325"
    assert result["best"]["rmse"] == 10.82


def test_collect_stage25_profile_summary_aggregates_seed_dirs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    for seed, mae, rmse, r2 in (
        ("20260325", 6.33, 13.44, 0.7578),
        ("20260331", 6.38, 13.48, 0.7561),
        ("20260407", 6.48, 13.72, 0.7478),
    ):
        run_dir = outputs / f"stage25_2p5b_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "split_test_metrics_predictor_v2_stage25_2p5b.csv").write_text(
            "seed,mae_ng_ml,rmse_ng_ml,r2\n"
            f"{seed},{mae},{rmse},{r2}\n",
            encoding="utf-8",
        )

    result = summary_script.collect_stage25_profile_summary(outputs, "2.5B")

    assert result["label"] == "Stage 2.5B"
    assert result["best"]["seed"] == "20260325"
    assert round(result["mean"]["mae"], 4) == 6.3967
    assert round(result["std"]["rmse"], 4) > 0


def test_collect_stage25_profile_summary_ignores_suffixed_search_runs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = outputs / "stage25_2p5b_seed20260325"
    run_dir.mkdir(parents=True)
    (run_dir / "split_test_metrics_predictor_v2_stage25_2p5b.csv").write_text(
        "seed,mae_ng_ml,rmse_ng_ml,r2\n20260325,6.33,13.44,0.7578\n",
        encoding="utf-8",
    )
    search_dir = outputs / "stage25_2p5b_seed20260325_search"
    search_dir.mkdir(parents=True)
    (search_dir / "split_test_metrics_predictor_v2_stage25_2p5b.csv").write_text(
        "seed,mae_ng_ml,rmse_ng_ml,r2\n20260325,9.99,99.99,0.1000\n",
        encoding="utf-8",
    )

    result = summary_script.collect_stage25_profile_summary(outputs, "2.5B")

    assert result["mean"]["mae"] == 6.33
