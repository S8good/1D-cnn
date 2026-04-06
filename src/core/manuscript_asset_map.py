from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSpec:
    source_name: str
    target_name: str
    label: str


MAIN_FIGURE_SPECS = [
    AssetSpec("正文版.png", "Figure1_Framework_Main.png", "Figure 1"),
    AssetSpec("ablation_comparison_figure.png", "Figure2_Ablation_A_B_C_D.png", "Figure 2"),
    AssetSpec("true_vs_pred_3c_figure.png", "Figure3_True_vs_Pred_ModelD.png", "Figure 3"),
    AssetSpec("hill_consistency_figure.png", "Figure4_Hill_Consistency.png", "Figure 4"),
    AssetSpec("stage3_comparison_figure.png", "Figure5A_Stage3_Comparison.png", "Figure 5A"),
    AssetSpec("stage3_hilmae_figure.png", "Figure5B_Hill_MAE_Zoom.png", "Figure 5B"),
]

SUPPLEMENTARY_FIGURE_SPECS = [
    AssetSpec("补充版.png", "FigureS1_Framework_Detailed.png", "Figure S1"),
    AssetSpec("bland_altman_3c_figure.png", "FigureS2_Bland_Altman.png", "Figure S2"),
    AssetSpec("segment_stats_figure.png", "FigureS3_Segmented_Error.png", "Figure S3"),
    AssetSpec("mvr_comparison_figure.png", "FigureS4_MVR_Comparison.png", "Figure S4"),
    AssetSpec("c_hill_comparison_figure.png", "FigureS5_Extended_Ablation_C_Hill_D.png", "Figure S5"),
]

BACKUP_FIGURE_SPECS = [
    AssetSpec("backup_stage3_seed_detail_figure.png", "Backup_Stage3_Seed_Detail.png", "Backup"),
]

TABLE_RENAME_MAP = {
    "ablation_summary.csv": ("main_figures", "Table1_Ablation_Summary.csv"),
    "mvr_summary.csv": ("supplementary_figures", "TableS1_MVR_Summary.csv"),
    "c_hill_comparison_summary.csv": ("supplementary_figures", "TableS2_C_Hill_Comparison_Summary.csv"),
    "c_hill_3seed_summary.csv": ("supplementary_figures", "TableS3_C_Hill_3Seed_Summary.csv"),
    "segment_stats_table.csv": ("supplementary_figures", "TableS4_Segmented_MAE_Summary.csv"),
    "stage3_paper_table.csv": ("supplementary_figures", "TableS5_Stage3_Configuration_Summary.csv"),
}


def all_target_names() -> list[str]:
    names = [spec.target_name for spec in MAIN_FIGURE_SPECS + SUPPLEMENTARY_FIGURE_SPECS + BACKUP_FIGURE_SPECS]
    names.extend(target for _folder, target in TABLE_RENAME_MAP.values())
    return names
