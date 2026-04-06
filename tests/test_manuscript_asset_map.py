from src.core.manuscript_asset_map import (
    BACKUP_FIGURE_SPECS,
    MAIN_FIGURE_SPECS,
    SUPPLEMENTARY_FIGURE_SPECS,
    TABLE_RENAME_MAP,
    all_target_names,
)


def test_asset_target_names_are_unique():
    names = all_target_names()
    assert len(names) == len(set(names))


def test_main_figure_count_matches_expected_numbering():
    assert len(MAIN_FIGURE_SPECS) == 6
    assert MAIN_FIGURE_SPECS[0].label == "Figure 1"
    assert MAIN_FIGURE_SPECS[-1].label == "Figure 5B"


def test_supplementary_assets_cover_figures_and_tables():
    assert len(SUPPLEMENTARY_FIGURE_SPECS) == 5
    assert len(TABLE_RENAME_MAP) >= 6
    assert BACKUP_FIGURE_SPECS[0].label == "Backup"
