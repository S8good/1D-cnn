from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.manuscript_asset_map import (  # noqa: E402
    BACKUP_FIGURE_SPECS,
    MAIN_FIGURE_SPECS,
    SUPPLEMENTARY_FIGURE_SPECS,
    TABLE_RENAME_MAP,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT_DIR / "outputs" / "paper_selected_figures"
MAIN_DIR = PAPER_DIR / "main_figures"
SUPP_DIR = PAPER_DIR / "supplementary_figures"
BACKUP_DIR = PAPER_DIR / "backup"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_image(path: Path, label: str):
    if label == "Backup":
        return
    image = Image.open(path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    font_size = max(22, min(width, height) // 20)
    font = _load_font(font_size)
    padding_x = max(12, font_size // 2)
    padding_y = max(10, font_size // 3)
    bbox = draw.textbbox((0, 0), label, font=font)
    box_w = (bbox[2] - bbox[0]) + 2 * padding_x
    box_h = (bbox[3] - bbox[1]) + 2 * padding_y
    x = max(16, width // 40)
    y = max(16, height // 40)
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=12, fill=(255, 255, 255, 225), outline=(60, 60, 60, 220), width=2)
    draw.text((x + padding_x, y + padding_y - 2), label, font=font, fill=(25, 25, 25, 255))
    annotated = Image.alpha_composite(image, overlay).convert("RGB")
    annotated.save(path)


def _process_image_specs(folder: Path, specs):
    existing_targets = {spec.target_name for spec in specs}
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() == ".png" and item.name not in existing_targets and all(item.name != spec.source_name for spec in specs):
            item.unlink()
    for spec in specs:
        source = folder / spec.source_name
        target = folder / spec.target_name
        if source.exists():
            source.rename(target)
        elif not target.exists():
            raise FileNotFoundError(f"Missing expected image: {source}")
        _annotate_image(target, spec.label)


def _copy_and_rename_tables():
    for source_name, (folder_name, target_name) in TABLE_RENAME_MAP.items():
        source = OUTPUTS_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(f"Missing expected table: {source}")
        target_dir = PAPER_DIR / folder_name
        target = target_dir / target_name
        shutil.copy2(source, target)


def main():
    _process_image_specs(MAIN_DIR, MAIN_FIGURE_SPECS)
    _process_image_specs(SUPP_DIR, SUPPLEMENTARY_FIGURE_SPECS)
    _process_image_specs(BACKUP_DIR, BACKUP_FIGURE_SPECS)
    _copy_and_rename_tables()
    print("Paper-selected figures finalized.")


if __name__ == "__main__":
    main()
