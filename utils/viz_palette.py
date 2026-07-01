"""Label palettes and terminal legend for segmentation visualization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# Fusion 360 Gallery — high-contrast 8-class palette (aligned with brepnet-viz)
FUSION360_COLORS = np.array(
    [
        [230, 25, 25],  # 0 ExtrudeSide
        [255, 140, 0],  # 1 ExtrudeEnd
        [50, 180, 50],  # 2 CutSide
        [30, 100, 255],  # 3 CutEnd
        [160, 32, 240],  # 4 Fillet
        [255, 220, 0],  # 5 Chamfer
        [255, 0, 128],  # 6 RevolveSide
        [139, 69, 19],  # 7 RevolveEnd
    ],
    dtype=np.uint8,
)

FUSION360_NAMES = [
    "ExtrudeSide",
    "ExtrudeEnd",
    "CutSide",
    "CutEnd",
    "Fillet",
    "Chamfer",
    "RevolveSide",
    "RevolveEnd",
]

FUSION360_NAMES_ZH = [
    "拉伸侧面",
    "拉伸端面",
    "切除侧面",
    "切除端面",
    "倒圆角",
    "倒角",
    "旋转侧面",
    "旋转端面",
]

MECCAD_NAMES = [
    "bearing",
    "bolt",
    "bracket",
    "coupling",
    "flange",
    "gear",
    "nut",
    "pulley",
    "screw",
    "shaft",
]


def _distinct_colors(n: int) -> np.ndarray:
    colors = []
    for i in range(n):
        hue = (i * 0.61803398875) % 1.0
        sat = 0.72 + 0.2 * ((i % 3) / 2.0)
        val = 0.82 + 0.12 * ((i % 2))
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.asarray(colors, dtype=np.uint8)


MECCAD_COLORS = _distinct_colors(max(25, len(MECCAD_NAMES)))


def dataset_viz_name(dataset_id: str) -> str:
    return {"360": "fusion360", "mechcad": "mechcad"}.get(dataset_id, dataset_id)


def get_palette(dataset_id: str, num_classes: int) -> tuple[np.ndarray, list[str]]:
    if dataset_id == "360":
        colors = FUSION360_COLORS
        names = list(FUSION360_NAMES)
    elif dataset_id == "mechcad":
        colors = MECCAD_COLORS[:num_classes]
        names = list(MECCAD_NAMES[:num_classes])
        while len(names) < num_classes:
            names.append(f"class_{len(names)}")
    else:
        colors = _distinct_colors(num_classes)
        names = [f"class_{i}" for i in range(num_classes)]
    if len(colors) < num_classes:
        colors = _distinct_colors(num_classes)
    return colors, names


def class_color(class_id: int, palette: np.ndarray) -> np.ndarray:
    return palette[class_id % len(palette)]


def _terminal_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _luminance(r: int, g: int, b: int) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def _ansi_rgb_bg_swatch(r: int, g: int, b: int, width: int = 4) -> str:
    fg = "0;0;0" if _luminance(r, g, b) > 150 else "255;255;255"
    block = "█" * width
    return f"\033[48;2;{r};{g};{b}m\033[38;2;{fg}m{block}\033[0m"


def _ansi_rgb_fg(r: int, g: int, b: int, text: str) -> str:
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def print_color_legend(
    dataset_id: str,
    num_classes: int,
    *,
    use_color: bool | None = None,
) -> None:
    palette, names = get_palette(dataset_id, num_classes)
    color_on = _terminal_supports_color() if use_color is None else use_color
    dataset_name = dataset_viz_name(dataset_id)
    zh_names = (
        FUSION360_NAMES_ZH
        if dataset_id == "360" and num_classes <= len(FUSION360_NAMES_ZH)
        else names
    )

    print(f"\n[{dataset_name}] 标签颜色对照（左=GT，右=Pred，同一套颜色）：")
    print("-" * 68)
    for class_id in range(min(num_classes, len(names))):
        r, g, b = (int(palette[class_id][0]), int(palette[class_id][1]), int(palette[class_id][2]))
        en = names[class_id]
        zh = zh_names[class_id] if class_id < len(zh_names) else en
        if color_on:
            swatch = _ansi_rgb_bg_swatch(r, g, b)
            label = _ansi_rgb_fg(r, g, b, zh)
            line = f"  [{class_id:2d}] {swatch}  {label} ({en})  RGB({r:3d},{g:3d},{b:3d})"
        else:
            line = f"  [{class_id:2d}] {zh} ({en})  RGB({r:3d},{g:3d},{b:3d})"
        print(line)
    print("-" * 68)
    print("输出布局：左侧 Ground Truth | 右侧 Prediction")
    print("PLY：三角面色；STEP：XCAF 面色（FreeCAD 等查看器可显示）\n")


def extract_stem_hash(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return stem[:8]


def format_viz_basename(
    dataset_id: str,
    sample_index: int,
    stem: str,
    acc: float,
    iou: float,
) -> str:
    dataset = dataset_viz_name(dataset_id)
    stem_hash = extract_stem_hash(stem)
    return f"{dataset}_{sample_index:04d}_{stem_hash}_acc{acc * 100:.1f}_iou{iou * 100:.1f}"
