#!/usr/bin/env python3
"""Visualize face segmentation: GT vs Pred side-by-side in one PLY or colored STEP."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torchmetrics.functional as tmf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import datasets.brt_dataset as brt_dataset
from models.brt_segmentation import SegmentationPL
from utils.viz_palette import (
    class_color,
    format_viz_basename,
    get_palette,
    print_color_legend,
)

try:
    from occwl.compound import Compound
    from occwl.uvgrid import uvgrid
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"occwl is required for visualization: {exc}") from exc


def compute_sample_metrics(
    pred: np.ndarray, gt: np.ndarray, num_classes: int
) -> tuple[float, float]:
    n = min(len(pred), len(gt))
    if n == 0:
        return 0.0, 0.0
    pred_t = torch.from_numpy(pred[:n].astype(np.int64))
    gt_t = torch.from_numpy(gt[:n].astype(np.int64))
    acc = float(
        tmf.accuracy(pred_t, gt_t, task="multiclass", num_classes=num_classes)
    )
    iou = float(
        tmf.jaccard_index(
            pred_t,
            gt_t,
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
    )
    return acc, iou


def find_step_file(stem: str, dataset_id: str, step_root: pathlib.Path | None) -> pathlib.Path | None:
    candidates: list[pathlib.Path] = []
    if step_root is not None:
        candidates.extend(step_root.glob(f"**/{stem}.stp"))
        candidates.extend(step_root.glob(f"**/{stem}.step"))
    if dataset_id == "360":
        base = pathlib.Path("/data/hdd/datasets/s2.0.0/breps/step")
        candidates.extend([base / f"{stem}.stp", base / f"{stem}.step"])
    elif dataset_id == "mechcad":
        base = pathlib.Path("/data/hdd/datasets/mechcad/mechcad")
        candidates.extend(base.glob(f"**/{stem}.stp"))
        candidates.extend(base.glob(f"**/{stem}.step"))
    for path in candidates:
        if path.exists():
            return path
    return None


def load_labels(label_path: pathlib.Path) -> np.ndarray:
    if label_path.suffix == ".seg":
        labels = np.loadtxt(label_path, dtype=np.int64)
        if labels.ndim == 0:
            labels = np.array([labels], dtype=np.int64)
        return labels
    return np.loadtxt(label_path, dtype=np.int64)


def tessellate_face(face, nu: int = 24, nv: int = 24) -> tuple[np.ndarray, np.ndarray]:
    points = uvgrid(face, nu, nv, method="point")
    mask = uvgrid(face, nu, nv, method="inside")
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0.5
    vertices = points.reshape(-1, 3)
    flags = mask.reshape(-1)
    active_idx = np.where(flags)[0]
    if active_idx.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    index_map = -np.ones(flags.shape[0], dtype=np.int32)
    index_map[active_idx] = np.arange(active_idx.size, dtype=np.int32)

    triangles: list[list[int]] = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            ids = [
                index_map[i * nv + j],
                index_map[i * nv + j + 1],
                index_map[(i + 1) * nv + j],
                index_map[(i + 1) * nv + j + 1],
            ]
            if any(idx < 0 for idx in ids):
                continue
            triangles.append([ids[0], ids[1], ids[2]])
            triangles.append([ids[1], ids[3], ids[2]])

    if not triangles:
        return vertices[active_idx], np.empty((0, 3), dtype=np.int32)
    return vertices[active_idx], np.asarray(triangles, dtype=np.int32)


def tessellate_solid(step_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    compound = Compound.load_from_step(step_path)
    solid = next(compound.solids())

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_face_idx: list[int] = []
    offset = 0

    for face_idx, face in enumerate(solid.faces()):
        verts, tris = tessellate_face(face)
        if verts.shape[0] == 0 or tris.shape[0] == 0:
            continue
        all_vertices.append(verts)
        all_faces.append(tris + offset)
        all_face_idx.extend([face_idx] * tris.shape[0])
        offset += verts.shape[0]

    if not all_vertices:
        raise RuntimeError(f"No mesh geometry generated for {step_path}")

    return (
        np.concatenate(all_vertices, axis=0).astype(np.float32),
        np.concatenate(all_faces, axis=0).astype(np.int32),
        np.asarray(all_face_idx, dtype=np.int64),
    )


def face_labels_to_triangle_colors(
    face_labels: np.ndarray, face_indices: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    tri_labels = face_labels[np.clip(face_indices, 0, len(face_labels) - 1)]
    return palette[tri_labels % len(palette)]


def export_ply_with_face_colors(
    path: pathlib.Path,
    verts: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for fi, fc in zip(faces, face_colors):
            f.write(f"3 {fi[0]} {fi[1]} {fi[2]} {fc[0]} {fc[1]} {fc[2]}\n")


def export_comparison_ply(
    output_path: pathlib.Path,
    verts: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    palette: np.ndarray,
    gap: float = 0.3,
) -> None:
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    offset = (x_max - x_min) + gap

    verts_gt = verts.copy()
    verts_pred = verts.copy()
    verts_pred[:, 0] += offset

    faces_gt = faces.copy()
    faces_pred = faces.copy() + len(verts)

    colors_gt = face_labels_to_triangle_colors(gt_labels, face_indices, palette)
    colors_pred = face_labels_to_triangle_colors(pred_labels, face_indices, palette)

    verts_all = np.vstack([verts_gt, verts_pred])
    faces_all = np.vstack([faces_gt, faces_pred])
    colors_all = np.vstack([colors_gt, colors_pred])
    export_ply_with_face_colors(output_path, verts_all, faces_all, colors_all)


def _copy_shape(shape):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf

    trsf = gp_Trsf()
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _translate_shape(shape, dx: float):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(dx, 0.0, 0.0))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _shape_x_extent(shape) -> tuple[float, float]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, _, _, xmax, _, _ = bbox.Get()
    return float(xmin), float(xmax)


def _color_shape_faces(
    color_tool,
    shape,
    labels: np.ndarray,
    palette: np.ndarray,
) -> None:
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    while exp.More():
        if face_idx >= len(labels):
            break
        face = topods.Face(exp.Current())
        rgb = class_color(int(labels[face_idx]), palette).astype(float) / 255.0
        color = Quantity_Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), Quantity_TOC_RGB)
        color_tool.SetColor(face, color, XCAFDoc_ColorSurf)
        face_idx += 1
        exp.Next()


def write_colored_comparison_step(
    step_in: pathlib.Path,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    step_out: pathlib.Path,
    palette: np.ndarray,
    gap: float = 0.3,
) -> None:
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.STEPControl import STEPControl_AsIs
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
    from OCC.Extend.DataExchange import read_step_file

    shape = read_step_file(str(step_in))
    xmin, xmax = _shape_x_extent(shape)
    offset = (xmax - xmin) + gap

    shape_gt = _copy_shape(shape)
    shape_pred = _translate_shape(_copy_shape(shape), offset)

    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    label_gt = shape_tool.AddShape(shape_gt)
    _color_shape_faces(color_tool, shape_tool.GetShape(label_gt), gt_labels, palette)

    label_pred = shape_tool.AddShape(shape_pred)
    _color_shape_faces(color_tool, shape_tool.GetShape(label_pred), pred_labels, palette)

    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    step_out.parent.mkdir(parents=True, exist_ok=True)
    if writer.Write(str(step_out)) != 1:
        raise RuntimeError(f"Failed to write colored STEP: {step_out}")


def move_batch_to_device(batch: dict, device: torch.device, cpu_keys: set[str]) -> dict:
    inputs = {}
    for key, value in batch.items():
        if key in ("filename",):
            continue
        if key in cpu_keys:
            inputs[key] = value.cpu() if torch.is_tensor(value) else value
        else:
            inputs[key] = value.to(device) if torch.is_tensor(value) else value
    return inputs


def predict_sample(model: SegmentationPL, sample: dict, device: torch.device) -> np.ndarray:
    if hasattr(model, "model") and hasattr(model.model, "model"):
        brt_seg = model.model
        topo_layer = brt_seg.model.topo_layer
    else:
        brt_seg = model
        topo_layer = brt_seg.model.topo_layer

    cpu_keys = set(topo_layer.cpu_input())
    batch = brt_dataset.BRTDataset_seg_online._collate(
        brt_dataset.BRTDataset_seg_online, [sample]
    )
    inputs = move_batch_to_device(batch, device, cpu_keys)
    with torch.no_grad():
        logits = brt_seg(inputs)
    return logits.argmax(dim=-1).cpu().numpy()


def load_sample_item(dataset_dir: pathlib.Path, item: dict, max_facet_len: int = 100) -> dict:
    loader = brt_dataset.BRTDataset_seg_online.__new__(brt_dataset.BRTDataset_seg_online)
    loader.masking_rate = None
    loader.load_label_from_file = True
    sample = loader.load_one_sample(item)
    if sample is None:
        raise SystemExit(f"Failed to load sample: {item['face']}")
    sample = loader.normalize(sample)
    sample = loader.padding(sample, padding_mode="circular", max_facet_len=max_facet_len)
    return loader.convert_to_float32(sample)


def list_split_items(dataset_dir: pathlib.Path, split: str) -> list[dict]:
    with open(dataset_dir / "datasplit.json", encoding="utf-8") as f:
        return json.load(f)[split]


def resolve_sample(
    dataset_dir: pathlib.Path, split: str, index: int | None, stem: str | None
) -> tuple[dict, str, int]:
    items = list_split_items(dataset_dir, split)
    sample_index: int
    if stem is not None:
        sample_index = next(
            i for i, entry in enumerate(items) if pathlib.Path(entry["face"]).stem == stem
        )
        item = items[sample_index]
    else:
        if index is None:
            raise SystemExit("Either --index or --stem is required")
        if index < 0 or index >= len(items):
            raise SystemExit(f"Index out of range: {index} (0..{len(items) - 1})")
        sample_index = index
        item = items[index]
    stem = pathlib.Path(item["face"]).stem
    return load_sample_item(dataset_dir, item), stem, sample_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--dataset_id", type=str, default="360", choices=("360", "mechcad"))
    parser.add_argument("--step_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--format", type=str, default="ply", choices=("ply", "stp"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gap", type=float, default=0.3, help="X gap between GT and Pred")
    parser.add_argument(
        "--print_legend",
        action="store_true",
        help="Print label color legend and exit (no checkpoint needed)",
    )
    parser.add_argument("--num_classes", type=int, default=None)
    args = parser.parse_args()

    num_classes = args.num_classes
    if num_classes is None and args.dataset_id == "360":
        num_classes = 8
    elif num_classes is None and args.dataset_id == "mechcad":
        num_classes = 25

    if args.print_legend:
        if num_classes is None:
            raise SystemExit("--num_classes is required with --print_legend")
        print_color_legend(args.dataset_id, num_classes)
        return

    if args.checkpoint is None or args.dataset_dir is None or args.output_dir is None:
        raise SystemExit("--checkpoint, --dataset_dir and --output_dir are required")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = SegmentationPL.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    num_classes = int(model.hparams.num_classes)

    print_color_legend(args.dataset_id, num_classes)

    sample, stem, sample_index = resolve_sample(
        pathlib.Path(args.dataset_dir), args.split, args.index, args.stem
    )
    pred = predict_sample(model, sample, device)

    datasplit = json.load(open(pathlib.Path(args.dataset_dir) / "datasplit.json", encoding="utf-8"))
    item = next(
        entry
        for entry in datasplit[args.split]
        if pathlib.Path(entry["face"]).stem == stem
    )
    gt = load_labels(pathlib.Path(item["label"]))
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    sample_acc, sample_iou = compute_sample_metrics(pred, gt, num_classes)
    palette, class_names = get_palette(args.dataset_id, num_classes)
    basename = format_viz_basename(args.dataset_id, sample_index, stem, sample_acc, sample_iou)

    step_root = pathlib.Path(args.step_root) if args.step_root else None
    step_path = find_step_file(stem, args.dataset_id, step_root)
    if step_path is None:
        raise SystemExit(f"STEP file not found for stem: {stem}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{basename}.{args.format}"

    if args.format == "ply":
        verts, faces, face_indices = tessellate_solid(step_path)
        if face_indices.max() >= len(gt):
            raise SystemExit(
                f"Tessellation references face {face_indices.max()} but only {len(gt)} labels exist"
            )
        export_comparison_ply(
            output_file,
            verts,
            faces,
            face_indices,
            gt,
            pred,
            palette,
            gap=args.gap,
        )
    else:
        write_colored_comparison_step(
            step_path, gt, pred, output_file, palette, gap=args.gap
        )

    summary = {
        "stem": stem,
        "sample_index": sample_index,
        "step": str(step_path),
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "output_dir": str(out_dir.resolve()),
        "output_file": str(output_file.resolve()),
        "format": args.format,
        "layout": "gt_left_pred_right",
        "sample_acc": sample_acc,
        "sample_iou": sample_iou,
        "basename": basename,
        "class_names": class_names,
        "pred_labels": pred.tolist(),
        "gt_labels": gt.tolist(),
    }
    summary_path = out_dir / f"{basename}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
