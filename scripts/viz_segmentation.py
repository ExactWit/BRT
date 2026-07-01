#!/usr/bin/env python3
"""Visualize face segmentation predictions vs ground truth as PLY or colored STEP."""

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

try:
    from occwl.compound import Compound
    from occwl.uvgrid import uvgrid
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"occwl is required for visualization: {exc}") from exc


PALETTE = np.array(
    [
        [160, 160, 160],
        [0, 0, 0],
        [200, 0, 0],
        [255, 40, 40],
        [255, 80, 80],
        [255, 120, 120],
        [255, 160, 160],
        [255, 200, 200],
        [180, 0, 180],
        [0, 120, 255],
        [0, 180, 120],
        [255, 180, 0],
        [120, 120, 0],
        [0, 180, 180],
        [180, 120, 60],
        [60, 60, 180],
        [220, 120, 0],
        [120, 220, 0],
        [0, 120, 220],
        [220, 0, 120],
        [80, 80, 80],
        [200, 200, 0],
        [0, 200, 200],
        [200, 0, 200],
        [100, 150, 200],
    ],
    dtype=np.uint8,
)


def class_color(class_id: int) -> np.ndarray:
    return PALETTE[class_id % len(PALETTE)]


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
    return iou, acc


def metric_filename_suffix(iou: float, acc: float) -> str:
    return f"iou{iou:.4f}_acc{acc:.4f}"


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


def load_labels(label_path: pathlib.Path, num_faces: int) -> np.ndarray:
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


def build_colored_mesh(step_path: pathlib.Path, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    compound = Compound.load_from_step(step_path)
    solid = next(compound.solids())

    all_vertices: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0

    for face_idx, face in enumerate(solid.faces()):
        if face_idx >= len(labels):
            break
        verts, tris = tessellate_face(face)
        if verts.shape[0] == 0 or tris.shape[0] == 0:
            continue
        color = class_color(int(labels[face_idx]))
        all_vertices.append(verts)
        all_colors.append(np.tile(color, (verts.shape[0], 1)))
        all_faces.append(tris + offset)
        offset += verts.shape[0]

    if not all_vertices:
        raise RuntimeError(f"No mesh geometry generated for {step_path}")

    return (
        np.concatenate(all_vertices, axis=0),
        np.concatenate(all_colors, axis=0),
        np.concatenate(all_faces, axis=0),
    )


def write_ply(path: pathlib.Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def write_colored_step(
    step_in: pathlib.Path,
    labels: np.ndarray,
    step_out: pathlib.Path,
) -> None:
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
    from OCC.Extend.DataExchange import read_step_file

    shape = read_step_file(str(step_in))
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    from OCC.Extend.TopologyUtils import TopologyExplorer

    exp = TopologyExplorer(shape, ignore_orientation=True)
    for face_idx, face in enumerate(exp.faces()):
        if face_idx >= len(labels):
            break
        label = int(labels[face_idx])
        rgb = class_color(label).astype(float) / 255.0
        color = Quantity_Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), Quantity_TOC_RGB)
        color_tool.SetColor(face, color, 1)

    shape_tool.AddShape(shape)
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, 1)
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
    # checkpoint is SegmentationPL; inner nn.Module is BRTSegmentation.
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


def resolve_sample(dataset_dir: pathlib.Path, split: str, index: int | None, stem: str | None) -> tuple[dict, str]:
    items = list_split_items(dataset_dir, split)
    if stem is not None:
        item = next((entry for entry in items if pathlib.Path(entry["face"]).stem == stem), None)
        if item is None:
            raise SystemExit(f"Sample stem not found in {split} split: {stem}")
    else:
        if index is None:
            raise SystemExit("Either --index or --stem is required")
        if index < 0 or index >= len(items):
            raise SystemExit(f"Index out of range: {index} (0..{len(items) - 1})")
        item = items[index]
    stem = pathlib.Path(item["face"]).stem
    return load_sample_item(dataset_dir, item), stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, default="360", choices=("360", "mechcad"))
    parser.add_argument("--step_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--format", type=str, default="ply", choices=("ply", "stp"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = SegmentationPL.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    sample, stem = resolve_sample(pathlib.Path(args.dataset_dir), args.split, args.index, args.stem)
    pred = predict_sample(model, sample, device)

    datasplit = json.load(open(pathlib.Path(args.dataset_dir) / "datasplit.json", encoding="utf-8"))
    item = next(
        entry
        for entry in datasplit[args.split]
        if pathlib.Path(entry["face"]).stem == stem
    )
    gt = load_labels(pathlib.Path(item["label"]), num_faces=len(pred))
    num_classes = int(model.hparams.num_classes)
    sample_iou, sample_acc = compute_sample_metrics(pred, gt, num_classes)
    metric_tag = metric_filename_suffix(sample_iou, sample_acc)

    step_root = pathlib.Path(args.step_root) if args.step_root else None
    step_path = find_step_file(stem, args.dataset_id, step_root)
    if step_path is None:
        raise SystemExit(f"STEP file not found for stem: {stem}")

    out_dir = pathlib.Path(args.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "ply":
        for name, labels in (("pred", pred), ("gt", gt)):
            verts, colors, faces = build_colored_mesh(step_path, labels)
            write_ply(
                out_dir / f"{stem}_{name}_{metric_tag}.ply",
                verts,
                colors,
                faces,
            )
    else:
        write_colored_step(
            step_path, pred, out_dir / f"{stem}_pred_{metric_tag}.stp"
        )
        write_colored_step(step_path, gt, out_dir / f"{stem}_gt_{metric_tag}.stp")

    summary = {
        "stem": stem,
        "step": str(step_path),
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "output_dir": str(out_dir.resolve()),
        "format": args.format,
        "sample_iou": sample_iou,
        "sample_acc": sample_acc,
        "metric_tag": metric_tag,
        "pred_labels": pred.tolist(),
        "gt_labels": gt.tolist(),
        "outputs": {
            "pred": f"{stem}_pred_{metric_tag}.{args.format}",
            "gt": f"{stem}_gt_{metric_tag}.{args.format}",
        },
    }
    with open(out_dir / "viz_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
