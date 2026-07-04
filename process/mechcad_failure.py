"""Structured failure records and taxonomy for mechcad triangle extraction."""

from __future__ import annotations

import fcntl
import json
import traceback
from pathlib import Path
from typing import Any


STAGE_LABELS = {
    "load_step": "occwl.Compound.load_from_step",
    "face_adjacency": "occwl.face_adjacency",
    "face_triangulation": "convertFaceToTriangles",
    "save": "torch.save",
}


def append_failure_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)


def load_skip_log(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step_path, reason = line.split("\t", 1)
        entries[str(Path(step_path).resolve())] = reason.strip()
    return entries


def load_timeout_paths(skip_log: Path) -> set[str]:
    out: set[str] = set()
    for path, reason in load_skip_log(skip_log).items():
        if reason.startswith("timeout"):
            out.add(path)
    return out


def classify_exception(stage: str, exc: BaseException) -> tuple[str, str, str | None]:
    """Return (bucket, detail, occ_layer)."""
    exc_type = type(exc).__name__
    msg = str(exc)
    lower = msg.lower()
    occ_layer = None

    if "GeomConvert" in msg or "Geom_Bezier" in msg or "Geom_BSpline" in msg:
        occ_layer = "OCC.GeomConvert"
    elif "Geom_RectangularTrimmedSurface" in msg:
        occ_layer = "OCC.Geom_RectangularTrimmedSurface"
    elif "face_adjacency" in stage or "manifold" in lower:
        occ_layer = "occwl.face_adjacency"
    elif stage == "load_step":
        occ_layer = "occwl.Compound.load_from_step"
    elif stage == "face_triangulation":
        occ_layer = "convertFaceToTriangles"

    if stage == "load_step":
        return "sample_defect", "step_read_error", occ_layer

    if "manifold" in lower:
        return "sample_defect", "non_manifold_brep", occ_layer
    if "doesn't belong to face" in lower or "edge doesn't belong" in lower:
        return "sample_defect", "brep_topology_inconsistent", occ_layer
    if "bad nurbs" in lower:
        return "sample_defect", "degenerate_nurbs_knots", occ_layer
    if "no patches" in lower:
        return "sample_defect", "zero_bezier_patches", occ_layer
    if "infinite surface" in lower:
        return "sample_defect", "infinite_surface", occ_layer
    if "weights values too small" in lower:
        return "sample_defect", "bspline_weights_degenerate", occ_layer
    if "standard_constructionerror" in lower or "geomconvert" in lower:
        return "sample_defect", "occ_surface_conversion_failed", occ_layer
    if "v1==v2" in lower or "trimmedsurface" in lower:
        return "sample_defect", "degenerate_trimmed_surface", occ_layer
    if "standard_rangeerror" in lower or "standard_domainerror" in lower:
        return "sample_defect", "occ_parameter_domain_error", occ_layer

    if exc_type == "TypeError" and "nonetype" in lower and "iterable" in lower:
        return "script_gap", "null_triangulation_iterable", occ_layer
    if "need at least one array to stack" in lower:
        return "script_gap", "empty_triangle_stack", occ_layer
    if exc_type == "ValueError" and stage == "face_triangulation":
        return "script_gap", "face_triangulation_value_error", occ_layer

    if exc_type == "ValueError":
        return "script_gap", "value_error_other", occ_layer
    if exc_type == "TypeError":
        return "script_gap", "type_error_other", occ_layer
    if exc_type == "AssertionError":
        return "sample_defect", "assertion_transfer_failed", occ_layer

    return "unknown", f"{exc_type}", occ_layer


def make_failure_record(
    *,
    step_path: Path,
    stage: str,
    exc: BaseException,
    face_index: int | None = None,
    solid_index: int = 0,
) -> dict[str, Any]:
    bucket, detail, occ_layer = classify_exception(stage, exc)
    return {
        "step_path": str(step_path.resolve()),
        "category": step_path.parent.name,
        "stem": step_path.stem,
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "solid_index": solid_index,
        "face_index": face_index,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "bucket": bucket,
        "detail": detail,
        "occ_layer": occ_layer,
        "traceback": traceback.format_exc(),
    }
