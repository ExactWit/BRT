"""Per-sample test metrics persisted next to experiment checkpoints."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from utils.checkpoint_info import checkpoint_record


PER_SAMPLE_FILENAME = "test_per_sample.json"
SCHEMA_VERSION = 1


def stem_from_face_path(face_path: str) -> str:
    return pathlib.Path(face_path).stem


def build_per_sample_payload(
    *,
    samples: list[dict[str, Any]],
    checkpoint: pathlib.Path,
    dataset_dir: pathlib.Path,
    dataset_id: str | None,
) -> dict[str, Any]:
    split_path = dataset_dir / "datasplit.json"
    stem_to_index: dict[str, int] = {}
    if split_path.exists():
        with open(split_path, encoding="utf-8") as f:
            test_items = json.load(f).get("test", [])
        for index, item in enumerate(test_items):
            stem_to_index[stem_from_face_path(item["face"])] = index

    enriched = []
    for rec in samples:
        entry = dict(rec)
        entry["index"] = stem_to_index.get(entry.get("stem", ""))
        enriched.append(entry)

    ckpt_info = checkpoint_record(checkpoint) or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_kind": checkpoint.name.replace(".ckpt", ""),
        "checkpoint_epoch": ckpt_info.get("epoch"),
        "checkpoint_epoch_1based": ckpt_info.get("epoch_1based"),
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_id": dataset_id,
        "num_samples": len(enriched),
        "samples": enriched,
    }


def write_per_sample_results(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_per_sample_results(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_per_sample_valid(
    payload: dict[str, Any] | None,
    *,
    checkpoint: pathlib.Path,
    dataset_dir: pathlib.Path,
) -> bool:
    if not payload or "samples" not in payload:
        return False
    if payload.get("checkpoint") != str(checkpoint.resolve()):
        return False
    if payload.get("dataset_dir") != str(dataset_dir.resolve()):
        return False
    return len(payload["samples"]) > 0


def list_test_samples_sorted(
    dataset_dir: pathlib.Path,
    run_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    """Return test split entries sorted by IoU ascending (missing metrics last)."""
    with open(dataset_dir / "datasplit.json", encoding="utf-8") as f:
        split = json.load(f)["test"]

    per_sample_path = run_dir / PER_SAMPLE_FILENAME
    by_stem: dict[str, dict[str, Any]] = {}
    if per_sample_path.exists():
        payload = load_per_sample_results(per_sample_path) or {}
        for rec in payload.get("samples", []):
            stem = rec.get("stem")
            if stem:
                by_stem[stem] = rec

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(split):
        stem = stem_from_face_path(item["face"])
        rec = by_stem.get(stem, {})
        rows.append(
            {
                "index": index,
                "stem": stem,
                "iou": rec.get("iou"),
                "acc": rec.get("acc"),
                "has_metrics": stem in by_stem,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple:
        iou = row.get("iou")
        if iou is None:
            return (1, 1.0, row["index"])
        return (0, float(iou), row["index"])

    rows.sort(key=sort_key)
    return rows


def list_test_samples_by_index(
    dataset_dir: pathlib.Path,
    run_dir: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    """Return test split entries in native datasplit index order."""
    with open(dataset_dir / "datasplit.json", encoding="utf-8") as f:
        split = json.load(f)["test"]

    by_stem: dict[str, dict[str, Any]] = {}
    if run_dir is not None:
        per_sample_path = run_dir / PER_SAMPLE_FILENAME
        if per_sample_path.exists():
            payload = load_per_sample_results(per_sample_path) or {}
            for rec in payload.get("samples", []):
                stem = rec.get("stem")
                if stem:
                    by_stem[stem] = rec

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(split):
        stem = stem_from_face_path(item["face"])
        rec = by_stem.get(stem, {})
        rows.append(
            {
                "index": index,
                "stem": stem,
                "iou": rec.get("iou"),
                "acc": rec.get("acc"),
                "has_metrics": stem in by_stem,
            }
        )
    return rows


def resolve_test_sample_index(dataset_dir: pathlib.Path, raw_index: str | int) -> dict[str, Any]:
    """Resolve user input (e.g. 251 or 0251) to a test split entry."""
    if isinstance(raw_index, str):
        text = raw_index.strip()
        if not text.isdigit():
            raise ValueError(f"invalid index: {raw_index}")
        index = int(text, 10)
    else:
        index = int(raw_index)

    rows = list_test_samples_by_index(dataset_dir)
    if index < 0 or index >= len(rows):
        raise IndexError(f"index {index} out of range (0..{len(rows) - 1})")
    return rows[index]
