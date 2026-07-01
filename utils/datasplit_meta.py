"""Write datasplit_meta.json sidecar next to datasplit.json."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any


def write_datasplit_meta(
    output_json: pathlib.Path,
    *,
    dataset_id: str,
    split_source_json: str | None = None,
    counts: dict[str, int],
    skipped_total: int = 0,
    extra: dict[str, Any] | None = None,
) -> pathlib.Path:
    meta_path = output_json.parent / "datasplit_meta.json"
    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_id": dataset_id,
        "datasplit_json": str(output_json.resolve()),
        "split_source_json": str(pathlib.Path(split_source_json).resolve()) if split_source_json else None,
        "counts": counts,
        "skipped_total": skipped_total,
    }
    if extra:
        payload.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return meta_path
