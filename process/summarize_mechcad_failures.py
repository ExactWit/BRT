#!/usr/bin/env python3
"""Summarize structured mechcad triangle failure JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def dedupe_by_step(records: list[dict]) -> list[dict]:
    """Keep first failure per STEP (earliest stage in pipeline)."""
    stage_order = {"load_step": 0, "face_adjacency": 1, "face_triangulation": 2, "save": 3}
    best: dict[str, dict] = {}
    for rec in records:
        key = rec["step_path"]
        prev = best.get(key)
        if prev is None or stage_order.get(rec["stage"], 99) < stage_order.get(prev["stage"], 99):
            best[key] = rec
    return list(best.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/processed/triangles_failure.jsonl"),
    )
    parser.add_argument(
        "--skip-log",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/processed/triangles_skip.log"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/processed/triangles_failure_report.json"),
    )
    args = parser.parse_args()

    records = dedupe_by_step(load_records(args.failure_log))
    timeout_count = 0
    if args.skip_log.exists():
        for line in args.skip_log.read_text(encoding="utf-8").splitlines():
            if "\ttimeout" in line:
                timeout_count += 1

    by_bucket = Counter(r["bucket"] for r in records)
    by_detail = Counter(r["detail"] for r in records)
    by_stage = Counter(r["stage"] for r in records)
    by_occ = Counter(r.get("occ_layer") or "unknown" for r in records)
    by_cat = Counter(r["category"] for r in records)
    by_cat_bucket: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_cat_bucket[r["category"]][r["bucket"]] += 1

    examples: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r["detail"]
        if len(examples[key]) < 3:
            examples[key].append(
                {
                    "category": r["category"],
                    "stem": r["stem"],
                    "stage": r["stage"],
                    "face_index": r.get("face_index"),
                    "exception_type": r["exception_type"],
                    "exception_message": r["exception_message"][:200],
                }
            )

    report = {
        "failure_records_raw": len(load_records(args.failure_log)),
        "failure_samples_unique": len(records),
        "excluded_timeout_from_skip_log": timeout_count,
        "by_bucket": dict(by_bucket),
        "by_detail": dict(by_detail.most_common()),
        "by_stage": dict(by_stage),
        "by_occ_layer": dict(by_occ.most_common()),
        "by_category": dict(by_cat),
        "by_category_bucket": {cat: dict(cnt) for cat, cnt in sorted(by_cat_bucket.items())},
        "examples_by_detail": dict(examples),
        "interpretation": {
            "sample_defect": "CAD/STEP/BRep/OCC 几何或拓扑缺陷，非脚本逻辑 bug",
            "script_gap": "当前 convertFaceToTriangles/build_triangles 未覆盖的边界情况，可改脚本缓解",
            "unknown": "需人工看 traceback",
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== MechCAD triangle 失败分析 ===")
    print(f"唯一失败样本: {len(records)}")
    print(f"skip_log 超时(已排除未重试): {timeout_count}")
    print("\n按 bucket:")
    for k, v in by_bucket.most_common():
        print(f"  {k}: {v}")
    print("\n按 detail (Top 12):")
    for k, v in by_detail.most_common(12):
        print(f"  {k}: {v}")
    print("\n按 occwl/OCC 层:")
    for k, v in by_occ.most_common(8):
        print(f"  {k}: {v}")
    print(f"\n完整报告: {args.report_out}")


if __name__ == "__main__":
    main()
