#!/usr/bin/env python3
"""Retry mechcad triangle extraction for samples missing .bin, excluding skip-log timeouts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "process"))

from mechcad_failure import load_timeout_paths
from solid_to_triangles2 import MECHCAD_CATEGORIES, main as solid_main


def collect_retry_candidates(
    source_root: Path,
    triangles_root: Path,
    skip_log: Path,
    categories: list[str] | None = None,
) -> tuple[list[Path], dict[str, int]]:
    timeout_paths = load_timeout_paths(skip_log)
    cats = categories or MECHCAD_CATEGORIES
    candidates: list[Path] = []
    stats = Counter()

    for cat in cats:
        cat_dir = source_root / cat
        if not cat_dir.is_dir():
            continue
        out_dir = triangles_root / cat
        for stp in sorted(cat_dir.glob("*.stp")):
            resolved = str(stp.resolve())
            if resolved in timeout_paths:
                stats["excluded_timeout"] += 1
                continue
            if (out_dir / f"{stp.stem}.bin").exists():
                stats["skipped_existing"] += 1
                continue
            candidates.append(stp)
            stats["to_retry"] += 1

    return candidates, dict(stats)


def group_by_category(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for p in paths:
        grouped.setdefault(p.parent.name, []).append(p)
    return grouped


def run_category(
    category: str,
    paths: list[Path],
    triangles_root: Path,
    source_root: Path,
    *,
    process_num: int,
    file_timeout: int,
    skip_log: Path,
    failure_log: Path,
    list_dir: Path,
) -> int:
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / f"{category}.lst"
    list_path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    input_dir = source_root / category
    output_dir = triangles_root / category
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(input_dir),
        str(output_dir),
        "--num_processes",
        str(process_num),
        "--no_random_name",
        "--method",
        "8",
        "--no_label",
        "--file_timeout",
        str(file_timeout),
        "--list",
        str(list_path),
        "--skip_log",
        str(skip_log),
        "--failure_log",
        str(failure_log),
    ]
    print(f"[retry] {category}: {len(paths)} files (timeout={file_timeout}s) -> {output_dir}")
    solid_main(cmd)
    new_bins = sum(1 for p in paths if (output_dir / f"{p.stem}.bin").exists())
    return new_bins


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry mechcad triangle extraction with structured failure logging.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/mechcad"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/processed"),
    )
    parser.add_argument(
        "--skip-log",
        type=Path,
        default=None,
        help="Default: <processed-dir>/triangles_skip.log",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Default: <processed-dir>/triangles_failure.jsonl",
    )
    parser.add_argument("--process-num", type=int, default=8)
    parser.add_argument(
        "--file-timeout",
        type=int,
        default=900,
        help="Per-file timeout in seconds (0=disable; not recommended)",
    )
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Write retry manifest JSON",
    )
    parser.add_argument(
        "--run-split",
        action="store_true",
        help="Regenerate datasplit.json after retry",
    )
    args = parser.parse_args()

    processed = args.processed_dir.resolve()
    skip_log = args.skip_log or (processed / "triangles_skip.log")
    failure_log = args.failure_log or (processed / "triangles_failure.jsonl")
    triangles_root = processed / "triangles"
    list_dir = processed / "retry_lists"
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None

    if failure_log.exists():
        failure_log.unlink()

    candidates, stats = collect_retry_candidates(
        args.source_root.resolve(),
        triangles_root,
        skip_log,
        categories,
    )
    grouped = group_by_category(candidates)

    print("=== retry manifest ===")
    print(json.dumps(stats, indent=2))
    print(f"categories with work: {sorted(grouped.keys())}")
    print(f"failure_log: {failure_log}")
    print(f"file_timeout: {args.file_timeout}s")

    manifest = {
        "stats": stats,
        "skip_log": str(skip_log),
        "failure_log": str(failure_log),
        "file_timeout": args.file_timeout,
        "by_category": {cat: len(paths) for cat, paths in sorted(grouped.items())},
    }

    success_by_cat = {}
    for cat in MECHCAD_CATEGORIES:
        if cat not in grouped:
            continue
        success_by_cat[cat] = run_category(
            cat,
            grouped[cat],
            triangles_root,
            args.source_root.resolve(),
            process_num=args.process_num,
            file_timeout=args.file_timeout,
            skip_log=skip_log,
            failure_log=failure_log,
            list_dir=list_dir,
        )

    manifest["new_bins_by_category"] = success_by_cat
    manifest["new_bins_total"] = sum(success_by_cat.values())

    manifest_path = args.manifest_out or (processed / "triangles_retry_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"New bins total: {manifest['new_bins_total']}")

    if args.run_split:
        split_script = REPO_ROOT / "process" / "split_mechcad_available.py"
        subprocess.run(
            [sys.executable, str(split_script), "--processed-dir", str(processed)],
            check=True,
        )


if __name__ == "__main__":
    main()
