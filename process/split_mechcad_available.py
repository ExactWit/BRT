#!/usr/bin/env python3
"""Build mechcad datasplit.json from already-processed triangle + topology pairs.

Skips categories/samples missing either side (e.g. stuck triangle extraction).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.datasplit_meta import write_datasplit_meta

LABEL_NUMBER = {
    "bearing": 0,
    "bolt": 1,
    "bracket": 2,
    "coupling": 3,
    "flange": 4,
    "gear": 5,
    "nut": 6,
    "pulley": 7,
    "screw": 8,
    "shaft": 9,
}


def collect_pairs(triangles_root: Path, topo_root: Path) -> tuple[list[dict], dict]:
    items: list[dict] = []
    stats: dict[str, dict] = {}

    for category, label_id in LABEL_NUMBER.items():
        tri_dir = triangles_root / category
        topo_dir = topo_root / category
        tri_stems = {p.stem for p in tri_dir.glob("*.bin")} if tri_dir.is_dir() else set()
        topo_stems = {p.stem for p in topo_dir.glob("*.bin")} if topo_dir.is_dir() else set()
        paired = sorted(tri_stems & topo_stems)
        stats[category] = {
            "triangles": len(tri_stems),
            "topology": len(topo_stems),
            "paired": len(paired),
            "label": label_id,
        }
        for stem in paired:
            items.append(
                {
                    "face": str((tri_dir / f"{stem}.bin").resolve()),
                    "topo": str((topo_dir / f"{stem}.bin").resolve()),
                    "label": label_id,
                    "category": category,
                    "stem": stem,
                }
            )
    return items, stats


def split_items(items: list[dict], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(description="Build mechcad datasplit from available processed bins.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/data/hdd/datasets/mechcad/processed"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=567)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <processed-dir>/datasplit.json",
    )
    args = parser.parse_args()

    processed = args.processed_dir.resolve()
    triangles_root = processed / "triangles"
    topo_root = processed / "topology" / "brt"
    output = args.output or (processed / "datasplit.json")

    items, stats = collect_pairs(triangles_root, topo_root)
    if not items:
        raise SystemExit(f"No paired samples under {triangles_root} and {topo_root}")

    splits = split_items(items, args.train_ratio, args.val_ratio, args.seed)
    counts = {k: len(v) for k, v in splits.items()}

    payload = {k: [{key: rec[key] for key in ("face", "topo", "label")} for rec in v] for k, v in splits.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_path = processed / "datasplit_available_summary.json"
    summary = {
        "total_paired": len(items),
        "counts": counts,
        "per_category": stats,
        "triangles_root": str(triangles_root),
        "topo_root": str(topo_root),
        "note": "Only samples with both triangle and topology bins are included.",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    meta_path = write_datasplit_meta(
        output,
        dataset_id="mechcad",
        split_source_json=None,
        counts=counts,
        extra={
            "split_script": "process/split_mechcad_available.py",
            "triangles_root": str(triangles_root),
            "topo_root": str(topo_root),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "per_category": stats,
        },
    )

    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {meta_path}")
    print("counts:", counts)
    for cat, row in stats.items():
        if row["paired"]:
            print(f"  {cat}: paired={row['paired']} (tri={row['triangles']}, topo={row['topology']})")


if __name__ == "__main__":
    main()
