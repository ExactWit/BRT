#!/usr/bin/env python3
"""Export Fusion360 STEP files that failed BRT triangle extraction."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def load_split_stems(dataset_json: Path) -> dict[str, set[str]]:
    with open(dataset_json) as f:
        data = json.load(f)
    return {
        "train": set(data["training_set"]),
        "val": set(data["validation_set"]),
        "test": set(data["test_set"]),
    }


def parse_triangle_log(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}

    text = log_path.read_text(errors="ignore")
    errors: dict[str, str] = {}

    for stem in re.findall(r"Build Error in (\S+)", text):
        errors.setdefault(stem, "Build Error")

    for stem in re.findall(r"Read Step Error in (\S+)", text):
        errors[stem] = "Read Step Error"

    parts = re.split(r"Build Error in (\S+)\n", text)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        stem = parts[i]
        body = parts[i + 1]
        if "need at least one array to stack" in body:
            err = "ValueError: empty triangles after face decomposition"
        elif "TypeError" in body and "NoneType" in body:
            err = "TypeError: triangle subdivision returned None (Bezier convert failed)"
        elif "Standard_ConstructionError" in body and "Geom_BezierSurface" in body:
            err = "RuntimeError: OCC Geom_BezierSurface Standard_ConstructionError"
        elif "Standard_DomainErrorGeomConvert" in body:
            err = "RuntimeError: OCC BSplineSurfaceToBezierSurface domain error"
        elif "bad nurbs" in body:
            err = "RuntimeError: bad nurbs"
        elif "no patches" in body:
            err = "RuntimeError: no patches"
        elif "ValueError" in body:
            err = "ValueError (geometry/decomposition)"
        else:
            runtime = re.findall(r"RuntimeError: ([^\n]+)", body)
            err = f"RuntimeError: {runtime[-1]}" if runtime else "Unknown build error"
        errors[stem] = err

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/hdd/datasets/s2.0.0"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/processed/brt",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Default: <repo>/process/logs/fusion360_triangles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <processed-dir>",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    processed = args.processed_dir or (args.dataset_dir / "processed" / "brt")
    log_path = args.log or (repo / "process" / "logs" / "fusion360_triangles")
    out_dir = args.output_dir or processed
    step_dir = args.dataset_dir / "breps" / "step"
    dataset_json = args.dataset_dir / "processed" / "dataset.json"

    split_of = {}
    for split_name, stems in load_split_stems(dataset_json).items():
        for stem in stems:
            split_of[stem] = split_name

    tri_dir = processed / "triangles"
    topo_dir = processed / "topology" / "brt"
    tri_stems = {p.stem for p in tri_dir.glob("*.bin")}
    missing = sorted(set(split_of) - tri_stems)
    log_errors = parse_triangle_log(log_path)

    rows = []
    for stem in missing:
        stp = step_dir / f"{stem}.stp"
        rows.append(
            {
                "stem": stem,
                "split": split_of.get(stem, ""),
                "step_path": str(stp),
                "step_exists": stp.exists(),
                "topo_exists": (topo_dir / f"{stem}.bin").exists(),
                "error": log_errors.get(stem, "unknown (no triangle .bin; check preprocess log)"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "triangle_failures.csv"
    json_path = out_dir / "triangle_failures.json"
    txt_path = out_dir / "triangle_failures_stems.txt"
    summary_path = out_dir / "triangle_failures_summary.txt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    with open(txt_path, "w") as f:
        for row in rows:
            f.write(row["step_path"] + "\n")

    ctr = Counter(r["error"] for r in rows)
    with open(summary_path, "w") as f:
        f.write(f"total_failed={len(rows)}\n")
        f.write(f"log_matched={sum(1 for r in rows if r['stem'] in log_errors)}\n")
        for split in ("train", "val", "test"):
            f.write(f"{split}={sum(1 for r in rows if r['split']==split)}\n")
        f.write("\nerror_breakdown:\n")
        for err, count in ctr.most_common():
            f.write(f"  {count}\t{err}\n")

    print(f"failed triangle extraction: {len(rows)}")
    print(f"log matched: {sum(1 for r in rows if r['stem'] in log_errors)}")
    print("by split:", {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")})
    print("top errors:")
    for err, count in ctr.most_common(5):
        print(f"  {count:4d}  {err}")
    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {txt_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
