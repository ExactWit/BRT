import argparse
import json
import os
from pathlib import Path


def build_item(stem, triangles_dir, topo_dir, seg_dir):
    face = triangles_dir / f"{stem}.bin"
    topo = topo_dir / f"{stem}.bin"
    label = seg_dir / f"{stem}.seg"
    if face.exists() and topo.exists() and label.exists():
        return {
            "face": str(face),
            "topo": str(topo),
            "label": str(label),
        }
    return None


def convert_stems(stems, triangles_dir, topo_dir, seg_dir):
    items = []
    missing = 0
    for stem in stems:
        item = build_item(stem, triangles_dir, topo_dir, seg_dir)
        if item is not None:
            items.append(item)
        else:
            missing += 1
    return items, missing


def load_split_stems(split_json_path):
    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    if "training_set" in split_data:
        return {
            "train": split_data["training_set"],
            "val": split_data["validation_set"],
            "test": split_data["test_set"],
        }

    if "train" in split_data and "test" in split_data:
        from sklearn.model_selection import train_test_split

        train_stems, val_stems = train_test_split(
            split_data["train"], test_size=0.18, random_state=567
        )
        return {
            "train": train_stems,
            "val": val_stems,
            "test": split_data["test"],
        }

    raise ValueError(
        f"Unrecognized split format in {split_json_path}. "
        "Expected BRepNet dataset.json (training_set/validation_set/test_set) "
        "or Fusion train_test.json (train/test)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build BRT datasplit.json for Fusion 360 Gallery segmentation dataset."
    )
    parser.add_argument(
        "processed_dir",
        type=str,
        help="BRT processed root containing triangles/ and topology/brt/ subdirs.",
    )
    parser.add_argument(
        "seg_dir",
        type=str,
        help="Directory with official *.seg face label files (e.g. breps/seg).",
    )
    parser.add_argument(
        "split_json",
        type=str,
        help="Official split file: BRepNet processed/dataset.json (preferred) or train_test.json.",
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Output datasplit.json path.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    triangles_dir = processed_dir / "triangles"
    topo_dir = processed_dir / "topology" / "brt"
    seg_dir = Path(args.seg_dir)

    for path, name in [
        (triangles_dir, "triangles"),
        (topo_dir, "topology/brt"),
        (seg_dir, "seg"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    split_stems = load_split_stems(args.split_json)

    splits = {}
    total_missing = 0
    for split_name in ("train", "val", "test"):
        stems = split_stems[split_name]
        items, missing = convert_stems(stems, triangles_dir, topo_dir, seg_dir)
        splits[split_name] = items
        total_missing += missing
        print(
            f"{split_name}: {len(items)} samples "
            f"({missing} skipped, missing triangles/topo/label)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Wrote {args.output_json} (total skipped: {total_missing})")


if __name__ == "__main__":
    main()
