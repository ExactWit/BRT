import os
import json
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.datasplit_meta import write_datasplit_meta


def find_step_files(directory):
    """
    Find all .bin files in the given directory and its subdirectories.
    """
    step_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bin"):
                step_files.append(os.path.join(root, file))
    return step_files


label_number = {
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


def split_files(file_paths, associated_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split the file paths into train, val, and test categories.

    Args:
        file_paths (list): A list of file paths to split.
        train_ratio (float): The ratio of files to include in the train set.
        val_ratio (float): The ratio of files to include in the val set.

    Returns:
        dict: A dictionary with keys 'train', 'val', and 'test' containing the split file paths.
    """

    random.shuffle(file_paths)

    items = []
    for i in range(len(file_paths)):
        dirname, filename = os.path.split(file_paths[i])
        _, label = os.path.split(dirname)
        graph_paths = os.path.join(associated_dir, label, filename)
        label = label_number.get(label, None)
        if label is not None:
            if os.path.exists(graph_paths):
                items.append({"face": file_paths[i], "topo": graph_paths, "label": label})

    total_files = len(items)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = items[:train_end]
    val_files = items[train_end:val_end]
    test_files = items[val_end:]

    print("total:", total_files)
    print(len(train_files), len(val_files), len(test_files))
    return {"train": train_files, "val": val_files, "test": test_files}


def main():
    parser = argparse.ArgumentParser(
        description="Search for .step and .stp files, split into train, val, and test sets, and output to a JSON file."
    )
    parser.add_argument("dir_triangles", type=str, help="Path to the directory to search for .bin files.")
    parser.add_argument("dir_topo", type=str, help="Path to the directory to search for associated graph files.")
    parser.add_argument("output_json", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    # Find all .step and .stp files
    step_files = find_step_files(args.dir_triangles)

    # Split the files into train, val, and test sets
    split_data = split_files(step_files, args.dir_topo)

    # Output the split data to a JSON file
    output_path = Path(args.output_json)
    with open(output_path, "w") as json_file:
        json.dump(split_data, json_file, indent=4)

    counts = {key: len(split_data[key]) for key in ("train", "val", "test")}
    meta_path = write_datasplit_meta(
        output_path,
        dataset_id="mechcad",
        split_source_json=None,
        counts=counts,
        extra={
            "dir_triangles": str(Path(args.dir_triangles).resolve()),
            "dir_topo": str(Path(args.dir_topo).resolve()),
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
    )

    print(f"Split data has been written to {args.output_json}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
