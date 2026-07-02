"""Build and persist experiment metadata for train/test reproducibility."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import time
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(args: list[str], repo_dir: pathlib.Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_git_info(repo_dir: pathlib.Path, branch: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    info["branch"] = branch or run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    info["commit"] = run_git(["rev-parse", "--short", "HEAD"], repo_dir)
    info["commit_full"] = run_git(["rev-parse", "HEAD"], repo_dir)
    dirty = run_git(["status", "--porcelain"], repo_dir)
    info["dirty"] = bool(dirty)
    if dirty:
        info["dirty_files"] = dirty.splitlines()[:20]
    info["remote"] = run_git(["config", "--get", "remote.origin.url"], repo_dir)
    return info


def load_datasplit_meta(processed_dir: pathlib.Path) -> dict[str, Any] | None:
    meta_path = processed_dir / "datasplit_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def summarize_datasplit(datasplit_path: pathlib.Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "datasplit_json": str(datasplit_path.resolve()),
        "exists": datasplit_path.exists(),
    }
    if not datasplit_path.exists():
        return summary

    stat = datasplit_path.stat()
    summary["size_bytes"] = stat.st_size
    summary["mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
    summary["sha256"] = sha256_file(datasplit_path)

    with open(datasplit_path, encoding="utf-8") as f:
        split = json.load(f)

    counts = {}
    for key in ("train", "val", "test"):
        if key in split and isinstance(split[key], list):
            counts[key] = len(split[key])
    summary["counts"] = counts
    return summary


def collect_datasplit_info(
    dataset_dir: pathlib.Path,
    dataset_id: str | None = None,
    split_source_json: str | None = None,
) -> dict[str, Any]:
    processed_dir = pathlib.Path(dataset_dir)
    datasplit_path = processed_dir / "datasplit.json"
    info = summarize_datasplit(datasplit_path)
    info["dataset_id"] = dataset_id

    sidecar = load_datasplit_meta(processed_dir)
    if sidecar is not None:
        info["sidecar"] = sidecar
        if split_source_json is None:
            split_source_json = sidecar.get("split_source_json")

    if split_source_json:
        source_path = pathlib.Path(split_source_json)
        info["split_source_json"] = str(source_path.resolve()) if source_path.exists() else split_source_json
        if source_path.exists():
            info["split_source_sha256"] = sha256_file(source_path)
            info["split_source_mtime"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(source_path.stat().st_mtime)
            )
    return info


def build_experiment_metadata(
    *,
    repo_dir: pathlib.Path,
    run_dir: pathlib.Path,
    experiment_name: str,
    log_name: str,
    log_version: str,
    run_tag: str | None = None,
    dataset_dir: str,
    dataset_id: str | None = None,
    git_branch: str | None = None,
    note: str | None = None,
    split_source_json: str | None = None,
    train_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    train_args = train_args or {}
    processed_dir = pathlib.Path(dataset_dir)
    tag = run_tag or log_version

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "results_layout": "dataset/date/tag",
        "run": {
            "experiment_name": experiment_name,
            "log_name": log_name,
            "log_version": log_version,
            "run_tag": tag,
            "run_dir": str(run_dir.resolve()),
            "created_at": created_at,
            "created_at_unix": int(time.time()),
        },
        "git": collect_git_info(repo_dir, branch=git_branch),
        "dataset": {
            "id": dataset_id,
            "processed_dir": str(processed_dir.resolve()),
            "num_classes": train_args.get("num_classes"),
            "num_control_pts": train_args.get("num_control_pts"),
        },
        "datasplit": collect_datasplit_info(
            processed_dir,
            dataset_id=dataset_id,
            split_source_json=split_source_json,
        ),
        "train": train_args,
        "note": note or "",
        "tags": [],
    }
    return metadata


def write_experiment_metadata(path: pathlib.Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_experiment_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_experiment_metadata(path: pathlib.Path, updates: dict[str, Any]) -> dict[str, Any]:
    metadata = load_experiment_metadata(path) or {}
    metadata.update(updates)
    write_experiment_metadata(path, metadata)
    return metadata


def meta_git_branch(meta: dict[str, Any]) -> str | None:
    if meta.get("git_branch"):
        return meta["git_branch"]
    return (meta.get("git") or {}).get("branch")


def meta_dataset_id(meta: dict[str, Any]) -> str | None:
    dataset = meta.get("dataset")
    if isinstance(dataset, str):
        return dataset
    if isinstance(dataset, dict):
        return dataset.get("id")
    return None


def meta_dataset_dir(meta: dict[str, Any]) -> str | None:
    if meta.get("dataset_dir"):
        return meta["dataset_dir"]
    dataset = meta.get("dataset")
    if isinstance(dataset, dict):
        return dataset.get("processed_dir")
    return None


def meta_num_classes(meta: dict[str, Any]) -> int | None:
    if meta.get("num_classes") is not None:
        return meta["num_classes"]
    dataset = meta.get("dataset")
    if isinstance(dataset, dict) and dataset.get("num_classes") is not None:
        return dataset["num_classes"]
    train = meta.get("train")
    if isinstance(train, dict) and train.get("num_classes") is not None:
        return train["num_classes"]
    return None


def meta_note(meta: dict[str, Any]) -> str:
    return (meta.get("note") or "").strip()
