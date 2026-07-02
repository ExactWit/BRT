"""Read PyTorch Lightning checkpoint metadata for experiment records."""

from __future__ import annotations

import pathlib
import time
from typing import Any


def load_pl_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def checkpoint_record(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    ckpt = load_pl_checkpoint(path)
    epoch = ckpt.get("epoch")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "filename": path.name,
        "global_step": int(ckpt["global_step"]) if ckpt.get("global_step") is not None else None,
    }
    if epoch is not None:
        epoch_int = int(epoch)
        record["epoch"] = epoch_int
        # 1-based epoch number for human-readable logs (PL stores 0-based).
        record["epoch_1based"] = epoch_int + 1
    callbacks = ckpt.get("callbacks") or {}
    for cb_state in callbacks.values():
        if isinstance(cb_state, dict) and cb_state.get("best_model_score") is not None:
            record["val_iou"] = float(cb_state["best_model_score"])
            break
    return record


def resolve_eval_checkpoint(run_dir: pathlib.Path) -> pathlib.Path | None:
    """Checkpoint for test/viz: prefer best validation IoU, else last."""
    for name in ("best.ckpt", "last.ckpt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def build_training_checkpoints_summary(
    run_dir: pathlib.Path,
    *,
    checkpoint_callback: Any,
    monitor: str = "val_iou",
    mode: str = "max",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "monitor": monitor,
        "mode": mode,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    best_path = pathlib.Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else run_dir / "best.ckpt"
    last_path = run_dir / "last.ckpt"
    if best_path.exists():
        best = checkpoint_record(best_path) or {}
        if checkpoint_callback.best_model_score is not None:
            best["val_iou"] = float(checkpoint_callback.best_model_score)
        summary["best"] = best
    if last_path.exists():
        summary["last"] = checkpoint_record(last_path)
    return summary
