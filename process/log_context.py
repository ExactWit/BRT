"""Thread/process-local sample anchor for mechcad preprocessing logs."""

from __future__ import annotations

import contextvars
import logging
from pathlib import Path

_sample_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("mechcad_sample", default=None)
_face_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar("mechcad_face", default=None)


def sample_anchor(step_path: str | Path) -> str:
    path = Path(step_path)
    return f"{path.parent.name}/{path.stem}"


def set_sample_context(step_path: str | Path | None) -> None:
    if step_path is None:
        _sample_ctx.set(None)
        _face_ctx.set(None)
        return
    _sample_ctx.set(sample_anchor(step_path))
    _face_ctx.set(None)


def set_face_context(face_index: int | None) -> None:
    _face_ctx.set(face_index)


def current_anchor() -> str:
    sample = _sample_ctx.get()
    face = _face_ctx.get()
    if not sample:
        return "unknown/unknown"
    if face is not None:
        return f"{sample}:face{face}"
    return sample


class SampleContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.sample = current_anchor()  # type: ignore[attr-defined]
        return True


def setup_mechcad_logging(level: int = logging.INFO) -> None:
    """Ensure console logs include [category/stem] anchors."""
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(levelname)s:%(name)s:[%(sample)s] %(message)s")
    has_context_handler = False
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.addFilter(SampleContextFilter())
            handler.setFormatter(fmt)
            has_context_handler = True
    if not has_context_handler:
        handler = logging.StreamHandler()
        handler.addFilter(SampleContextFilter())
        handler.setFormatter(fmt)
        root.addHandler(handler)
