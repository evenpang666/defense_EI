"""Helpers to build supervisor input context from scene snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scene_context_from_snapshot(snapshot_path: Path, front_image_path: Path) -> dict[str, Any]:
    if not snapshot_path.exists():
        return {"objects": [], "camera": {}, "source_snapshot": str(snapshot_path)}
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return {"objects": [], "camera": {}, "source_snapshot": str(snapshot_path), "parse_error": True}

    assets = payload.get("assets", [])
    object_entries: list[dict[str, Any]] = []
    if isinstance(assets, list):
        for item in assets:
            if not isinstance(item, dict):
                continue
            object_entries.append(
                {
                    "name": item.get("name"),
                    "pos": item.get("pos"),
                    "quat": item.get("quat"),
                }
            )
    return {
        "objects": object_entries,
        "camera": {"front_image_path": str(front_image_path.resolve())},
        "source_snapshot": str(snapshot_path.resolve()),
    }
