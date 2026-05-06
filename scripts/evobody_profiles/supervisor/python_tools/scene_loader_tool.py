"""Scene loader tool for supervisor image bootstrap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import execute_code_runtime as exec_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(path_str: str) -> Path:
    path = Path((path_str or "").strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _ensure_runtime_initialized() -> None:
    if not hasattr(exec_runtime, "RUNTIME") or getattr(exec_runtime, "RUNTIME", None) is None:
        exec_runtime.RUNTIME = exec_runtime.CliRuntime()


def scene_loader_script(
    scene_json_path: str = "chemistry.json",
    current_view_path: str = "logs/current_view.png",
    randomize_texture: bool = False,
) -> str:
    """Load scene JSON and save current front-camera image.

    This is the supervisor bootstrap tool ("场景载入脚本"): when current_view.png
    is missing, call this tool first, then continue planning with the generated image.
    """
    try:
        _ensure_runtime_initialized()
        scene_path = _resolve_path(scene_json_path)
        view_path = _resolve_path(current_view_path)
        if not scene_path.exists():
            return json.dumps(
                {"ok": False, "error": f"scene json not found: {scene_path}"},
                ensure_ascii=False,
            )

        exec_runtime._safe_load_scene(scene_path, randomize_texture=bool(randomize_texture))
        exec_runtime._save_front(view_path)
        return json.dumps(
            {
                "ok": True,
                "signal": "CURRENT_VIEW_READY",
                "scene_json_path": str(scene_path),
                "current_view_path": str(view_path),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

