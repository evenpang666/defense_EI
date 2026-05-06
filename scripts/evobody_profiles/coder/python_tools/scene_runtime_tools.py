"""Scene runtime tools for Evobody coder agent."""

from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import execute_code_runtime as exec_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_ERROR_PATH = LOG_ROOT / "last_execution_error.txt"


def _resolve_path(path_str: str) -> Path:
    path = Path((path_str or "").strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _ensure_runtime_initialized() -> None:
    if not hasattr(exec_runtime, "RUNTIME") or getattr(exec_runtime, "RUNTIME", None) is None:
        exec_runtime.RUNTIME = exec_runtime.CliRuntime()


def load_mujoco_scene_from_json(scene_json_path: str, randomize_texture: bool = False) -> str:
    """Load MuJoCo scene from scene json file."""
    try:
        _ensure_runtime_initialized()
        scene_path = _resolve_path(scene_json_path)
        if not scene_path.exists():
            return json.dumps({"ok": False, "error": f"scene json not found: {scene_path}"}, ensure_ascii=False)
        exec_runtime._safe_load_scene(scene_path, randomize_texture=bool(randomize_texture))
        return json.dumps(
            {
                "ok": True,
                "scene_json_path": str(scene_path),
                "signal": "SCENE_LOADED",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def execute_generated_code_in_loaded_scene(
    code: str,
    final_image_path: str = "logs/finished_view.png",
    error_output_path: str = "logs/last_execution_error.txt",
) -> str:
    """Execute generated code in current loaded scene and save final camera image."""
    image_path = _resolve_path(final_image_path)
    error_path = _resolve_path(error_output_path) if (error_output_path or "").strip() else DEFAULT_ERROR_PATH
    error_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _ensure_runtime_initialized()
        runtime = getattr(exec_runtime, "RUNTIME", None)
        if runtime is None or runtime.model is None or runtime.data is None:
            msg = "runtime scene is not loaded, call load_mujoco_scene_from_json first"
            error_path.write_text(msg, encoding="utf-8")
            return json.dumps({"ok": False, "error": msg, "error_log_path": str(error_path)}, ensure_ascii=False)

        result = exec_runtime.execute_code_with_recording(code or "", task_prompt="evobody_codegen", save_video=False)
        exec_runtime._save_front(image_path)

        if bool(result.get("success", False)):
            if error_path.exists():
                error_path.unlink()
            return json.dumps(
                {
                    "ok": True,
                    "signal": "EXECUTION_OK",
                    "final_image_path": str(image_path),
                },
                ensure_ascii=False,
            )

        err = str(result.get("error", "unknown execution error"))
        error_path.write_text(err, encoding="utf-8")
        return json.dumps(
            {
                "ok": False,
                "error": err,
                "final_image_path": str(image_path),
                "error_log_path": str(error_path),
            },
            ensure_ascii=False,
        )
    except Exception:
        err = traceback.format_exc()
        error_path.write_text(err, encoding="utf-8")
        return json.dumps(
            {
                "ok": False,
                "error": "execution raised exception",
                "error_log_path": str(error_path),
            },
            ensure_ascii=False,
        )


def syntax_check_code_via_ast_tree(code: str) -> str:
    """Check syntax by parsing AST tree and return a machine-readable signal."""
    try:
        source = (code or "").strip()
        tree = ast.parse(source)
        _ = ast.dump(tree, annotate_fields=False)
        return json.dumps({"ok": True, "signal": "NO_SYNTAX_ERROR"}, ensure_ascii=False)
    except SyntaxError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"SyntaxError: {exc}",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "error": f"SyntaxCheckFailed: {exc}",
            },
            ensure_ascii=False,
        )
