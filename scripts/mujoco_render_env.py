"""Pick a valid MuJoCo GL backend before ``import mujoco``.

Linux headless defaults to EGL; Windows only allows GLFW/WGL (EGL raises
``RuntimeError: invalid value for environment variable MUJOCO_GL: egl``).

Call :func:`ensure_mujoco_gl_environment` at process startup, before importing mujoco.
If ``MUJOCO_GL`` is already set to a valid value for the current OS, it is kept.
Invalid values (e.g. EGL on Windows) are replaced with a safe default.
"""

from __future__ import annotations

import os
import platform

# Linux-only in mujoco Python gl_context.py
_WIN_INVALID = frozenset({"egl", "osmesa", "glx", "cgl"})
# Not supported on Darwin in the same sense as Linux EGL
_DARWIN_INVALID = frozenset({"egl", "osmesa", "glx", "wgl"})


def ensure_mujoco_gl_environment() -> None:
    system = platform.system()
    raw = os.environ.get("MUJOCO_GL", "")
    cur = raw.lower().strip()

    if system == "Windows":
        if cur in _WIN_INVALID:
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ.setdefault("MUJOCO_GL", "glfw")
        pop = os.environ.get("PYOPENGL_PLATFORM", "").lower().strip()
        if pop in ("egl", "osmesa"):
            del os.environ["PYOPENGL_PLATFORM"]
    elif system == "Darwin":
        if cur in _DARWIN_INVALID:
            os.environ["MUJOCO_GL"] = "glfw"
        else:
            os.environ.setdefault("MUJOCO_GL", "glfw")
        pop = os.environ.get("PYOPENGL_PLATFORM", "").lower().strip()
        if pop in ("egl", "osmesa", "glx"):
            del os.environ["PYOPENGL_PLATFORM"]
    else:
        # Linux and similar: EGL is the usual headless default.
        os.environ.setdefault("MUJOCO_GL", "egl")

    cur = os.environ.get("MUJOCO_GL", "").lower().strip()
    if system == "Linux" and cur == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
