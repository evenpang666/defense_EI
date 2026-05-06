"""Shared runtime helpers for executing action code in a loaded scene.

This module contains the runtime logic previously hosted in the CLI script,
so other entrypoints (for example Gradio) can depend on it directly.
"""

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import imageio.v2 as imageio
import glfw
import mujoco
import mujoco.viewer
import numpy as np


def _try_load_optional_mjlab_plugin() -> None:
    loader = getattr(mujoco, "mj_loadPluginLibrary", None)
    if not callable(loader):
        return
    candidates: list[str] = []
    env_candidate = str(os.environ.get("MJLAB_PLUGIN_PATH", "")).strip()
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend(["./libmjlab.so.3.3.0", "./mjlab.dll", "./libmjlab.dylib"])
    for plugin_path in candidates:
        try:
            loader(plugin_path)
            return
        except Exception:
            continue


_try_load_optional_mjlab_plugin()

_VIEWER_THREAD_LOCK = threading.Lock()
_VIEWER_THREAD: threading.Thread | None = None
_VIEWER_STOP_EVENT = threading.Event()
_TELEOP_LOCK = threading.Lock()
_KEYBOARD_LISTENING = False
_TELEOP_STATE: "TeleopState | None" = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"
ATOMIC_OPS_PATH = Path(__file__).resolve().parent / "evoma_atomic_ops.py"


def _load_atomic_ops_source() -> str:
    """Read atomic-ops source and inject it into user execution namespace."""
    try:
        return ATOMIC_OPS_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


from generate_cli import (  # noqa: E402
    EEController,
    CliRuntime,
    ExecutionRecorder,
    quat_from_euler,
    quat_mul,
    quat_slerp,
)


def _normalize_saved_subtask_prompt(prompt: str | None) -> str:
    text = " ".join((prompt or "").replace("\x00", " ").split()).strip().lower()
    if not text:
        return "manual_execution"
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "manual_execution"


def _log(msg: str, level: str = "INFO") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


@dataclass
class TeleopConfig:
    key_pos_step_m: float = 0.008
    key_rot_step_rad: float = 0.08
    max_pos_step_m: float = 0.01
    max_rot_step_rad: float = 0.15
    solve_substeps: int = 2
    sim_steps_per_frame: int = 2
    gripper_open: float = 0.0
    gripper_close: float = 255.0


class TeleopState:
    def __init__(self, ee: EEController):
        self.gripper_closed = False
        self.pending_toggle_gripper = False
        self.pending_reset_target = False
        self.pending_dp = np.zeros(3, dtype=float)
        self.pending_rv = np.zeros(3, dtype=float)
        self.ctrl_latch_until = 0.0

        cur_pos, cur_quat = ee.ee_pose()
        self.target_pos = cur_pos.copy()
        self.target_quat = cur_quat.copy()

    def request_motion(self, dp: np.ndarray, rv: np.ndarray) -> None:
        self.pending_dp = self.pending_dp + np.asarray(dp, dtype=float)
        self.pending_rv = self.pending_rv + np.asarray(rv, dtype=float)

    def request_toggle_gripper(self) -> None:
        self.pending_toggle_gripper = True

    def request_reset_target(self) -> None:
        self.pending_reset_target = True

    def request_ctrl_latch(self, duration_s: float = 0.35) -> None:
        self.ctrl_latch_until = max(self.ctrl_latch_until, time.time() + float(duration_s))

    def is_ctrl_latched(self) -> bool:
        return time.time() <= self.ctrl_latch_until

    def pop_pending(self) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        dp = self.pending_dp.copy()
        rv = self.pending_rv.copy()
        tg = self.pending_toggle_gripper
        reset = self.pending_reset_target
        self.pending_dp[:] = 0.0
        self.pending_rv[:] = 0.0
        self.pending_toggle_gripper = False
        self.pending_reset_target = False
        return dp, rv, tg, reset


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(qq))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return qq / n


def _clamp_vec_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


def _apply_keyboard_delta(state: TeleopState, dp: np.ndarray, rv: np.ndarray, cfg: TeleopConfig) -> None:
    dp = _clamp_vec_norm(np.asarray(dp, dtype=float), float(cfg.max_pos_step_m))
    rv = _clamp_vec_norm(np.asarray(rv, dtype=float), float(cfg.max_rot_step_rad))
    angle = float(np.linalg.norm(rv))
    dq_scaled = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if angle < 1e-8 else _quat_from_axis_angle(rv / angle, angle)
    state.target_pos = state.target_pos + dp
    state.target_quat = _normalize_quat(quat_mul(dq_scaled, state.target_quat))


def _key_to_motion_delta(key: int, cfg: TeleopConfig) -> tuple[np.ndarray, np.ndarray]:
    dp = np.zeros(3, dtype=float)
    rv = np.zeros(3, dtype=float)
    pos_step = float(cfg.key_pos_step_m)

    if key == glfw.KEY_UP:
        dp[0] -= pos_step
    elif key == glfw.KEY_DOWN:
        dp[0] += pos_step
    elif key == glfw.KEY_LEFT:
        dp[1] -= pos_step
    elif key == glfw.KEY_RIGHT:
        dp[1] += pos_step
    elif key == glfw.KEY_PAGE_UP:
        dp[2] += pos_step
    elif key == glfw.KEY_PAGE_DOWN:
        dp[2] -= pos_step

    return dp, rv


def _key_to_ctrl_motion_delta(key: int, cfg: TeleopConfig) -> tuple[np.ndarray, np.ndarray]:
    dp = np.zeros(3, dtype=float)
    rv = np.zeros(3, dtype=float)
    rot_step = float(cfg.key_rot_step_rad)

    if key == glfw.KEY_UP:
        rv[0] += rot_step
    elif key == glfw.KEY_DOWN:
        rv[0] -= rot_step
    elif key == glfw.KEY_LEFT:
        rv[1] += rot_step
    elif key == glfw.KEY_RIGHT:
        rv[1] -= rot_step
    elif key == glfw.KEY_PAGE_UP:
        rv[2] += rot_step
    elif key == glfw.KEY_PAGE_DOWN:
        rv[2] -= rot_step

    return dp, rv


def set_keyboard_listening(enabled: bool) -> None:
    global _KEYBOARD_LISTENING
    with _TELEOP_LOCK:
        _KEYBOARD_LISTENING = bool(enabled)


def _safe_load_scene(scene_json_path: Path, randomize_texture: bool = False) -> None:
    """Load a scene from JSON into the runtime."""
    global RUNTIME
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_json_path}")

    with RUNTIME.lock:
        if RUNTIME.model is not None and RUNTIME.scene_xml_path is not None:
            RUNTIME._cleanup_cached_build_runtime_xml()

        RUNTIME.load_scene_config(
            scene_json_path,
            randomize_texture=randomize_texture,
        )
        _log(f"Scene loaded: {scene_json_path}")


def _cleanup_runtime_scene_cache() -> None:
    """Cleanup generated runtime XML cache directory if present."""
    global RUNTIME
    if "RUNTIME" not in globals() or RUNTIME is None:
        return

    with RUNTIME.lock:
        if RUNTIME.scene_xml_path is None:
            return
        try:
            RUNTIME._cleanup_cached_build_runtime_xml()
            _log("Cleaned up generated runtime XML cache.")
        except Exception as exc:
            _log(f"Failed to clean runtime XML cache: {exc}", level="WARNING")


def stop_viewer() -> None:
    global _VIEWER_THREAD
    with _VIEWER_THREAD_LOCK:
        _VIEWER_STOP_EVENT.set()
        thread = _VIEWER_THREAD
        _VIEWER_THREAD = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)


def start_viewer() -> None:
    global _VIEWER_THREAD, _TELEOP_STATE, _KEYBOARD_LISTENING
    with _VIEWER_THREAD_LOCK:
        stop_thread = _VIEWER_THREAD
        _VIEWER_STOP_EVENT.set()
    if stop_thread is not None and stop_thread.is_alive():
        stop_thread.join(timeout=2.0)

    with _VIEWER_THREAD_LOCK:
        _VIEWER_STOP_EVENT.clear()
        if "RUNTIME" not in globals() or RUNTIME is None:
            raise RuntimeError("Runtime is not initialized")
        if RUNTIME.model is None or RUNTIME.data is None:
            raise RuntimeError("Runtime scene is not loaded")
        if RUNTIME.ee is None:
            raise RuntimeError("Runtime end-effector controller is not initialized")
        with _TELEOP_LOCK:
            _TELEOP_STATE = TeleopState(RUNTIME.ee)
            _KEYBOARD_LISTENING = False
        RUNTIME._viewer_ready.clear()
        _VIEWER_THREAD = threading.Thread(target=_viewer_thread, daemon=True)
        _VIEWER_THREAD.start()


def _apply_robot_arm_perturbation(enable: bool) -> None:
    """Apply perturbation to robot arm state."""
    global RUNTIME
    if enable:
        with RUNTIME.lock:
            if RUNTIME.model is not None and RUNTIME.data is not None:
                jnt_span = RUNTIME.ee.jnt_span
                perturbation = np.random.normal(0, 0.01, 6)
                RUNTIME.data.qpos[jnt_span] += perturbation
                mujoco.mj_forward(RUNTIME.model, RUNTIME.data)


def _save_front(output_path: Path) -> None:
    """Save front view image."""
    global RUNTIME
    with RUNTIME.lock:
        if RUNTIME.model is None or RUNTIME.data is None:
            return
        renderer = mujoco.Renderer(RUNTIME.model, height=512, width=512)
        renderer.update_scene(RUNTIME.data)
        pixels = renderer.render()
        renderer.close()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, pixels)
        _log(f"Saved front view to {output_path}")


def execute_code_with_recording(
    code: str,
    task_prompt: str | None = None,
    save_video: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute atomic action code with recording."""
    global RUNTIME

    if output_dir is None:
        output_dir = LOG_ROOT / "execution_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    class CustomExecutionRecorder(ExecutionRecorder):
        def __init__(self, runtime: CliRuntime, output_dir: Path, task_prompt: str | None = None, enabled: bool = True):
            # Initialize recorder state fields expected by base record()/state_action() logic.
            super().__init__(runtime, task_prompt=task_prompt, enabled=False)
            self.enabled = enabled
            self.task_prompt = _normalize_saved_subtask_prompt(task_prompt)
            self.run_dir = output_dir
            self.frames_dir = output_dir / "frames"
            self.wrist_frames_dir = output_dir / "wrist_frames"

            if not enabled:
                return

            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.wrist_frames_dir.mkdir(parents=True, exist_ok=True)

            model = runtime.model
            data = runtime.data
            if model is None or data is None:
                raise RuntimeError("Runtime model/data is not initialized")

            self.renderer = mujoco.Renderer(model, 256, 256)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self.renderer._scene_option.sitegroup[:] = False

            for key, cam_name in {
                "image": "table_cam_front",
                "wrist_image": "/ur:wrist_cam",
            }.items():
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if camera_id < 0:
                    continue
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(camera)
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = camera_id
                self.cameras[key] = camera

            if "image" not in self.cameras:
                raise ValueError("Camera 'table_cam_front' not found")
            with self.runtime.lock:
                ee = self.runtime.ee
                if ee is not None:
                    pos, quat = ee.ee_pose()
                    self._prev_ee_pos = np.asarray(pos, dtype=float).copy()
                    self._prev_ee_quat = np.asarray(quat, dtype=float).copy()

    recorder = CustomExecutionRecorder(RUNTIME, output_dir, task_prompt=task_prompt, enabled=save_video)
    RUNTIME._viewer_ready.wait(timeout=10)
    with RUNTIME.lock:
        RUNTIME._busy = True

    results = {
        "success": False,
        "error": None,
        "video_path": None,
        "state_action_path": None,
        "output_dir": str(output_dir),
    }

    try:
        def _step(tag: str) -> None:
            with RUNTIME.lock:
                mujoco.mj_step(RUNTIME.model, RUNTIME.data)
                recorder.record(tag)

        def ee_pose() -> tuple[np.ndarray, np.ndarray]:
            with RUNTIME.lock:
                pos, quat = RUNTIME.ee.ee_pose()
                return pos.copy(), quat.copy()

        def move_to(pos, quat=None, num_steps: int = 100) -> None:
            with RUNTIME.lock:
                start_pos, start_quat = RUNTIME.ee.ee_pose()
            target_pos = np.asarray(pos, dtype=float)
            target_quat = np.asarray(quat, dtype=float) if quat is not None else start_quat
            target_quat = target_quat / np.linalg.norm(target_quat)
            num_steps_i = max(1, int(num_steps))
            for i in range(1, num_steps_i + 1):
                t = i / num_steps_i
                waypoint_pos = (1.0 - t) * start_pos + t * target_pos
                waypoint_quat = quat_slerp(start_quat, target_quat, t)
                for _ in range(2):
                    with RUNTIME.lock:
                        RUNTIME.ee.solve_step(waypoint_pos, waypoint_quat)
                    _step("move_to")

        def move_ee(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, steps=120) -> None:
            with RUNTIME.lock:
                cur_pos, cur_quat = RUNTIME.ee.ee_pose()
            target_pos = cur_pos + np.array([dx, dy, dz], dtype=float)
            dquat = quat_from_euler(droll, dpitch, dyaw)
            target_quat = quat_mul(dquat, cur_quat)
            target_quat /= np.linalg.norm(target_quat)
            move_to(target_pos, target_quat, int(steps))

        def gripper_control(value: float, delay: int = 50) -> None:
            with RUNTIME.lock:
                RUNTIME.ee.set_gripper(value)
            for _ in range(max(1, int(delay))):
                _step("gripper")

        def get_object_pose(object_name: str) -> dict[str, Any]:
            with RUNTIME.lock:
                model = RUNTIME.model
                data = RUNTIME.data

                if model is None or data is None:
                    return {
                        "found": False,
                        "error": "Runtime not initialized",
                    }

                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)

                if body_id >= 0:
                    pos = data.xpos[body_id].copy().tolist()
                    quat = data.xquat[body_id].copy().tolist()
                    return {
                        "found": True,
                        "name": object_name,
                        "pos": pos,
                        "quat": quat,
                    }

                return {
                    "found": False,
                    "error": f"Object '{object_name}' not found in the scene",
                }

        api = {
            "np": np,
            "ee_pose": ee_pose,
            "move_to": move_to,
            "move_ee": move_ee,
            "gripper_control": gripper_control,
            "set_gripper": lambda value: gripper_control(value, delay=1),
            "sleep": time.sleep,
            "print": print,
            "get_object_pose": get_object_pose,
        }
        safe_builtins = {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "abs": abs,
            "float": float,
            "int": int,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "bool": bool,
            "enumerate": enumerate,
            "zip": zip,
            "round": round,
            "sum": sum,
            "isinstance": isinstance,
            "getattr": getattr,
            "hasattr": hasattr,
            "setattr": setattr,
            "__build_class__": __build_class__,
            "__name__": "__atomic__",
        }
        exec_globals = {"__builtins__": safe_builtins}
        exec_globals.update(api)

        atomic_ops_source = _load_atomic_ops_source()
        if atomic_ops_source:
            exec(compile(atomic_ops_source, str(ATOMIC_OPS_PATH), "exec"), exec_globals)

        exec(code, exec_globals)

        video_path = recorder.close()
        if video_path:
            results["video_path"] = str(video_path)

        state_action_path = output_dir / "state_action.json"
        state_action_path.write_text(
            json.dumps(
                {
                    "task_prompt": task_prompt or "N/A",
                    "records": recorder.records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        results["state_action_path"] = str(state_action_path)
        results["success"] = True
        _log("Code execution completed successfully.")

    except Exception as exc:
        results["error"] = str(exc)
        _log(f"Code execution failed: {exc}", level="ERROR")
        import traceback

        _log(traceback.format_exc(), level="ERROR")

    finally:
        with RUNTIME.lock:
            RUNTIME._busy = False

    return results


def _viewer_thread() -> None:
    """Run the native MuJoCo passive viewer for the current runtime scene."""
    global RUNTIME
    model = None
    data = None
    with RUNTIME.lock:
        model = RUNTIME.model
        data = RUNTIME.data
    if model is None or data is None:
        RUNTIME._viewer_ready.set()
        return

    try:
        cfg = TeleopConfig()

        def key_callback(key: int) -> None:
            with _TELEOP_LOCK:
                state = _TELEOP_STATE
                listening = _KEYBOARD_LISTENING
            if not listening or state is None:
                return

            if key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
                with _TELEOP_LOCK:
                    if _TELEOP_STATE is not None:
                        _TELEOP_STATE.request_ctrl_latch(0.40)
                return
            if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                with _TELEOP_LOCK:
                    if _TELEOP_STATE is not None:
                        _TELEOP_STATE.request_toggle_gripper()
                return
            if key == glfw.KEY_BACKSPACE:
                with _TELEOP_LOCK:
                    if _TELEOP_STATE is not None:
                        _TELEOP_STATE.request_reset_target()
                return

            if state.is_ctrl_latched():
                dp, rv = _key_to_ctrl_motion_delta(key, cfg)
            else:
                dp, rv = _key_to_motion_delta(key, cfg)
            if float(np.linalg.norm(dp)) > 0.0 or float(np.linalg.norm(rv)) > 0.0:
                with _TELEOP_LOCK:
                    if _TELEOP_STATE is not None:
                        _TELEOP_STATE.request_motion(dp, rv)

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            RUNTIME._viewer_ready.set()
            while viewer.is_running() and not _VIEWER_STOP_EVENT.is_set():
                with RUNTIME.lock:
                    with _TELEOP_LOCK:
                        state = _TELEOP_STATE
                        listening = _KEYBOARD_LISTENING

                    if listening and state is not None and RUNTIME.ee is not None:
                        with _TELEOP_LOCK:
                            state = _TELEOP_STATE
                            if state is not None:
                                dp, rv, tg, reset = state.pop_pending()
                            else:
                                dp = np.zeros(3, dtype=float)
                                rv = np.zeros(3, dtype=float)
                                tg = False
                                reset = False

                        if reset:
                            cur_pos, cur_quat = RUNTIME.ee.ee_pose()
                            with _TELEOP_LOCK:
                                if _TELEOP_STATE is not None:
                                    _TELEOP_STATE.target_pos = cur_pos.copy()
                                    _TELEOP_STATE.target_quat = cur_quat.copy()

                        if tg:
                            with _TELEOP_LOCK:
                                if _TELEOP_STATE is not None:
                                    _TELEOP_STATE.gripper_closed = not _TELEOP_STATE.gripper_closed
                                    gripper_closed = _TELEOP_STATE.gripper_closed
                                else:
                                    gripper_closed = False
                            grip_val = cfg.gripper_close if gripper_closed else cfg.gripper_open
                            RUNTIME.ee.set_gripper(grip_val)

                        with _TELEOP_LOCK:
                            state_now = _TELEOP_STATE
                            if state_now is not None:
                                _apply_keyboard_delta(state_now, dp, rv, cfg)
                                target_pos = state_now.target_pos.copy()
                                target_quat = state_now.target_quat.copy()
                            else:
                                target_pos = None
                                target_quat = None

                        if target_pos is not None and target_quat is not None:
                            for _ in range(cfg.solve_substeps):
                                RUNTIME.ee.solve_step(target_pos, target_quat)
                            for _ in range(cfg.sim_steps_per_frame):
                                mujoco.mj_step(RUNTIME.model, RUNTIME.data)

                    viewer.sync()
                time.sleep(1.0 / 60.0)
    except Exception as exc:
        _log(f"Viewer failed: {exc}", level="ERROR")
    finally:
        RUNTIME._viewer_ready.set()
