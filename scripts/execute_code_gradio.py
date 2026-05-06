import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import mujoco
import execute_mujoco as teleop_scene
import execute_code_runtime as exec_cli

try:
    mujoco.mj_loadPluginLibrary("./libmjlab.so.3.3.0")
except Exception:
    pass

# User code runs via exec_cli.execute_code_with_recording, which pre-registers
# primitives (move_to, move_ee, gripper_control, ee_pose, np) plus the
# composite library from scripts/evoma_atomic_ops.py (pick_and_place, push, pull,
# press, open, close, pour, move_x/y/z, rotate_x/y/z, get_object_abs_pose,
# recover_grasp_pose_from_offset). Same injection exists in generate_cli.execute_code_with_recording.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
LOG_ROOT = PROJECT_ROOT / "logs"
MANUAL_CODE_ROOT = LOG_ROOT / "manual_code"
DEFAULT_SCENE = "chemistry.json"
GRASP_OFFSETS_PATH = PROJECT_ROOT / "grasp_offsets.json"
DEFAULT_CODE = (
    "# Primitives: move_to(pos, quat_wxyz, num_steps), move_ee(dx,dy,dz,droll,dpitch,dyaw,steps),\n"
    "#            gripper_control(value, delay), ee_pose() -> (pos, quat_wxyz), np\n"
    "# Quaternion for move_to / ee_pose is (w, x, y, z).\n"
    "# Pre-registered composite APIs (evoma_atomic_ops.py — call directly, do not redefine):\n"
    "#   pick_and_place, push, pull, press, open, close, pour,\n"
    "#   move_x, move_y, move_z, rotate_x, rotate_y, rotate_z,\n"
    "#   get_object_abs_pose(object_poses, object_name),\n"
    "#   recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg)\n"
    "\n"
    "move_to(pos=[0.555, 0.0, 0.9449], quat=[0.0, 1.0, 0.0, 0.0], num_steps=100)\n"
)


class AppRuntime:
    def __init__(self) -> None:
        self.initialized = False
        self.scene_path: Path | None = None
        self.state_lock = threading.Lock()

    def ensure_runtime(self) -> None:
        with self.state_lock:
            if self.initialized:
                return
            exec_cli.RUNTIME = exec_cli.CliRuntime()
            self.initialized = True

    def load_scene(self, scene_path: Path) -> None:
        self.ensure_runtime()
        exec_cli.stop_viewer()
        _load_scene_any(scene_path)
        self.scene_path = scene_path
        exec_cli.start_viewer()

    def reload_scene(self) -> None:
        if self.scene_path is None:
            raise RuntimeError("No scene has been loaded yet.")
        self.load_scene(self.scene_path)


APP = AppRuntime()


def _resolve_input_path(file_obj, path_text: str | None) -> Path | None:
    if file_obj is not None:
        candidate = getattr(file_obj, "name", None) or str(file_obj)
        if candidate:
            return Path(candidate)
    text = (path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _append_log(logs: str, line: str) -> str:
    if not line:
        return logs or ""
    return f"{logs}\n{line}" if logs else line


def _run_with_captured_logs(fn, *args, **kwargs) -> tuple[Any, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue().strip()


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n2 = float(np.dot(q, q))
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / n2


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=float)
    out = _quat_mul(_quat_mul(q, qv), _quat_inv(q))
    return out[1:]


def _quat_to_euler_rpy_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.pi / 2.0 * np.sign(sinp) if abs(float(sinp)) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def _wrap_deg180(angles_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(angles_deg, dtype=float)
    return (vals + 180.0) % 360.0 - 180.0


def _sanitize_code_filename(filename: str) -> str:
    """Keep filename simple and safe; disallow path traversal and odd characters."""
    text = (filename or "").strip()
    text = Path(text).name
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        raise ValueError("Please provide a valid file name.")
    if not text.endswith(".py"):
        text = f"{text}.py"
    return text


def _save_manual_code(code_text: str, filename: str, logs: str) -> tuple[str, str]:
    if not code_text or not code_text.strip():
        logs = _append_log(logs, "[Save] Code snippet is empty, nothing was saved.")
        return logs, ""

    try:
        safe_name = _sanitize_code_filename(filename)
        MANUAL_CODE_ROOT.mkdir(parents=True, exist_ok=True)
        save_path = MANUAL_CODE_ROOT / safe_name
        save_path.write_text(code_text, encoding="utf-8")
        logs = _append_log(logs, f"[Save] Code saved to: {save_path}")
        return logs, str(save_path)
    except Exception as exc:
        logs = _append_log(logs, f"[Save] Failed: {exc}")
        return logs, ""


def _extract_scene_pose_info(scene_path: Path | None = None) -> dict[str, Any]:
    """Read current object poses from runtime MuJoCo state (not static source JSON)."""
    info: dict[str, Any] = {
        "scene_path": str(scene_path) if scene_path is not None else "",
        "objects": [],
    }

    runtime = getattr(exec_cli, "RUNTIME", None)
    if runtime is None:
        info["note"] = "Runtime is not initialized yet."
        return info

    with runtime.lock:
        model = runtime.model
        data = runtime.data
        if model is None or data is None:
            info["note"] = "Scene is not loaded yet."
            return info

        scene_json = runtime.scene_json_path
        scene_xml = runtime.scene_xml_path
        if scene_json is not None:
            info["scene_path"] = str(scene_json)
        elif scene_xml is not None:
            info["scene_path"] = str(scene_xml)

        metadata_by_name: dict[str, dict[str, Any]] = {}
        for idx, asset in enumerate(runtime.scene_assets if isinstance(runtime.scene_assets, list) else []):
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).strip()
            if not name:
                continue
            metadata_by_name[name] = {
                "index": idx,
                "key": asset.get("key", ""),
            }

        added_names: set[str] = set()
        for name, meta in metadata_by_name.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                continue
            pos = [float(x) for x in data.xpos[body_id].tolist()]
            quat = [float(x) for x in data.xquat[body_id].tolist()]
            info["objects"].append(
                {
                    "index": meta["index"],
                    "name": name,
                    "key": meta["key"],
                    "pos": pos,
                    "quat": quat,
                }
            )
            added_names.add(name)

        # Fallback for XML-only or assets without metadata: include all free-joint bodies.
        if not info["objects"]:
            for body_id in range(1, int(model.nbody)):
                body_jnt_num = int(model.body_jntnum[body_id])
                body_jnt_adr = int(model.body_jntadr[body_id])
                has_free_joint = False
                for k in range(body_jnt_num):
                    jnt_id = body_jnt_adr + k
                    if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                        has_free_joint = True
                        break
                if not has_free_joint:
                    continue

                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if not body_name or body_name in added_names:
                    continue
                info["objects"].append(
                    {
                        "index": len(info["objects"]),
                        "name": str(body_name),
                        "key": "",
                        "pos": [float(x) for x in data.xpos[body_id].tolist()],
                        "quat": [float(x) for x in data.xquat[body_id].tolist()],
                    }
                )

        if runtime.ee is not None:
            try:
                ee_pos, ee_quat = runtime.ee.ee_pose()
                info["robot_arm"] = {
                    "ee_pos": [float(x) for x in ee_pos.tolist()],
                    "ee_quat": [float(x) for x in ee_quat.tolist()],
                }
            except Exception:
                pass

        info["object_count"] = len(info["objects"])
    return info


def _poll_pose_info() -> dict[str, Any]:
    """Periodic callback used by UI timer to keep pose panel in sync with runtime."""
    return _extract_scene_pose_info(APP.scene_path)


def _object_choices_from_pose_info(pose_info: dict[str, Any] | None) -> list[str]:
    choices: list[str] = []
    if not isinstance(pose_info, dict):
        return choices
    for obj in pose_info.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        key = str(obj.get("key", "")).strip()
        name = str(obj.get("name", "")).strip()
        if key:
            label = f"{key} | {name}" if name else key
        elif name:
            label = name
        else:
            continue
        choices.append(label)
    return choices


def _extract_selected_object(selected_object: str) -> tuple[str, str]:
    text = (selected_object or "").strip()
    if not text:
        return "", ""
    if " | " in text:
        key, name = text.split(" | ", 1)
        return key.strip(), name.strip()
    return text, text


def _sanitize_mj_name(text: str, fallback: str = "asset") -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", (text or "").strip())
    name = name.strip("._-")
    return name or fallback


def _resolve_native_asset_xml(asset_key: str) -> Path:
    category, _, leaf = asset_key.partition("/")
    asset_name = leaf or category
    candidate = MODEL_ROOT / category / f"{asset_name}.xml"
    if candidate.exists():
        return candidate
    fallback = MODEL_ROOT / category / f"{_sanitize_mj_name(asset_name)}.xml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot resolve asset XML for key: {asset_key}")


def _parse_root_body_name(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {xml_path}")
    body = worldbody.find("body")
    if body is None:
        raise ValueError(f"No <body> found in {xml_path}")
    body_name = body.get("name")
    if not body_name:
        raise ValueError(f"Missing root body name in {xml_path}")
    return body_name


def _load_build_scene_native(scene_json_path: Path, payload: dict[str, Any]) -> None:
    """Load a build-scene JSON using native skybox and desk/table assets.

    This keeps the asset layout and robot placement from the JSON file, but
    omits the custom background and table texture overlays used by data generation.
    """
    runtime = exec_cli.RUNTIME
    robot = payload.get("robot", {}) if isinstance(payload.get("robot"), dict) else {}
    camera = payload.get("camera", {}) if isinstance(payload.get("camera"), dict) else {}
    assets = payload.get("assets", []) if isinstance(payload.get("assets"), list) else []

    robot_base_pos = np.asarray(robot.get("base_pos", [0.0, 0.0, 0.824]), dtype=float)
    robot_base_quat = np.asarray(robot.get("base_quat", [0.0, 0.0, 0.0, -1.0]), dtype=float)
    robot_joint_targets = np.asarray(robot.get("joint_targets", [0.0] * 6), dtype=float)
    robot_gripper = float(robot.get("gripper", 0.0))
    camera_pos = np.asarray(camera.get("pos", [0.0, -1.4, 1.45]), dtype=float)
    camera_quat = np.asarray(camera.get("quat", [0.819, 0.574, 0.0, 0.0]), dtype=float)

    if robot_base_pos.shape != (3,):
        robot_base_pos = np.array([0.0, 0.0, 0.824], dtype=float)
    if robot_base_quat.shape != (4,) or np.linalg.norm(robot_base_quat) < 1e-8:
        robot_base_quat = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)
    robot_base_quat = robot_base_quat / np.linalg.norm(robot_base_quat)

    if robot_joint_targets.shape != (6,):
        robot_joint_targets = np.zeros(6, dtype=float)

    if camera_pos.shape != (3,):
        camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)
    if camera_quat.shape != (4,) or np.linalg.norm(camera_quat) < 1e-8:
        camera_quat = np.array([0.819, 0.574, 0.0, 0.0], dtype=float)
    camera_quat = camera_quat / np.linalg.norm(camera_quat)

    runtime_dir = LOG_ROOT / "gradio_native_scene_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_xml = runtime_dir / "native_runtime_scene.xml"

    build_items: list[dict[str, Any]] = []
    used_instance_names: set[str] = set()
    for idx, entry in enumerate(assets, start=1):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", "")).strip()
        if not key:
            continue
        preferred_name = str(entry.get("name", "")).strip() or Path(key).name
        instance_name_base = _sanitize_mj_name(preferred_name, fallback="asset")
        instance_name = instance_name_base
        suffix = 2
        while instance_name in used_instance_names:
            instance_name = f"{instance_name_base}_{suffix}"
            suffix += 1
        used_instance_names.add(instance_name)

        pos = np.asarray(entry.get("pos", [0.0, 0.0, 0.845]), dtype=float)
        quat = np.asarray(entry.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        scale = np.asarray(entry.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        if pos.shape != (3,):
            pos = np.array([0.0, 0.0, 0.845], dtype=float)
        if quat.shape != (4,) or np.linalg.norm(quat) < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        quat = quat / np.linalg.norm(quat)
        if scale.shape != (3,):
            scale = np.array([1.0, 1.0, 1.0], dtype=float)
        scale = np.clip(scale.astype(float), 0.05, 50.0)
        build_items.append(
            {
                "id": idx,
                "key": key,
                "instance_name": instance_name,
                "pos": pos,
                "quat": quat,
                "scale": scale,
            }
        )

    base_pos_text = " ".join(f"{x:.6f}" for x in robot_base_pos)
    base_quat_text = " ".join(f"{x:.6f}" for x in robot_base_quat)
    cam_pos_text = " ".join(f"{x:.6f}" for x in camera_pos)
    cam_quat_text = " ".join(f"{x:.6f}" for x in camera_quat)

    lines = [
        '<mujoco model="builder_native">',
        '  <option integrator="implicitfast" impratio="10" cone="elliptic" noslip_iterations="2">',
        '    <flag multiccd="enable"/>',
        '  </option>',
        '  <visual>',
        '    <global azimuth="220" elevation="-30" offwidth="1280" offheight="960"/>',
        '  </visual>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>',
        f'    <model name="desk_model" file="{(MODEL_ROOT / "misc" / "simple_table.xml").as_posix()}" content_type="text/xml"/>',
        f'    <model name="ur5e_model" file="{(MODEL_ROOT / "robot" / "ur5e_gripper.xml").as_posix()}" content_type="text/xml"/>',
    ]

    for item in build_items:
        asset_xml = _resolve_native_asset_xml(item["key"])
        model_name = _sanitize_mj_name(f'{item["instance_name"]}_model', fallback="asset_model")
        lines.append(f'    <model name="{model_name}" file="{asset_xml.as_posix()}" content_type="text/xml"/>')

    lines.extend(
        [
            '  </asset>',
            '  <worldbody>',
            '    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>',
            '    <geom name="floor" pos="0 0 0" size="2.5 2.5 0.05" type="plane" material="groundplane"/>',
            '    <body name="desk" pos="0 0 0" quat="1 0 0 1">',
            '      <attach model="desk_model" body="vention table" prefix="desk/"/>',
            f'      <camera name="table_cam_front" pos="{cam_pos_text}" quat="{cam_quat_text}" fovy="45" resolution="1280 960"/>',
            '    </body>',
            f'    <body name="ur5e_center" pos="{base_pos_text}" quat="{base_quat_text}">',
            '      <attach model="ur5e_model" body="world" prefix="/ur:"/>',
            '    </body>',
        ]
    )

    for item in build_items:
        asset_xml = _resolve_native_asset_xml(item["key"])
        root_body_name = _parse_root_body_name(asset_xml)
        pos_text = " ".join(f"{x:.6f}" for x in item["pos"])
        quat_text = " ".join(f"{x:.6f}" for x in item["quat"])
        body_name = item["instance_name"]
        joint_name = _sanitize_mj_name(f"{body_name}_joint", fallback="asset_joint")
        model_name = _sanitize_mj_name(f"{body_name}_model", fallback="asset_model")
        lines.extend(
            [
                f'    <body name="{body_name}" pos="{pos_text}" quat="{quat_text}">',
                f'      <joint name="{joint_name}" type="free"/>',
                f'      <attach model="{model_name}" body="{root_body_name}" prefix="{body_name}/"/>',
                '    </body>',
            ]
        )

    lines.extend(['  </worldbody>', '</mujoco>'])
    runtime_xml.write_text("\n".join(lines), encoding="utf-8")

    with runtime.lock:
        if runtime.model is not None and runtime.scene_xml_path is not None:
            runtime._cleanup_cached_build_runtime_xml()
        model = mujoco.MjModel.from_xml_path(str(runtime_xml))
        data = mujoco.MjData(model)
        ee = exec_cli.EEController(model, data)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        try:
            data.qpos[ee.jnt_span] = robot_joint_targets
            data.ctrl[ee.act_span] = robot_joint_targets
            data.ctrl[ee.gripper_act_id] = float(np.clip(robot_gripper, 0.0, 255.0))
        except Exception:
            pass
        mujoco.mj_forward(model, data)

        runtime.model = model
        runtime.data = data
        runtime.ee = ee
        runtime.scene_json_path = scene_json_path
        runtime.scene_xml_path = runtime_xml
        runtime.scene_assets = assets

    exec_cli._log(f"Scene loaded with native background/table: {scene_json_path}")


def _load_scene_any(scene_path: Path) -> None:
    suffix = scene_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        payload_format = str(payload.get("format", "")).strip()
        if payload_format in {"evobody_build_scene_v1", "evobody_build_scene_v2", "evobody_manual_scene_v1"} or (
            not payload_format and isinstance(payload.get("assets"), list) and not payload.get("scene_xml")
        ):
            runtime_scene = teleop_scene._build_runtime_from_scene(scene_path)
            with exec_cli.RUNTIME.lock:
                old_scene_xml = exec_cli.RUNTIME.scene_xml_path
                if old_scene_xml is not None:
                    old_parent = old_scene_xml.parent
                    if old_parent.name.startswith("execute_mujoco_runtime_") and old_parent.exists():
                        try:
                            shutil.rmtree(old_parent)
                        except Exception:
                            pass

                exec_cli.RUNTIME.model = runtime_scene.model
                exec_cli.RUNTIME.data = runtime_scene.data
                exec_cli.RUNTIME.ee = runtime_scene.ee
                exec_cli.RUNTIME.scene_json_path = scene_path
                exec_cli.RUNTIME.scene_xml_path = runtime_scene.runtime_xml
                exec_cli.RUNTIME.scene_assets = payload.get("assets", []) if isinstance(payload.get("assets"), list) else []

            exec_cli._log(f"Scene loaded via execute_mujoco builder: {scene_path}")
        else:
            exec_cli._safe_load_scene(scene_path, False)
        return

    if suffix != ".xml":
        raise ValueError("Only .json and .xml scene files are supported")

    with exec_cli.RUNTIME.lock:
        if exec_cli.RUNTIME.model is not None and exec_cli.RUNTIME.scene_xml_path is not None:
            exec_cli.RUNTIME._cleanup_cached_build_runtime_xml()

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        try:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        except Exception:
            pass
        ee = exec_cli.EEController(model, data)
        mujoco.mj_forward(model, data)

        exec_cli.RUNTIME.model = model
        exec_cli.RUNTIME.data = data
        exec_cli.RUNTIME.ee = ee
        exec_cli.RUNTIME.scene_json_path = None
        exec_cli.RUNTIME.scene_xml_path = scene_path
        exec_cli.RUNTIME.scene_assets = []

    exec_cli._log(f"Scene loaded from XML: {scene_path}")


def _load_scene_from_path(scene_path: Path, logs: str) -> tuple[str, dict[str, Any] | None, gr.Dropdown]:
    try:
        APP.load_scene(scene_path)
        logs = _append_log(logs, f"[Scene] Loaded: {scene_path}")
        pose_info = _extract_scene_pose_info(scene_path)
        return logs, pose_info, gr.update(choices=_object_choices_from_pose_info(pose_info), value=None)
    except Exception as exc:
        logs = _append_log(logs, f"[Scene] Load failed: {exc}")
        return logs, None, gr.update(choices=[], value=None)


def _reload_scene(logs: str) -> tuple[str, dict[str, Any] | None, gr.Dropdown]:
    if APP.scene_path is None:
        logs = _append_log(logs, "[Scene] No scene has been loaded yet.")
        return logs, None, gr.update(choices=[], value=None)
    return _load_scene_from_path(APP.scene_path, logs)


def _listen_keyboard(logs: str) -> str:
    exec_cli.set_keyboard_listening(True)
    return _append_log(logs, "[Keyboard] Listening enabled. MuJoCo window now accepts teleop keys.")


def _cancel_listen_keyboard(logs: str) -> str:
    exec_cli.set_keyboard_listening(False)
    return _append_log(logs, "[Keyboard] Listening disabled.")


def _execute_code(
    code_text: str,
    task_prompt: str,
    logs: str,
) -> tuple[str, dict[str, Any] | None, gr.Dropdown]:
    APP.ensure_runtime()
    if APP.scene_path is None:
        logs = _append_log(logs, "[Exec] No scene loaded. Please start with a scene path.")
        return logs, None, gr.update(choices=[], value=None)

    if not code_text or not code_text.strip():
        logs = _append_log(logs, "[Exec] Please input code snippet first.")
        return logs, None, gr.update(choices=[], value=None)

    try:
        results, captured = _run_with_captured_logs(
            exec_cli.execute_code_with_recording,
            code_text,
            task_prompt=task_prompt.strip() if task_prompt else "UI debug task",
            save_video=False,
            output_dir=LOG_ROOT / "execution_results" / f"ui_run_{threading.get_native_id()}_{int(time.time())}",
        )

        if captured:
            logs = _append_log(logs, captured)

        if not results.get("success", False):
            logs = _append_log(logs, f"[Exec] Failed: {results.get('error', 'unknown error')}")
            return logs, _extract_scene_pose_info(APP.scene_path), gr.update(choices=_object_choices_from_pose_info(_extract_scene_pose_info(APP.scene_path)), value=None)
        pose_info = _extract_scene_pose_info(APP.scene_path)

        logs = _append_log(logs, "[Exec] Finished successfully.")
        return logs, pose_info, gr.update(choices=_object_choices_from_pose_info(pose_info), value=None)
    except Exception as exc:
        logs = _append_log(logs, f"[Exec] Exception: {exc}")
        return logs, _extract_scene_pose_info(APP.scene_path), gr.update(choices=_object_choices_from_pose_info(_extract_scene_pose_info(APP.scene_path)), value=None)


def _generate_video(
    code_text: str,
    task_prompt: str,
    logs: str,
) -> tuple[str, dict[str, Any] | None, gr.Dropdown, str]:
    APP.ensure_runtime()
    if APP.scene_path is None:
        logs = _append_log(logs, "[Video] No scene loaded. Please start with a scene path.")
        return logs, None, gr.update(choices=[], value=None), ""

    if not code_text or not code_text.strip():
        logs = _append_log(logs, "[Video] Please input code snippet first.")
        return logs, None, gr.update(choices=[], value=None), ""

    try:
        output_dir = LOG_ROOT / "execution_results" / f"ui_video_{threading.get_native_id()}_{int(time.time())}"
        results, captured = _run_with_captured_logs(
            exec_cli.execute_code_with_recording,
            code_text,
            task_prompt=task_prompt.strip() if task_prompt else "UI debug task",
            save_video=True,
            output_dir=output_dir,
        )

        if captured:
            logs = _append_log(logs, captured)

        pose_info = _extract_scene_pose_info(APP.scene_path)
        record_choices = gr.update(choices=_object_choices_from_pose_info(pose_info), value=None)

        if not results.get("success", False):
            logs = _append_log(logs, f"[Video] Failed: {results.get('error', 'unknown error')}")
            return logs, pose_info, record_choices, ""

        video_path = str(results.get("video_path") or "")
        if video_path:
            logs = _append_log(logs, f"[Video] Saved: {video_path}")
        else:
            logs = _append_log(logs, "[Video] Execution succeeded, but no video file was generated.")
        return logs, pose_info, record_choices, video_path
    except Exception as exc:
        logs = _append_log(logs, f"[Video] Exception: {exc}")
        pose_info = _extract_scene_pose_info(APP.scene_path)
        return logs, pose_info, gr.update(choices=_object_choices_from_pose_info(pose_info), value=None), ""


def _record_pose(selected_object: str, pose_name: str, logs: str) -> tuple[str, dict[str, Any] | None]:
    APP.ensure_runtime()
    key, body_name = _extract_selected_object(selected_object)
    pose_name = (pose_name or "").strip()
    if not key and not body_name:
        return _append_log(logs, "[Record] Please select an object first."), None
    if not pose_name:
        return _append_log(logs, "[Record] Please input pose name first."), None

    runtime = exec_cli.RUNTIME
    try:
        with runtime.lock:
            model = runtime.model
            data = runtime.data
            ee = runtime.ee
            if model is None or data is None or ee is None:
                raise RuntimeError("Runtime scene is not ready.")
            mujoco.mj_forward(model, data)
            ee_pos, ee_quat = ee.ee_pose()
            ee_pos = np.asarray(ee_pos, dtype=float)
            ee_quat = np.asarray(ee_quat, dtype=float)
            ee_quat = ee_quat / max(float(np.linalg.norm(ee_quat)), 1e-12)

            body_id = -1
            if body_name:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0 and key:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, key.split("/")[-1])
            if body_id < 0:
                raise ValueError(f"Cannot find body for selected object: {selected_object}")

            obj_pos = np.asarray(data.xpos[body_id], dtype=float)
            obj_quat = np.asarray(data.xquat[body_id], dtype=float)
            obj_quat = obj_quat / max(float(np.linalg.norm(obj_quat)), 1e-12)
            delta_pos = ee_pos - obj_pos
            ee_rpy = _quat_to_euler_rpy_deg(ee_quat)
            obj_rpy = _quat_to_euler_rpy_deg(obj_quat)
            delta_rpy = _wrap_deg180(ee_rpy - obj_rpy)

        payload: dict[str, Any] = {}
        if GRASP_OFFSETS_PATH.exists():
            try:
                payload = json.loads(GRASP_OFFSETS_PATH.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        object_key = key or body_name
        obj_node = payload.get(object_key)
        if not isinstance(obj_node, dict):
            obj_node = {}
            payload[object_key] = obj_node
        obj_node[pose_name] = {
            "pos": [round(float(v), 6) for v in delta_pos.tolist()],
            "3d_rotation": [round(float(v), 6) for v in delta_rpy.tolist()],
        }
        GRASP_OFFSETS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        logs = _append_log(
            logs,
            f"[Record] Saved {object_key}/{pose_name} to {GRASP_OFFSETS_PATH} "
            f"pos={obj_node[pose_name]['pos']} rpy={obj_node[pose_name]['3d_rotation']}",
        )
        return logs, _extract_scene_pose_info(APP.scene_path)
    except Exception as exc:
        return _append_log(logs, f"[Record] Failed: {exc}"), None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EvoBody Scene Debugger") as demo:
        gr.Markdown("## EvoBody Scene Debugger")
        gr.Markdown(
            "Load a scene once from the command line, run custom action code, and watch motion in the native MuJoCo viewer. "
            "Execution uses the same namespace as `execute_code_runtime`: primitives plus pre-registered "
            "`evoma_atomic_ops` (`pick_and_place`, `push`, `pull`, `press`, `open`, `close`, `pour`, "
            "`move_x`/`move_y`/`move_z`, `rotate_x`/`rotate_y`/`rotate_z`, "
            "`get_object_abs_pose`, `recover_grasp_pose_from_offset`)."
        )

        with gr.Row():
            with gr.Column(scale=5):
                task_prompt = gr.Textbox(value="UI debug task", label="Task prompt", lines=2)
                save_filename = gr.Textbox(value="manual_action", label="Save filename")
                code_input = gr.Code(value=DEFAULT_CODE, language="python", label="Code snippet")

                with gr.Row():
                    reload_btn = gr.Button("Reload", variant="secondary")
                    run_btn = gr.Button("Execute", variant="primary")
                    video_btn = gr.Button("Generate Video", variant="primary")
                    listen_btn = gr.Button("Listen", variant="secondary")
                    cancel_listen_btn = gr.Button("Cancel Listening", variant="secondary")
                    refresh_pose_btn = gr.Button("Refresh Poses")
                    save_code_btn = gr.Button("Save Code")
                gr.Markdown("### Grasp Offsets Recorder")
                record_object = gr.Dropdown(choices=[], label="Select object", interactive=True)
                record_pose_name = gr.Textbox(value="", label="Pose Name", placeholder="e.g. beaker_edge")
                record_pose_btn = gr.Button("record_pose", variant="secondary")

            with gr.Column(scale=5):
                logs = gr.Textbox(label="Execution logs", lines=20, interactive=False)
                video_path = gr.Textbox(label="Generated video path", interactive=False)
                pose_info = gr.JSON(label="Live object poses (runtime)")
                saved_code_path = gr.Textbox(label="Saved code path", interactive=False)

            pose_timer = gr.Timer(value=0.5)

        reload_btn.click(
            _reload_scene,
            inputs=[logs],
            outputs=[logs, pose_info, record_object],
        )
        run_btn.click(
            _execute_code,
            inputs=[code_input, task_prompt, logs],
            outputs=[logs, pose_info, record_object],
        )
        video_btn.click(
            _generate_video,
            inputs=[code_input, task_prompt, logs],
            outputs=[logs, pose_info, record_object, video_path],
        )
        listen_btn.click(
            _listen_keyboard,
            inputs=[logs],
            outputs=[logs],
        )
        cancel_listen_btn.click(
            _cancel_listen_keyboard,
            inputs=[logs],
            outputs=[logs],
        )
        refresh_pose_btn.click(
            _poll_pose_info,
            inputs=[],
            outputs=[pose_info],
        )
        save_code_btn.click(
            _save_manual_code,
            inputs=[code_input, save_filename, logs],
            outputs=[logs, saved_code_path],
        )
        record_pose_btn.click(
            _record_pose,
            inputs=[record_object, record_pose_name, logs],
            outputs=[logs, pose_info],
        )
        pose_timer.tick(
            _poll_pose_info,
            inputs=[],
            outputs=[pose_info],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gradio scene debugger with a native MuJoCo viewer")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE, help="Path to the scene JSON or XML file")
    args = parser.parse_args()

    scene_path = _resolve_input_path(None, args.scene)
    if scene_path is None or not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {args.scene}")

    APP.ensure_runtime()
    APP.load_scene(scene_path)

    app = build_app()
    app.queue()
    app.launch(server_name="localhost", server_port=7863, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
