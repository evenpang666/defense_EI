"""Interactive MuJoCo teleoperation with keyboard Cartesian control.

Features:
- Load default chemistry scene (`chemistry.json`, evobody_build_scene_v2).
- Use MuJoCo explicit viewer window.
- Use keyboard to move/rotate end-effector target.
- Toggle gripper with keyboard.

Example:
    python scripts/execute_mujoco.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import mujoco
import mujoco.viewer
import numpy as np
import glfw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_SCENE_JSON = PROJECT_ROOT / "chemistry.json"
GRASP_OFFSETS_PATH = PROJECT_ROOT / "grasp_offsets.json"


def _log(msg: str) -> None:
    print(f"[execute_mujoco] {msg}", flush=True)


_MJLAB_PLUGIN_AVAILABLE: bool | None = None


def _try_load_optional_mjlab_plugin() -> bool:
    """Try loading optional mjlab plugin across MuJoCo versions/platforms."""
    loader = getattr(mujoco, "mj_loadPluginLibrary", None)
    if not callable(loader):
        return False

    candidates: list[str] = []
    env_candidate = str(os.environ.get("MJLAB_PLUGIN_PATH", "")).strip()
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend([
        "./libmjlab.so.3.3.0",
        "./mjlab.dll",
        "./libmjlab.dylib",
    ])

    for plugin_path in candidates:
        try:
            loader(plugin_path)
            _log(f"Loaded plugin: {plugin_path}")
            return True
        except Exception:
            continue

    _log("Optional mjlab plugin not found; continuing without it.")
    return False


def _ensure_optional_mjlab_plugin() -> bool:
    global _MJLAB_PLUGIN_AVAILABLE
    if _MJLAB_PLUGIN_AVAILABLE is None:
        _MJLAB_PLUGIN_AVAILABLE = _try_load_optional_mjlab_plugin()
    return _MJLAB_PLUGIN_AVAILABLE


def _mj(name: str):
    fn = getattr(mujoco, name, None)
    if fn is None:
        raise RuntimeError(f"Missing MuJoCo API: {name}")
    return fn


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(out))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return out / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_to_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_err = quat_mul(target, quat_conj(current))
    if q_err[0] < 0:
        q_err = -q_err
    vec = q_err[1:]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-8:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(vec_norm, q_err[0])
    return vec / vec_norm * angle


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    half = 0.5 * float(angle)
    s = math.sin(half)
    return normalize_quat_wxyz(np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float))


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), roll)
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=float), pitch)
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=float), yaw)
    return normalize_quat_wxyz(quat_mul(qz, quat_mul(qy, qx)))


def quat_from_mat33_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz(np.array([w, x, y, z], dtype=float))


def _quat_to_euler_rpy_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(np.asarray(q, dtype=float))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(float(sinp)) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def _wrap_deg180(angles_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(angles_deg, dtype=float)
    return (vals + 180.0) % 360.0 - 180.0


def _object_key_from_body_name(body_name: str) -> str:
    text = str(body_name or "").strip()
    if not text:
        return ""
    if text.startswith("/"):
        text = text[1:]
    return text.split("/", 1)[0]


def _save_grasp_offset_for_selected_body(model: Any, data: Any, ee: "EEController", selected_body_id: int) -> str:
    body_id = int(selected_body_id)
    if body_id <= 0:
        return "[Record] No object selected. Double-click an object in viewer first."

    try:
        body_name = str(model.body(body_id).name)
    except Exception:
        return f"[Record] Invalid selected body id: {body_id}"

    object_key = _object_key_from_body_name(body_name)
    if not object_key:
        return "[Record] Failed to resolve selected object name."

    _mj("mj_forward")(model, data)
    ee_pos, ee_quat = ee.ee_pose()
    ee_pos = np.asarray(ee_pos, dtype=float)
    ee_quat = normalize_quat_wxyz(np.asarray(ee_quat, dtype=float))

    obj_pos = np.asarray(data.xpos[body_id], dtype=float)
    obj_quat = normalize_quat_wxyz(np.asarray(data.xquat[body_id], dtype=float))

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

    obj_node = payload.get(object_key)
    if not isinstance(obj_node, dict):
        obj_node = {}
        payload[object_key] = obj_node

    base_pose_name = time.strftime("teleop_%Y%m%d_%H%M%S")
    pose_name = base_pose_name
    suffix = 2
    while pose_name in obj_node:
        pose_name = f"{base_pose_name}_{suffix}"
        suffix += 1

    obj_node[pose_name] = {
        "pos": [round(float(v), 6) for v in delta_pos.tolist()],
        "3d_rotation": [round(float(v), 6) for v in delta_rpy.tolist()],
    }
    GRASP_OFFSETS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return (
        f"[Record] Saved {object_key}/{pose_name} from selected body {body_name} "
        f"to {GRASP_OFFSETS_PATH} pos={obj_node[pose_name]['pos']} "
        f"rpy={obj_node[pose_name]['3d_rotation']}"
    )


def _sanitize_mj_name(name: str, fallback: str = "asset") -> str:
    text = str(name or "").strip()
    if not text:
        return fallback
    out = []
    for c in text:
        if c.isalnum() or c in ("_", "/", "-", ":"):
            out.append(c)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_/:-")
    return cleaned or fallback


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return list(default)


def _format_float_list(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def _scale_numbers(values: list[float], scale: np.ndarray) -> list[float]:
    return [float(v) * float(scale[min(i, 2)]) for i, v in enumerate(values)]


def _scale_attr(elem: ET.Element, attr: str, scale: np.ndarray) -> None:
    if attr not in elem.attrib:
        return
    vals = _parse_float_list(elem.get(attr), [])
    if vals:
        elem.set(attr, _format_float_list(_scale_numbers(vals, scale)))


def _scale_geom_size(elem: ET.Element, scale: np.ndarray) -> None:
    vals = _parse_float_list(elem.get("size"), [])
    if not vals:
        return
    gtype = (elem.get("type") or "").lower()
    if gtype == "sphere" and len(vals) == 1:
        elem.set("size", f"{vals[0] * float(np.mean(scale)):.6f}")
        return
    if gtype in {"capsule", "cylinder"} and len(vals) == 2:
        rscale = float((scale[0] + scale[1]) / 2.0)
        elem.set("size", _format_float_list([vals[0] * rscale, vals[1] * scale[2]]))
        return
    elem.set("size", _format_float_list(_scale_numbers(vals, scale)))


def _transform_asset_tree_for_scale(root: ET.Element, scale: np.ndarray) -> None:
    for elem in root.iter():
        tag = elem.tag
        if tag in {"body", "geom", "site", "camera", "light", "inertial", "joint"}:
            _scale_attr(elem, "pos", scale)
        if tag == "geom":
            _scale_geom_size(elem, scale)
            _scale_attr(elem, "fromto", scale)
        elif tag == "site":
            _scale_attr(elem, "size", scale)
            _scale_attr(elem, "fromto", scale)
        elif tag == "mesh":
            current = _parse_float_list(elem.get("scale"), [1.0, 1.0, 1.0])
            if len(current) == 1:
                current = current * 3
            if len(current) == 2:
                current = [current[0], current[1], 1.0]
            elem.set("scale", _format_float_list(np.asarray(current[:3], dtype=float) * scale))


def _absolutize_xml_file_refs(root: ET.Element, source_dir: Path) -> None:
    asset_base_dir = source_dir
    mesh_dir = source_dir
    texture_dir = source_dir

    compiler = root.find("compiler")
    if compiler is not None:
        assetdir = compiler.get("assetdir")
        if assetdir:
            asset_base_dir = (source_dir / assetdir).resolve() if not Path(assetdir).is_absolute() else Path(assetdir).resolve()
            compiler.set("assetdir", asset_base_dir.as_posix())
            mesh_dir = asset_base_dir
            texture_dir = asset_base_dir
        meshdir = compiler.get("meshdir")
        if meshdir:
            mesh_dir = (asset_base_dir / meshdir).resolve() if not Path(meshdir).is_absolute() else Path(meshdir).resolve()
            compiler.set("meshdir", mesh_dir.as_posix())
        texturedir = compiler.get("texturedir")
        if texturedir:
            texture_dir = (asset_base_dir / texturedir).resolve() if not Path(texturedir).is_absolute() else Path(texturedir).resolve()
            compiler.set("texturedir", texture_dir.as_posix())

    for elem in root.iter():
        if elem.tag not in {"mesh", "texture", "include"}:
            continue
        file_attr = elem.get("file")
        if not file_attr or Path(file_attr).is_absolute():
            continue
        if elem.tag == "mesh":
            abs_path = (mesh_dir / file_attr).resolve()
        elif elem.tag == "texture":
            abs_path = (texture_dir / file_attr).resolve()
        else:
            abs_path = (source_dir / file_attr).resolve()
        elem.set("file", abs_path.as_posix())


def _strip_plugin_tags(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "plugin":
                parent.remove(child)


def _parse_root_body_name(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {xml_path}")
    bodies = worldbody.findall("body")
    if not bodies:
        raise ValueError(f"No <body> found in {xml_path}")
    names = [str(b.get("name")) for b in bodies if b.get("name")]
    if "world" in names:
        return "world"
    if names:
        return names[0]
    raise ValueError(f"Cannot infer root body in {xml_path}")


def _discover_build_assets() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((MODEL_ROOT / "object").glob("*.xml")):
        try:
            out[f"object/{path.stem}"] = {"xml_path": path, "root_body_name": _parse_root_body_name(path)}
        except Exception:
            continue
    for path in sorted((MODEL_ROOT / "instrument").glob("*.xml")):
        try:
            out[f"instrument/{path.stem}"] = {"xml_path": path, "root_body_name": _parse_root_body_name(path)}
        except Exception:
            continue
    return out


def _write_scaled_asset_xml(
    source_xml: Path,
    out_xml: Path,
    scale: np.ndarray,
    *,
    strip_plugins: bool = False,
) -> Path:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    if strip_plugins or not _ensure_optional_mjlab_plugin():
        _strip_plugin_tags(root)
    _transform_asset_tree_for_scale(root, scale)
    _absolutize_xml_file_refs(root, source_xml.parent)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    return out_xml


def _normalize_asset_xml_refs(source_xml: Path, out_xml: Path, *, strip_plugins: bool = False) -> Path:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    if strip_plugins or not _ensure_optional_mjlab_plugin():
        _strip_plugin_tags(root)
    _absolutize_xml_file_refs(root, source_xml.parent)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    return out_xml


class EEController:
    def __init__(self, model: Any, data: Any):
        self.model = model
        self.data = data
        self.site_id = model.site("/ur:2f85:pinch").id
        self.jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        self.dof_adr = model.joint("/ur:shoulder_pan").dofadr.item()
        self.act_id = model.actuator("/ur:shoulder_pan").id
        self.gripper_act_id = model.actuator("/ur:2f85:fingers_actuator").id
        self.jnt_span = slice(self.jnt_adr, self.jnt_adr + 6)
        self.dof_span = slice(self.dof_adr, self.dof_adr + 6)
        self.act_span = slice(self.act_id, self.act_id + 6)

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        xmat = np.asarray(self.data.site_xmat[self.site_id], dtype=float).reshape(3, 3)
        quat = quat_from_mat33_wxyz(xmat)
        return self.data.site_xpos[self.site_id].copy(), quat

    def set_gripper(self, value: float) -> None:
        self.data.ctrl[self.gripper_act_id] = float(np.clip(value, 0.0, 255.0))

    def solve_step(self, target_pos: np.ndarray, target_quat: np.ndarray) -> float:
        cur_pos, cur_quat = self.ee_pose()
        pos_err = target_pos - cur_pos
        rot_err = quat_to_rotvec(target_quat, cur_quat)
        err = np.concatenate([pos_err, rot_err])

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        _mj("mj_jacSite")(self.model, self.data, jacp, jacr, self.site_id)
        jac = np.vstack([jacp[:, self.dof_span], jacr[:, self.dof_span]])

        lhs = jac @ jac.T + 1e-4 * np.eye(6)
        dq = jac.T @ np.linalg.solve(lhs, 0.7 * err)
        self.data.ctrl[self.act_span] = self.data.qpos[self.jnt_span] + dq
        return float(np.linalg.norm(err))


@dataclass
class SceneRuntime:
    model: Any
    data: Any
    ee: EEController
    runtime_xml: Path


def _create_runtime_dir() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    out = LOG_ROOT / f"execute_mujoco_runtime_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _build_runtime_from_scene(scene_json_path: Path) -> SceneRuntime:
    plugin_available = _ensure_optional_mjlab_plugin()
    payload = json.loads(scene_json_path.read_text(encoding="utf-8"))
    assets_map = _discover_build_assets()

    robot = payload.get("robot", {}) if isinstance(payload.get("robot"), dict) else {}
    camera = payload.get("camera", {}) if isinstance(payload.get("camera"), dict) else {}
    assets = payload.get("assets", []) if isinstance(payload.get("assets"), list) else []

    robot_base_pos = np.asarray(robot.get("base_pos", [0.0, 0.0, 0.824]), dtype=float)
    robot_base_quat = normalize_quat_wxyz(np.asarray(robot.get("base_quat", [0.0, 0.0, 0.0, -1.0]), dtype=float))
    robot_joint_targets = np.asarray(robot.get("joint_targets", [0.0] * 6), dtype=float)
    robot_gripper = float(robot.get("gripper", 0.0))
    camera_pos = np.asarray(camera.get("pos", [0.0, -1.4, 1.45]), dtype=float)
    camera_quat = normalize_quat_wxyz(np.asarray(camera.get("quat", [0.819, 0.574, 0.0, 0.0]), dtype=float))

    if robot_base_pos.shape != (3,):
        robot_base_pos = np.array([0.0, 0.0, 0.824], dtype=float)
    if robot_joint_targets.shape != (6,):
        robot_joint_targets = np.zeros(6, dtype=float)
    if camera_pos.shape != (3,):
        camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)

    build_items: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for idx, entry in enumerate(assets, start=1):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key", "")).strip()
        if not key or key not in assets_map:
            continue

        preferred = str(entry.get("name", "")).strip() or Path(key).name
        base = _sanitize_mj_name(preferred, fallback="asset")
        name = base
        n = 2
        while name in used_names:
            name = f"{base}_{n}"
            n += 1
        used_names.add(name)

        pos = np.asarray(entry.get("pos", [0.0, 0.0, 0.845]), dtype=float)
        quat = normalize_quat_wxyz(np.asarray(entry.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float))
        scale = np.asarray(entry.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        if pos.shape != (3,):
            pos = np.array([0.0, 0.0, 0.845], dtype=float)
        if scale.shape != (3,):
            scale = np.array([1.0, 1.0, 1.0], dtype=float)
        scale = np.clip(scale.astype(float), 0.05, 50.0)

        build_items.append({
            "id": idx,
            "key": key,
            "instance_name": name,
            "pos": pos,
            "quat": quat,
            "scale": scale,
        })

    runtime_dir = _create_runtime_dir()
    runtime_xml = runtime_dir / "scene_runtime.xml"
    runtime_assets_dir = runtime_dir / "runtime_assets"

    def _emit_runtime_xml(strip_plugins: bool) -> None:
        lines = [
            '<mujoco model="execute_mujoco_scene">',
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
            asset_def = assets_map[item["key"]]
            source_xml = Path(asset_def["xml_path"])
            scale = item["scale"]
            if np.allclose(scale, np.ones(3, dtype=float)):
                model_file = _normalize_asset_xml_refs(
                    source_xml,
                    runtime_assets_dir / f"{source_xml.stem}__abs__id{item['id']}.xml",
                    strip_plugins=strip_plugins,
                )
            else:
                sx, sy, sz = [float(x) for x in scale.tolist()]
                model_file = _write_scaled_asset_xml(
                    source_xml,
                    runtime_assets_dir / f"{source_xml.stem}__scaled__id{item['id']}__s_{sx:.4f}_{sy:.4f}_{sz:.4f}.xml",
                    scale,
                    strip_plugins=strip_plugins,
                )
            model_name = _sanitize_mj_name(f"{item['instance_name']}_model", fallback="asset_model")
            lines.append(f'    <model name="{model_name}" file="{model_file.as_posix()}" content_type="text/xml"/>')

        lines.extend([
            '  </asset>',
            '  <worldbody>',
            '    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>',
            '    <geom name="floor" pos="0 0 0" size="2.5 2.5 0.05" type="plane" material="groundplane"/>',
            '    <body name="desk" pos="0 0 0" quat="1 0 0 1">',
            '      <attach model="desk_model" body="vention table" prefix="desk/"/>',
            f'      <camera name="table_cam_front" pos="{_format_float_list(camera_pos)}" quat="{_format_float_list(camera_quat)}" fovy="45" resolution="1280 960"/>',
            '    </body>',
            f'    <body name="ur5e_center" pos="{_format_float_list(robot_base_pos)}" quat="{_format_float_list(robot_base_quat)}">',
            '      <attach model="ur5e_model" body="world" prefix="/ur:"/>',
            '    </body>',
        ])

        for item in build_items:
            asset_def = assets_map[item["key"]]
            body_name = item["instance_name"]
            model_name = _sanitize_mj_name(f"{body_name}_model", fallback="asset_model")
            joint_name = _sanitize_mj_name(f"{body_name}_joint", fallback="asset_joint")
            lines.extend([
                f'    <body name="{body_name}" pos="{_format_float_list(item["pos"])}" quat="{_format_float_list(item["quat"])}">',
                f'      <joint name="{joint_name}" type="free"/>',
                f'      <attach model="{model_name}" body="{asset_def["root_body_name"]}" prefix="{body_name}/"/>',
                '    </body>',
            ])

        lines.extend(['  </worldbody>', '</mujoco>'])
        runtime_xml.write_text("\n".join(lines), encoding="utf-8")

    strip_plugins_fallback = not plugin_available
    _emit_runtime_xml(strip_plugins=strip_plugins_fallback)
    try:
        model = getattr(mujoco, "MjModel").from_xml_path(str(runtime_xml))
    except Exception as exc:
        msg = str(exc)
        plugin_missing = "plugin " in msg and " not found" in msg
        if (not strip_plugins_fallback) and plugin_missing:
            _log("Plugin was reported loaded but unresolved in XML; retrying with plugin tags stripped.")
            _emit_runtime_xml(strip_plugins=True)
            model = getattr(mujoco, "MjModel").from_xml_path(str(runtime_xml))
        else:
            raise
    data = getattr(mujoco, "MjData")(model)
    ee = EEController(model, data)

    _mj("mj_resetDataKeyframe")(model, data, 0)
    data.qpos[ee.jnt_span] = robot_joint_targets
    data.ctrl[ee.act_span] = robot_joint_targets
    data.ctrl[ee.gripper_act_id] = float(np.clip(robot_gripper, 0.0, 255.0))
    _mj("mj_forward")(model, data)

    _log(f"Loaded scene: {scene_json_path}")
    _log(f"Runtime XML: {runtime_xml}")
    return SceneRuntime(model=model, data=data, ee=ee, runtime_xml=runtime_xml)


@dataclass
class TeleopConfig:
    key_pos_step_m: float = 0.01
    key_rot_step_rad: float = 0.10
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
        self.pending_save_offset = False
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

    def request_save_offset(self) -> None:
        self.pending_save_offset = True

    def request_ctrl_latch(self, duration_s: float = 0.35) -> None:
        self.ctrl_latch_until = max(self.ctrl_latch_until, time.time() + float(duration_s))

    def is_ctrl_latched(self) -> bool:
        return time.time() <= self.ctrl_latch_until

    def pop_pending(self) -> tuple[np.ndarray, np.ndarray, bool, bool, bool]:
        dp = self.pending_dp.copy()
        rv = self.pending_rv.copy()
        tg = self.pending_toggle_gripper
        reset = self.pending_reset_target
        save_offset = self.pending_save_offset
        self.pending_dp[:] = 0.0
        self.pending_rv[:] = 0.0
        self.pending_toggle_gripper = False
        self.pending_reset_target = False
        self.pending_save_offset = False
        return dp, rv, tg, reset, save_offset


def _clamp_vec_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


def _apply_keyboard_delta(state: TeleopState, dp: np.ndarray, rv: np.ndarray, cfg: TeleopConfig) -> None:
    dp = _clamp_vec_norm(np.asarray(dp, dtype=float), float(cfg.max_pos_step_m))
    rv = _clamp_vec_norm(np.asarray(rv, dtype=float), float(cfg.max_rot_step_rad))
    angle = float(np.linalg.norm(rv))
    dq_scaled = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if angle < 1e-8 else quat_from_axis_angle(rv / angle, angle)
    state.target_pos = state.target_pos + dp
    state.target_quat = normalize_quat_wxyz(quat_mul(dq_scaled, state.target_quat))


def _key_to_motion_delta(key: int, cfg: TeleopConfig) -> tuple[np.ndarray, np.ndarray]:
    dp = np.zeros(3, dtype=float)
    rv = np.zeros(3, dtype=float)
    pos_step = float(cfg.key_pos_step_m)
    rot_step = float(cfg.key_rot_step_rad)

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


def run(args: argparse.Namespace) -> int:
    scene = Path(args.scene).expanduser()
    if not scene.is_absolute():
        scene = (PROJECT_ROOT / scene).resolve()
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")

    payload = json.loads(scene.read_text(encoding="utf-8"))
    fmt = str(payload.get("format", "")).strip()
    if fmt not in {"evobody_build_scene_v1", "evobody_build_scene_v2", "evobody_manual_scene_v1"}:
        raise ValueError(f"Unsupported scene format: {fmt!r}")

    runtime = _build_runtime_from_scene(scene)
    model = runtime.model
    data = runtime.data
    ee = runtime.ee

    cfg = TeleopConfig(
        key_pos_step_m=float(args.key_pos_step),
        key_rot_step_rad=float(args.key_rot_step),
        max_pos_step_m=float(args.max_pos_step),
        max_rot_step_rad=float(args.max_rot_step),
        solve_substeps=max(1, int(args.solve_substeps)),
        sim_steps_per_frame=max(1, int(args.sim_steps_per_frame)),
    )

    state = TeleopState(ee)
    _log("Keys: Up/Down(-/+X), Left/Right(-/+Y), PageUp/PageDown(+/-Z)")
    _log("Keys: Ctrl+Up/Down(+/-roll), Ctrl+Left/Right(+/-pitch), Ctrl+PageUp/PageDown(+/-yaw)")
    _log("Tip: if Ctrl combo is not captured on your platform, press Ctrl then press the direction key quickly")
    _log("Keys: Enter(toggle gripper), R(save EE grasp offset for selected object)")
    _log("Keys: Backspace(reset target to current EE)")

    def key_callback(key: int) -> None:
        if key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            state.request_ctrl_latch(0.40)
            return
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            state.request_toggle_gripper()
            return
        if key in (ord("r"), ord("R")):
            state.request_save_offset()
            return
        if key == glfw.KEY_BACKSPACE:
            state.request_reset_target()
            return
        if state.is_ctrl_latched():
            dp, rv = _key_to_ctrl_motion_delta(key, cfg)
        else:
            dp, rv = _key_to_motion_delta(key, cfg)
        if float(np.linalg.norm(dp)) > 0.0 or float(np.linalg.norm(rv)) > 0.0:
            state.request_motion(dp, rv)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            last_frame = time.perf_counter()
            while viewer.is_running():
                dp, rv, tg, reset, save_offset = state.pop_pending()
                if reset:
                    cur_pos, cur_quat = ee.ee_pose()
                    state.target_pos = cur_pos.copy()
                    state.target_quat = cur_quat.copy()
                    _log("Target reset to current EE pose")

                if save_offset:
                    pert = getattr(viewer, "pert", None)
                    selected_body_id = int(getattr(pert, "select", 0))
                    _log(_save_grasp_offset_for_selected_body(model, data, ee, selected_body_id))

                if tg:
                    state.gripper_closed = not state.gripper_closed
                    grip_val = cfg.gripper_close if state.gripper_closed else cfg.gripper_open
                    ee.set_gripper(grip_val)
                    _log(f"Gripper: {'CLOSE' if state.gripper_closed else 'OPEN'}")

                _apply_keyboard_delta(state, dp, rv, cfg)

                for _ in range(cfg.solve_substeps):
                    ee.solve_step(state.target_pos, state.target_quat)
                for _ in range(cfg.sim_steps_per_frame):
                    _mj("mj_step")(model, data)

                viewer.sync()

                frame_dt = float(model.opt.timestep) * float(cfg.sim_steps_per_frame)
                now = time.perf_counter()
                elapsed = now - last_frame
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                last_frame = time.perf_counter()
    finally:
        try:
            if runtime.runtime_xml.exists():
                shutil.rmtree(runtime.runtime_xml.parent)
                _log(f"Cleaned runtime dir: {runtime.runtime_xml.parent}")
        except Exception as exc:
            _log(f"Cleanup warning: {exc}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo GUI + keyboard Cartesian teleop")
    parser.add_argument("--scene", type=str, default=str(DEFAULT_SCENE_JSON), help="Scene JSON path")
    parser.add_argument("--key-pos-step", type=float, default=0.008, help="Position step for each key press (meters)")
    parser.add_argument("--key-rot-step", type=float, default=0.08, help="Rotation step for each key press (rad)")
    parser.add_argument("--max-pos-step", type=float, default=0.01, help="Max position step per frame")
    parser.add_argument("--max-rot-step", type=float, default=0.15, help="Max rotation step per frame")
    parser.add_argument("--solve-substeps", type=int, default=2, help="IK solve substeps per frame")
    parser.add_argument("--sim-steps-per-frame", type=int, default=2, help="Physics steps per frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _ensure_optional_mjlab_plugin()
        return run(args)
    except KeyboardInterrupt:
        _log("Interrupted by user")
        return 130
    except Exception as exc:
        _log(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
