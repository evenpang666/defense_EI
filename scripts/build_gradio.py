import json
import os
import queue
import re
import sys
import atexit
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import mujoco
import mujoco.viewer


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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_SCENE_CONFIG = LOG_ROOT / "exported_scene.json"
DEFAULT_EXPORT_PREFIX = LOG_ROOT / "exported_scene"
GRASP_OFFSETS_PATH = PROJECT_ROOT / "grasp_offsets.json"


def parse_root_body_name(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {xml_path}")
    bodies = worldbody.findall("body")
    if not bodies:
        raise ValueError(f"No <body> found in {xml_path}")
    names = [b.get("name") for b in bodies if b.get("name")]
    if "world" in names:
        return "world"
    return names[0]


def parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return list(default)


def format_float_list(values) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def sanitize_key(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return safe.strip("_") or "asset"


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


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    n2 = float(np.dot(q, q))
    if n2 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / n2


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    out = quat_mul(quat_mul(q, qv), quat_inv(q))
    return out[1:]


def mat9_to_quat_wxyz(mat9: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=float))
    n = float(np.linalg.norm(quat))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / n


def quat_to_euler_rpy_deg(q: np.ndarray) -> np.ndarray:
    # q is wxyz, output intrinsic XYZ (roll, pitch, yaw) in degrees.
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(float(sinp)) >= 1.0:
        pitch = np.pi / 2.0 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def wrap_deg180(angles_deg: np.ndarray) -> np.ndarray:
    vals = np.asarray(angles_deg, dtype=float)
    return (vals + 180.0) % 360.0 - 180.0


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    q = quat_mul(qz, quat_mul(qy, qx))
    return q / np.linalg.norm(q)


def resolve_input_path(file_obj, path_text: str | None) -> Path | None:
    if file_obj is not None:
        candidate = getattr(file_obj, "name", None) or str(file_obj)
        if candidate:
            return Path(candidate)
    text = (path_text or "").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def resolve_output_prefix(path_text: str | None) -> Path:
    text = (path_text or "").strip()
    if not text:
        return DEFAULT_EXPORT_PREFIX
    p = Path(text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.suffix.lower() in {".xml", ".json"}:
        p = p.with_suffix("")
    return p


def relpath_posix(target: Path, base: Path) -> str:
    try:
        return Path(target).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return Path(target).resolve().as_posix() if base is None else Path(
            Path(
                __import__("os").path.relpath(str(target.resolve()), str(base.resolve()))
            )
        ).as_posix()


def scale_numbers(values: list[float], scale: np.ndarray) -> list[float]:
    if not values:
        return values
    out = []
    for i, v in enumerate(values):
        out.append(float(v) * float(scale[min(i, 2)]))
    return out


def scale_attr(elem: ET.Element, attr: str, scale: np.ndarray):
    if attr not in elem.attrib:
        return
    vals = parse_float_list(elem.get(attr), [])
    if not vals:
        return
    elem.set(attr, format_float_list(scale_numbers(vals, scale)))


def scale_geom_size(elem: ET.Element, scale: np.ndarray):
    if "size" not in elem.attrib:
        return
    vals = parse_float_list(elem.get("size"), [])
    if not vals:
        return
    gtype = (elem.get("type") or "").lower()
    if gtype in {"sphere"} and len(vals) == 1:
        uniform = float(np.mean(scale))
        elem.set("size", f"{vals[0] * uniform:.6f}")
        return
    if gtype in {"capsule", "cylinder"} and len(vals) == 2:
        radius_scale = float((scale[0] + scale[1]) / 2.0)
        elem.set("size", format_float_list([vals[0] * radius_scale, vals[1] * scale[2]]))
        return
    elem.set("size", format_float_list(scale_numbers(vals, scale)))


def transform_asset_tree_for_scale(root: ET.Element, scale: np.ndarray, source_dir: Path):
    for elem in root.iter():
        tag = elem.tag
        if tag in {"body", "geom", "site", "camera", "light", "inertial", "joint"}:
            scale_attr(elem, "pos", scale)
        if tag == "geom":
            scale_geom_size(elem, scale)
            scale_attr(elem, "fromto", scale)
        elif tag == "site":
            scale_attr(elem, "size", scale)
            scale_attr(elem, "fromto", scale)
        elif tag == "mesh":
            current = parse_float_list(elem.get("scale"), [1.0, 1.0, 1.0])
            if len(current) == 1:
                current = current * 3
            if len(current) == 2:
                current = [current[0], current[1], 1.0]
            current = np.asarray(current[:3], dtype=float)
            new_scale = current * scale
            elem.set("scale", format_float_list(new_scale))
        elif tag == "texture":
            continue


@dataclass
class AssetDef:
    key: str
    xml_path: Path
    root_body_name: str


@dataclass
class PlacedAsset:
    id: int
    key: str
    model_name: str
    joint_name: str
    prefix: str
    pos: np.ndarray
    quat: np.ndarray
    scale: np.ndarray


class EEController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        self.act_id = model.actuator("/ur:shoulder_pan").id
        self.gripper_act_id = model.actuator("/ur:2f85:fingers_actuator").id
        self.jnt_span = slice(self.jnt_adr, self.jnt_adr + 6)
        self.act_span = slice(self.act_id, self.act_id + 6)

    def set_joint_targets(self, joints: np.ndarray):
        self.data.qpos[self.jnt_span] = joints
        self.data.ctrl[self.act_span] = joints

    def set_gripper(self, value: float):
        self.data.ctrl[self.gripper_act_id] = float(np.clip(value, 0.0, 255.0))


class BuildRuntime:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="evobody_build_"))
        self.generated_runtime_assets_dir = self.temp_dir / "runtime_assets"
        self.runtime_xml = self.temp_dir / "build_runtime.xml"
        self.assets = self._discover_assets()
        self.safe_key_to_key = {sanitize_key(k): k for k in self.assets.keys()}
        self.placed_assets: list[PlacedAsset] = []
        self._next_asset_id = 1

        self.robot_base_pos = np.array([0.0, 0.0, 0.824], dtype=float)
        self.robot_base_quat = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)
        self.robot_joint_targets = np.array([-1.5, -1.5, -1.5, 1.5, -1.5, 1.5], dtype=float)
        self.robot_gripper = 0.0
        self.camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)
        self.camera_quat = np.array([0.819, 0.574, 0.0, 0.0], dtype=float)

        self.lock = threading.RLock()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.ee: EEController | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._scene_update_callback = None

        self._write_runtime_xml()
        self._reload_model_from_xml()

    def _discover_assets(self) -> dict[str, AssetDef]:
        results: dict[str, AssetDef] = {}
        candidates = []
        candidates.extend((MODEL_ROOT / "object").glob("*.xml"))
        candidates.extend((MODEL_ROOT / "instrument").glob("*.xml"))
        for path in sorted(candidates):
            try:
                root_body = parse_root_body_name(path)
            except Exception:
                continue
            key = f"{path.parent.name}/{path.stem}"
            results[key] = AssetDef(key=key, xml_path=path, root_body_name=root_body)
        return results

    def _instance_asset_xml_path(self, item: PlacedAsset, asset_dir: Path) -> Path:
        asset_def = self.assets[item.key]
        sx, sy, sz = [float(x) for x in item.scale.tolist()]
        name = f"{asset_def.xml_path.stem}__scaled__id{item.id}__s_{sx:.4f}_{sy:.4f}_{sz:.4f}.xml"
        return asset_def.xml_path.parent / name

    def _write_scaled_asset_xml(self, item: PlacedAsset, asset_dir: Path) -> Path:
        asset_def = self.assets[item.key]
        if np.allclose(item.scale, np.ones(3, dtype=float)):
            return asset_def.xml_path
        out_path = self._instance_asset_xml_path(item, asset_dir)
        tree = ET.parse(asset_def.xml_path)
        root = tree.getroot()
        transform_asset_tree_for_scale(root, item.scale, asset_def.xml_path.parent)
        tree.write(out_path, encoding="utf-8", xml_declaration=False)
        return out_path

    def _build_scene_xml_text(self, generated_asset_dir: Path, base_dir: Path | None = None) -> str:
        base_dir = base_dir or self.temp_dir
        base_pos = format_float_list(self.robot_base_pos)
        base_quat = format_float_list(self.robot_base_quat)
        cam_pos = format_float_list(self.camera_pos)
        cam_quat = format_float_list(self.camera_quat)
        lines = [
            '<mujoco model="builder">',
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
            f'    <model name="desk_model" file="{relpath_posix(MODEL_ROOT / "misc" / "simple_table.xml", base_dir)}" content_type="text/xml"/>',
            f'    <model name="ur5e_model" file="{relpath_posix(MODEL_ROOT / "robot" / "ur5e_gripper.xml", base_dir)}" content_type="text/xml"/>',
        ]

        for item in self.placed_assets:
            scaled_xml = self._write_scaled_asset_xml(item, generated_asset_dir)
            lines.append(
                f'    <model name="{item.model_name}" file="{relpath_posix(scaled_xml, base_dir)}" content_type="text/xml"/>'
            )

        lines.extend(
            [
                '  </asset>',
                '  <worldbody>',
                '    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>',
                '    <geom name="floor" pos="0 0 0" size="2.5 2.5 0.05" type="plane" material="groundplane"/>',
                '    <body name="desk" pos="0 0 0" quat="1 0 0 1">',
                '      <attach model="desk_model" body="vention table" prefix="desk/"/>',
                f'      <camera name="table_cam_front" pos="{cam_pos}" quat="{cam_quat}" fovy="45" resolution="1280 960"/>',
                '    </body>',
                f'    <body name="ur5e_center" pos="{base_pos}" quat="{base_quat}">',
                '      <attach model="ur5e_model" body="world" prefix="/ur:"/>',
                '    </body>',
            ]
        )

        for item in self.placed_assets:
            asset = self.assets[item.key]
            pos = format_float_list(item.pos)
            quat = format_float_list(item.quat)
            sx, sy, sz = [float(x) for x in item.scale.tolist()]
            lines.extend(
                [
                    f'    <body name="asset_{item.id}" pos="{pos}" quat="{quat}">',
                    f'      <!-- asset_key={item.key}; scale={sx:.6f} {sy:.6f} {sz:.6f} -->',
                    f'      <joint name="{item.joint_name}" type="free"/>',
                    f'      <attach model="{item.model_name}" body="{asset.root_body_name}" prefix="{item.prefix}"/>',
                    '    </body>',
                ]
            )

        lines.extend(['  </worldbody>', '</mujoco>'])
        return "\n".join(lines)

    def _write_runtime_xml(self):
        self.generated_runtime_assets_dir.mkdir(parents=True, exist_ok=True)
        xml_text = self._build_scene_xml_text(self.generated_runtime_assets_dir, self.temp_dir)
        self.runtime_xml.write_text(xml_text, encoding="utf-8")

    def _reload_model_from_xml(self):
        self.model = mujoco.MjModel.from_xml_path(str(self.runtime_xml))
        self.data = mujoco.MjData(self.model)
        try:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        except Exception:
            pass
        self.ee = EEController(self.model, self.data)
        self._apply_robot_state_locked()
        self._sync_all_assets_locked()
        mujoco.mj_forward(self.model, self.data)
        self._notify_scene_update_locked()

    def set_scene_update_callback(self, callback):
        with self.lock:
            self._scene_update_callback = callback
            self._notify_scene_update_locked()

    def _notify_scene_update_locked(self):
        callback = self._scene_update_callback
        if callback is None or self.model is None or self.data is None:
            return
        callback(self.model, self.data)

    def _apply_robot_state_locked(self):
        if self.ee is None:
            return
        self.ee.set_joint_targets(self.robot_joint_targets)
        self.ee.set_gripper(self.robot_gripper)

    def _sync_all_assets_locked(self):
        for item in self.placed_assets:
            self._sync_asset_to_sim(item)

    def request_reload(self):
        with self.lock:
            self._write_runtime_xml()
            self._reload_model_from_xml()
            self.log_queue.put("[Scene] Reloaded")

    def _scene_payload(self) -> dict:
        return {
            "format": "evobody_build_scene_v2",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "robot": {
                "base_pos": [float(x) for x in self.robot_base_pos.tolist()],
                "base_quat": [float(x) for x in self.robot_base_quat.tolist()],
                "joint_targets": [float(x) for x in self.robot_joint_targets.tolist()],
                "gripper": float(self.robot_gripper),
            },
            "camera": {
                "pos": [float(x) for x in self.camera_pos.tolist()],
                "quat": [float(x) for x in self.camera_quat.tolist()],
            },
            "assets": [
                {
                    "key": item.key,
                    "pos": [float(x) for x in item.pos.tolist()],
                    "quat": [float(x) for x in item.quat.tolist()],
                    "scale": [float(x) for x in item.scale.tolist()],
                }
                for item in self.placed_assets
            ],
        }

    def export_scene(self, output_prefix: Path) -> tuple[Path, Path]:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = output_prefix.with_suffix(".json")
        xml_path = output_prefix.with_suffix(".xml")
        asset_dir = output_prefix.parent / f"{output_prefix.name}_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        json_path.write_text(json.dumps(self._scene_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        xml_text = self._build_scene_xml_text(asset_dir, xml_path.parent)
        xml_path.write_text(xml_text, encoding="utf-8")
        self.log_queue.put(f"[Scene] Exported XML: {xml_path}")
        self.log_queue.put(f"[Scene] Exported JSON: {json_path}")
        return xml_path, json_path

    def _append_loaded_asset(self, key: str, pos: np.ndarray, quat: np.ndarray, scale: np.ndarray):
        asset_id = self._next_asset_id
        self._next_asset_id += 1
        self.placed_assets.append(
            PlacedAsset(
                id=asset_id,
                key=key,
                model_name=f"user_model_{asset_id}",
                joint_name=f"asset_{asset_id}_joint",
                prefix=f"user{asset_id}/",
                pos=pos,
                quat=quat,
                scale=scale,
            )
        )

    def load_scene_config(self, load_path: Path | None = None):
        if load_path is None:
            load_path = DEFAULT_SCENE_CONFIG
        if not load_path.exists():
            raise FileNotFoundError(f"Config not found: {load_path}")

        content = json.loads(load_path.read_text(encoding="utf-8"))
        assets = content.get("assets", [])
        robot = content.get("robot", {})
        camera = content.get("camera", {})

        with self.lock:
            self.robot_base_pos = np.asarray(robot.get("base_pos", [0.0, 0.0, 0.824]), dtype=float)
            self.robot_base_quat = np.asarray(robot.get("base_quat", [0.0, 0.0, 0.0, -1.0]), dtype=float)
            joint_targets = np.asarray(robot.get("joint_targets", [0.0] * 6), dtype=float)
            if joint_targets.shape != (6,):
                joint_targets = np.zeros(6, dtype=float)
            self.robot_joint_targets = joint_targets
            self.robot_gripper = float(robot.get("gripper", 0.0))
            self.camera_pos = np.asarray(camera.get("pos", [0.0, -1.4, 1.45]), dtype=float)
            self.camera_quat = np.asarray(camera.get("quat", [0.819, 0.574, 0.0, 0.0]), dtype=float)
            if self.camera_pos.shape != (3,):
                self.camera_pos = np.array([0.0, -1.4, 1.45], dtype=float)
            if self.camera_quat.shape != (4,) or np.linalg.norm(self.camera_quat) < 1e-8:
                self.camera_quat = np.array([0.819, 0.574, 0.0, 0.0], dtype=float)
            self.camera_quat = self.camera_quat / np.linalg.norm(self.camera_quat)

            self.placed_assets.clear()
            self._next_asset_id = 1
            for entry in assets:
                key = entry.get("key")
                if key not in self.assets:
                    self.log_queue.put(f"[Scene] Skip unknown asset: {key}")
                    continue
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
                self._append_loaded_asset(key, pos, quat, scale)
            self.request_reload()

        self.log_queue.put(f"[Scene] Loaded JSON: {load_path}")

    def _infer_asset_key_and_scale(self, model_file: Path) -> tuple[str | None, np.ndarray]:
        try:
            resolved = model_file.resolve()
        except Exception:
            resolved = model_file

        for key, asset in self.assets.items():
            try:
                if asset.xml_path.resolve() == resolved:
                    return key, np.array([1.0, 1.0, 1.0], dtype=float)
            except Exception:
                if asset.xml_path == model_file:
                    return key, np.array([1.0, 1.0, 1.0], dtype=float)

        stem = model_file.stem
        m = re.match(r"(?P<safe>.+)__id\d+__s_(?P<sx>-?\d+(?:\.\d+)?)_(?P<sy>-?\d+(?:\.\d+)?)_(?P<sz>-?\d+(?:\.\d+)?)$", stem)
        if not m:
            return None, np.array([1.0, 1.0, 1.0], dtype=float)
        safe = m.group("safe")
        key = self.safe_key_to_key.get(safe)
        if key is None:
            return None, np.array([1.0, 1.0, 1.0], dtype=float)
        scale = np.array([
            float(m.group("sx")),
            float(m.group("sy")),
            float(m.group("sz")),
        ], dtype=float)
        scale = np.clip(scale, 0.05, 50.0)
        return key, scale

    def load_scene_xml(self, load_path: Path):
        if not load_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {load_path}")

        tree = ET.parse(load_path)
        root = tree.getroot()
        asset_root = root.find("asset")
        worldbody = root.find("worldbody")
        if asset_root is None or worldbody is None:
            raise ValueError("Invalid scene XML: missing <asset> or <worldbody>")

        model_files: dict[str, Path] = {}
        for model_elem in asset_root.findall("model"):
            name = model_elem.get("name")
            file_attr = model_elem.get("file")
            if not name or not file_attr:
                continue
            p = Path(file_attr)
            if not p.is_absolute():
                p = (load_path.parent / p).resolve()
            model_files[name] = p

        with self.lock:
            ur_body = worldbody.find("./body[@name='ur5e_center']")
            if ur_body is not None:
                self.robot_base_pos = np.asarray(parse_float_list(ur_body.get("pos"), [0.0, 0.0, 0.824]), dtype=float)
                self.robot_base_quat = np.asarray(parse_float_list(ur_body.get("quat"), [0.0, 0.0, 0.0, -1.0]), dtype=float)

            desk_body = worldbody.find("./body[@name='desk']")
            if desk_body is not None:
                cam = desk_body.find("./camera[@name='table_cam_front']")
                if cam is not None:
                    self.camera_pos = np.asarray(parse_float_list(cam.get("pos"), [0.0, -1.4, 1.45]), dtype=float)
                    self.camera_quat = np.asarray(parse_float_list(cam.get("quat"), [0.819, 0.574, 0.0, 0.0]), dtype=float)
                    if np.linalg.norm(self.camera_quat) > 1e-8:
                        self.camera_quat = self.camera_quat / np.linalg.norm(self.camera_quat)

            self.placed_assets.clear()
            self._next_asset_id = 1
            for body in worldbody.findall("body"):
                name = body.get("name", "")
                if not name.startswith("asset_"):
                    continue
                attach = body.find("attach")
                if attach is None:
                    continue
                model_name = attach.get("model")
                if not model_name:
                    continue
                model_file = model_files.get(model_name)
                if model_file is None:
                    self.log_queue.put(f"[Scene] Skip asset with unknown model file: {model_name}")
                    continue
                key, scale = self._infer_asset_key_and_scale(model_file)
                if key is None:
                    self.log_queue.put(f"[Scene] Skip unknown asset model file: {model_file}")
                    continue
                pos = np.asarray(parse_float_list(body.get("pos"), [0.0, 0.0, 0.845]), dtype=float)
                quat = np.asarray(parse_float_list(body.get("quat"), [1.0, 0.0, 0.0, 0.0]), dtype=float)
                if np.linalg.norm(quat) < 1e-8:
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                quat = quat / np.linalg.norm(quat)
                self._append_loaded_asset(key, pos, quat, scale)

            self.request_reload()

        self.log_queue.put(f"[Scene] Loaded XML: {load_path}")

    def load_scene_any(self, load_path: Path):
        suffix = load_path.suffix.lower()
        if suffix == ".json":
            self.load_scene_config(load_path)
        elif suffix == ".xml":
            self.load_scene_xml(load_path)
        else:
            raise ValueError("Only .json and .xml scene files are supported")

    def _sync_asset_to_sim(self, target: PlacedAsset):
        if self.model is None or self.data is None:
            return
        try:
            jnt_adr = self.model.joint(target.joint_name).qposadr.item()
        except KeyError:
            return
        self.data.qpos[jnt_adr : jnt_adr + 3] = target.pos
        self.data.qpos[jnt_adr + 3 : jnt_adr + 7] = target.quat

    def get_asset(self, asset_id: int | None) -> PlacedAsset | None:
        if asset_id is None:
            return None
        return next((a for a in self.placed_assets if a.id == asset_id), None)

    def add_asset(self, asset_key: str, x: float = 0.0, y: float = 0.0, z: float = 0.845):
        with self.lock:
            asset_id = self._next_asset_id
            self._next_asset_id += 1
            self.placed_assets.append(
                PlacedAsset(
                    id=asset_id,
                    key=asset_key,
                    model_name=f"user_model_{asset_id}",
                    joint_name=f"asset_{asset_id}_joint",
                    prefix=f"user{asset_id}/",
                    pos=np.array([x, y, z], dtype=float),
                    quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    scale=np.array([1.0, 1.0, 1.0], dtype=float),
                )
            )
            self.request_reload()
            self.log_queue.put(f"[Scene] Added asset: {asset_key}")

    def remove_asset(self, asset_id: int):
        with self.lock:
            self.placed_assets = [a for a in self.placed_assets if a.id != asset_id]
            self.request_reload()
            self.log_queue.put(f"[Scene] Removed asset: {asset_id}")

    def translate_asset(self, asset_id: int, dx: float, dy: float, dz: float):
        with self.lock:
            target = self.get_asset(asset_id)
            if target is None:
                return
            target.pos = target.pos + np.array([dx, dy, dz], dtype=float)
            target.pos[0] = float(np.clip(target.pos[0], -1.1, 1.1))
            target.pos[1] = float(np.clip(target.pos[1], -1.1, 1.1))
            target.pos[2] = float(np.clip(target.pos[2], 0.80, 1.40))
            self._sync_asset_to_sim(target)
            mujoco.mj_forward(self.model, self.data)

    def rotate_asset(self, asset_id: int, axis: str, degree: float):
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            return
        with self.lock:
            target = self.get_asset(asset_id)
            if target is None:
                return
            if axis == "x":
                vec = np.array([1.0, 0.0, 0.0])
            elif axis == "y":
                vec = np.array([0.0, 1.0, 0.0])
            else:
                vec = np.array([0.0, 0.0, 1.0])
            delta = quat_from_axis_angle(vec, np.deg2rad(degree))
            target.quat = quat_mul(delta, target.quat)
            target.quat /= np.linalg.norm(target.quat)
            self._sync_asset_to_sim(target)
            mujoco.mj_forward(self.model, self.data)

    def set_asset_scale(self, asset_id: int, sx: float, sy: float, sz: float):
        with self.lock:
            target = self.get_asset(asset_id)
            if target is None:
                return
            target.scale = np.clip(np.array([sx, sy, sz], dtype=float), 0.05, 50.0)
            self.request_reload()
            self.log_queue.put(
                f"[Scene] Asset {asset_id} scale -> ({target.scale[0]:.3f}, {target.scale[1]:.3f}, {target.scale[2]:.3f})"
            )

    def set_robot_base(self, x: float, y: float, z: float):
        with self.lock:
            self.robot_base_pos = np.array([x, y, z], dtype=float)
            self.request_reload()
            self.log_queue.put(
                f"[Robot] Base updated to ({self.robot_base_pos[0]:.3f}, {self.robot_base_pos[1]:.3f}, {self.robot_base_pos[2]:.3f})"
            )

    def set_robot_joints(self, joints: list[float], gripper: float):
        with self.lock:
            if len(joints) != 6:
                raise ValueError("Need exactly 6 joint values")
            self.robot_joint_targets = np.asarray(joints, dtype=float)
            self.robot_gripper = float(np.clip(gripper, 0.0, 255.0))
            self._apply_robot_state_locked()
            mujoco.mj_forward(self.model, self.data)
            self.log_queue.put("[Robot] Joint targets and gripper updated")

    def set_camera_position(self, x: float, y: float, z: float):
        with self.lock:
            self.camera_pos = np.array([x, y, z], dtype=float)
            self.request_reload()
            self.log_queue.put(
                f"[Camera] Position updated to ({self.camera_pos[0]:.3f}, {self.camera_pos[1]:.3f}, {self.camera_pos[2]:.3f})"
            )

    def rotate_camera(self, axis: str, degree: float):
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            return
        with self.lock:
            if axis == "x":
                vec = np.array([1.0, 0.0, 0.0])
            elif axis == "y":
                vec = np.array([0.0, 1.0, 0.0])
            else:
                vec = np.array([0.0, 0.0, 1.0])
            delta = quat_from_axis_angle(vec, np.deg2rad(degree))
            self.camera_quat = quat_mul(delta, self.camera_quat)
            self.camera_quat = self.camera_quat / np.linalg.norm(self.camera_quat)
            self.request_reload()
            self.log_queue.put(f"[Camera] Rotated around {axis.upper()} by {degree:.1f} deg")

    def set_camera_euler(self, roll_deg: float, pitch_deg: float, yaw_deg: float):
        with self.lock:
            self.camera_quat = quat_from_euler(
                np.deg2rad(float(roll_deg)),
                np.deg2rad(float(pitch_deg)),
                np.deg2rad(float(yaw_deg)),
            )
            self.request_reload()
            self.log_queue.put(
                f"[Camera] Euler updated to ({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}) deg"
            )

    def _find_gripper_site_id(self) -> int:
        if self.model is None:
            return -1
        candidates = [
            "/ur:2f85:pinch",
            "ur:2f85:pinch",
            "2f85:pinch",
            "/ur:attachment_site",
            "ur:attachment_site",
            "attachment_site",
        ]
        for name in candidates:
            try:
                return self.model.site(name).id
            except Exception:
                continue
        return -1

    def get_gripper_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.data is None:
            raise RuntimeError("Model not initialized")
        site_id = self._find_gripper_site_id()
        if site_id < 0:
            raise RuntimeError("Gripper site not found (pinch/attachment_site)")
        pos = np.array(self.data.site_xpos[site_id], dtype=float)
        quat = mat9_to_quat_wxyz(np.array(self.data.site_xmat[site_id], dtype=float))
        return pos, quat

    def record_grasp_pose_for_asset(self, asset_id: int, pose_name: str, grasp_offsets_path: Path = GRASP_OFFSETS_PATH):
        with self.lock:
            target = self.get_asset(asset_id)
            if target is None:
                raise ValueError("Selected asset not found")
            if self.model is None or self.data is None:
                raise RuntimeError("Model not initialized")
            mujoco.mj_forward(self.model, self.data)
            grip_pos_w, grip_quat_w = self.get_gripper_pose_world()

            obj_pos_w = np.array(target.pos, dtype=float)
            obj_quat_w = np.array(target.quat, dtype=float)
            obj_quat_w = obj_quat_w / max(float(np.linalg.norm(obj_quat_w)), 1e-12)
            # Record as world-frame offsets from object to gripper.
            delta_pos = grip_pos_w - obj_pos_w
            grip_rpy_deg = quat_to_euler_rpy_deg(grip_quat_w)
            obj_rpy_deg = quat_to_euler_rpy_deg(obj_quat_w)
            delta_rpy_deg = wrap_deg180(grip_rpy_deg - obj_rpy_deg)

            payload: dict = {}
            if grasp_offsets_path.exists():
                try:
                    payload = json.loads(grasp_offsets_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
            if not isinstance(payload, dict):
                payload = {}

            node = payload.get(target.key)
            if not isinstance(node, dict):
                node = {}
                payload[target.key] = node

            node[pose_name] = {
                "pos": [round(float(v), 6) for v in delta_pos.tolist()],
                "3d_rotation": [round(float(v), 6) for v in delta_rpy_deg.tolist()],
            }
            grasp_offsets_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            self.log_queue.put(
                f"[GraspOffsets] Recorded {target.key}/{pose_name} -> "
                f"pos={node[pose_name]['pos']}, rpy={node[pose_name]['3d_rotation']}"
            )


class ExplicitMujocoInterface:
    def __init__(self, scene_lock: threading.RLock):
        self._scene_lock = scene_lock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reload_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._scene: tuple[mujoco.MjModel, mujoco.MjData] | None = None

    def set_scene(self, model: mujoco.MjModel, data: mujoco.MjData):
        with self._lock:
            self._scene = (model, data)
            self._reload_event.set()
            should_start = self._thread is None or not self._thread.is_alive()
            if should_start:
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run, name="build-gradio-viewer", daemon=True)
                self._thread.start()

    def stop(self):
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                scene = self._scene
                self._reload_event.clear()
            if scene is None:
                time.sleep(0.05)
                continue

            model, data = scene
            try:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    while viewer.is_running() and not self._stop_event.is_set():
                        with self._lock:
                            scene_changed = self._scene is not None and self._scene != (model, data)
                            should_reload = self._reload_event.is_set()
                        if scene_changed or should_reload:
                            break
                        with self._scene_lock:
                            viewer.sync()
                        time.sleep(1 / 60.0)
            except Exception as exc:
                print(f"[build_gradio] Viewer error: {exc}", flush=True)
                time.sleep(0.2)


RUNTIME = BuildRuntime()
MUJOCO_INTERFACE = ExplicitMujocoInterface(RUNTIME.lock)
RUNTIME.set_scene_update_callback(MUJOCO_INTERFACE.set_scene)
atexit.register(MUJOCO_INTERFACE.stop)


def _drain_logs() -> str:
    logs = []
    while True:
        try:
            logs.append(RUNTIME.log_queue.get_nowait())
        except Exception:
            break
    return "\n".join(logs)


def _collect_logs(previous: str = "") -> str:
    chunk = _drain_logs()
    if not previous:
        return chunk
    if not chunk:
        return previous
    return previous + "\n" + chunk


def _placed_choices() -> list[str]:
    out = []
    for item in RUNTIME.placed_assets:
        out.append(
            f"{item.id}: {item.key} @ ({item.pos[0]:.2f}, {item.pos[1]:.2f}, {item.pos[2]:.2f}) | scale=({item.scale[0]:.2f}, {item.scale[1]:.2f}, {item.scale[2]:.2f})"
        )
    return out


def _extract_asset_id(choice: str | None):
    if not choice:
        return None
    try:
        return int(choice.split(":", 1)[0])
    except Exception:
        return None


def refresh_panel(log_text):
    return gr.update(choices=_placed_choices()), _collect_logs(log_text or "")


def add_asset(asset_key, log_text):
    if asset_key:
        RUNTIME.add_asset(asset_key)
    return gr.update(choices=_placed_choices(), value=None), _collect_logs(log_text or "")


def remove_asset(selected_asset, log_text):
    asset_id = _extract_asset_id(selected_asset)
    if asset_id is None and RUNTIME.placed_assets:
        asset_id = RUNTIME.placed_assets[-1].id
    if asset_id is not None:
        RUNTIME.remove_asset(asset_id)
    return gr.update(choices=_placed_choices(), value=None), _collect_logs(log_text or "")


def inspect_asset(selected_asset):
    asset_id = _extract_asset_id(selected_asset)
    item = RUNTIME.get_asset(asset_id)
    if item is None:
        return 1.0, 1.0, 1.0, "No asset selected"
    info = (
        f"Asset {item.id}\n"
        f"key: {item.key}\n"
        f"pos: ({item.pos[0]:.4f}, {item.pos[1]:.4f}, {item.pos[2]:.4f})\n"
        f"scale: ({item.scale[0]:.4f}, {item.scale[1]:.4f}, {item.scale[2]:.4f})"
    )
    return float(item.scale[0]), float(item.scale[1]), float(item.scale[2]), info


def apply_asset_scale(selected_asset, sx, sy, sz, log_text):
    asset_id = _extract_asset_id(selected_asset)
    if asset_id is None:
        RUNTIME.log_queue.put("[UI] Select one placed asset first")
        return gr.update(), _collect_logs(log_text or "")
    RUNTIME.set_asset_scale(asset_id, float(sx), float(sy), float(sz))
    choices = _placed_choices()
    current = next((c for c in choices if c.startswith(f"{asset_id}:")), None)
    return gr.update(choices=choices, value=current), _collect_logs(log_text or "")


def transform_asset(selected_asset, tx, ty, tz, rx, ry, rz, move_step, rot_step, log_text):
    asset_id = _extract_asset_id(selected_asset)
    if asset_id is None:
        RUNTIME.log_queue.put("[UI] Select one placed asset first")
        return gr.update(), _collect_logs(log_text or "")

    if tx != 0 or ty != 0 or tz != 0:
        RUNTIME.translate_asset(asset_id, tx * float(move_step), ty * float(move_step), tz * float(move_step))
    if rx != 0:
        RUNTIME.rotate_asset(asset_id, "x", rx * float(rot_step))
    if ry != 0:
        RUNTIME.rotate_asset(asset_id, "y", ry * float(rot_step))
    if rz != 0:
        RUNTIME.rotate_asset(asset_id, "z", rz * float(rot_step))

    choices = _placed_choices()
    current = next((c for c in choices if c.startswith(f"{asset_id}:")), None)
    return gr.update(choices=choices, value=current), _collect_logs(log_text or "")


def auto_apply_robot(base_x, base_y, base_z, j1, j2, j3, j4, j5, j6, gripper, log_text):
    RUNTIME.set_robot_base(float(base_x), float(base_y), float(base_z))
    RUNTIME.set_robot_joints([j1, j2, j3, j4, j5, j6], gripper)
    return _collect_logs(log_text or "")


def auto_apply_camera_position(cam_x, cam_y, cam_z, log_text):
    RUNTIME.set_camera_position(float(cam_x), float(cam_y), float(cam_z))
    return _collect_logs(log_text or "")


def export_scene(output_prefix_text, log_text):
    output_prefix = resolve_output_prefix(output_prefix_text)
    xml_path, json_path = RUNTIME.export_scene(output_prefix)
    return str(xml_path), str(json_path), _collect_logs(log_text or "")


def record_pose(selected_asset, pose_name, log_text):
    asset_id = _extract_asset_id(selected_asset)
    pose_name = (pose_name or "").strip()
    if asset_id is None:
        RUNTIME.log_queue.put("[UI] Please select one placed object first")
        return _collect_logs(log_text or "")
    if not pose_name:
        RUNTIME.log_queue.put("[UI] Please input pose name first")
        return _collect_logs(log_text or "")

    try:
        RUNTIME.record_grasp_pose_for_asset(asset_id, pose_name, GRASP_OFFSETS_PATH)
    except Exception as exc:
        RUNTIME.log_queue.put(f"[GraspOffsets] Record failed: {exc}")
    return _collect_logs(log_text or "")


def load_scene(scene_file, scene_path_text, log_text):
    path = resolve_input_path(scene_file, scene_path_text)
    if path is None:
        raise ValueError("Please provide a .xml or .json scene file")
    RUNTIME.load_scene_any(path)

    joints = RUNTIME.robot_joint_targets.tolist()

    def _f(value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    base_x = _f(RUNTIME.robot_base_pos[0], 0.0)
    base_y = _f(RUNTIME.robot_base_pos[1], 0.0)
    base_z = _f(RUNTIME.robot_base_pos[2], 0.824)

    j1 = _f(joints[0] if len(joints) > 0 else 0.0, 0.0)
    j2 = _f(joints[1] if len(joints) > 1 else 0.0, 0.0)
    j3 = _f(joints[2] if len(joints) > 2 else 0.0, 0.0)
    j4 = _f(joints[3] if len(joints) > 3 else 0.0, 0.0)
    j5 = _f(joints[4] if len(joints) > 4 else 0.0, 0.0)
    j6 = _f(joints[5] if len(joints) > 5 else 0.0, 0.0)

    gripper = _f(RUNTIME.robot_gripper, 0.0)
    cam_x = _f(RUNTIME.camera_pos[0], 0.0)
    cam_y = _f(RUNTIME.camera_pos[1], -1.4)
    cam_z = _f(RUNTIME.camera_pos[2], 1.45)

    return (
        gr.update(choices=_placed_choices(), value=None),
        base_x,
        base_y,
        base_z,
        j1,
        j2,
        j3,
        j4,
        j5,
        j6,
        gripper,
        cam_x,
        cam_y,
        cam_z,
        _collect_logs(log_text or ""),
    )


def build_app():
    asset_keys = sorted(RUNTIME.assets.keys())

    with gr.Blocks(title="EvoBody Builder") as demo:
        gr.Markdown("## EvoBody Builder")
        gr.Markdown("支持 XML / JSON 场景导入，资产缩放，原生 MuJoCo 窗口预览，以及手动指定输出路径导出 XML + JSON。")

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### Asset Builder")
                asset_dropdown = gr.Dropdown(choices=asset_keys, label="Asset Library", interactive=True)
                with gr.Row():
                    add_btn = gr.Button("Add Asset", variant="primary")
                    remove_btn = gr.Button("Remove Selected")

                placed_dropdown = gr.Dropdown(choices=_placed_choices(), label="Placed Assets", interactive=True)
                asset_info = gr.Textbox(label="Selected Asset Info", lines=4, interactive=False)
                with gr.Row():
                    scale_x = gr.Number(value=1.0, label="Scale X")
                    scale_y = gr.Number(value=1.0, label="Scale Y")
                    scale_z = gr.Number(value=1.0, label="Scale Z")
                apply_scale_btn = gr.Button("Apply Size / Scale")

                move_step = gr.Slider(0.005, 0.10, value=0.02, step=0.005, label="Move Step (m)")
                rot_step = gr.Slider(1, 45, value=10, step=1, label="Rotate Step (deg)")

                gr.Markdown("Move")
                with gr.Row():
                    tx_p = gr.Button("+X")
                    tx_n = gr.Button("-X")
                    ty_p = gr.Button("+Y")
                    ty_n = gr.Button("-Y")
                    tz_p = gr.Button("+Z")
                    tz_n = gr.Button("-Z")

                gr.Markdown("Rotate")
                with gr.Row():
                    rx_p = gr.Button("+Rx")
                    rx_n = gr.Button("-Rx")
                    ry_p = gr.Button("+Ry")
                    ry_n = gr.Button("-Ry")
                    rz_p = gr.Button("+Rz")
                    rz_n = gr.Button("-Rz")

                gr.Markdown("### Robot Config")
                base_x = gr.Slider(-1.2, 1.2, value=float(RUNTIME.robot_base_pos[0]), step=0.01, label="Base X")
                base_y = gr.Slider(-1.2, 1.2, value=float(RUNTIME.robot_base_pos[1]), step=0.01, label="Base Y")
                base_z = gr.Slider(0.6, 1.2, value=float(RUNTIME.robot_base_pos[2]), step=0.005, label="Base Z")
                j1 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[0]), step=0.01, label="Joint 1")
                j2 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[1]), step=0.01, label="Joint 2")
                j3 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[2]), step=0.01, label="Joint 3")
                j4 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[3]), step=0.01, label="Joint 4")
                j5 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[4]), step=0.01, label="Joint 5")
                j6 = gr.Slider(-3.14, 3.14, value=float(RUNTIME.robot_joint_targets[5]), step=0.01, label="Joint 6")
                gripper = gr.Slider(0.0, 255.0, value=float(RUNTIME.robot_gripper), step=1.0, label="Gripper")

                gr.Markdown("### Camera Config")
                cam_x = gr.Slider(-2.0, 2.0, value=float(RUNTIME.camera_pos[0]), step=0.01, label="Camera X")
                cam_y = gr.Slider(-2.0, 2.0, value=float(RUNTIME.camera_pos[1]), step=0.01, label="Camera Y")
                cam_z = gr.Slider(0.2, 2.5, value=float(RUNTIME.camera_pos[2]), step=0.01, label="Camera Z")

                gr.Markdown("### Scene Import / Export")
                scene_file = gr.File(label="Import Scene File (.xml / .json)", file_types=[".xml", ".json"])
                scene_path = gr.Textbox(value="logs/exported_scene.json", label="Scene Path (optional, if not uploading)")
                output_prefix = gr.Textbox(value="logs/exported_scene", label="Output Path Prefix (without extension)")
                xml_output_path = gr.Textbox(label="Exported XML Path", interactive=False)
                json_output_path = gr.Textbox(label="Exported JSON Path", interactive=False)
                with gr.Row():
                    export_btn = gr.Button("Export XML + JSON", variant="primary")
                    load_scene_btn = gr.Button("Load Scene")

                gr.Markdown("### Grasp Offsets Recorder")
                pose_name_input = gr.Textbox(
                    value="",
                    label="Pose Name (e.g. beaker_edge)",
                    placeholder="input pose name, then click record_pose",
                )
                record_pose_btn = gr.Button("record_pose", variant="secondary")

            with gr.Column(scale=6):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                logs_box = gr.Textbox(value="", label="Logs", lines=20, interactive=False)

        add_btn.click(add_asset, inputs=[asset_dropdown, logs_box], outputs=[placed_dropdown, logs_box])
        remove_btn.click(remove_asset, inputs=[placed_dropdown, logs_box], outputs=[placed_dropdown, logs_box])

        placed_dropdown.change(inspect_asset, inputs=[placed_dropdown], outputs=[scale_x, scale_y, scale_z, asset_info])
        apply_scale_btn.click(apply_asset_scale, inputs=[placed_dropdown, scale_x, scale_y, scale_z, logs_box], outputs=[placed_dropdown, logs_box])

        tx_p.click(transform_asset, [placed_dropdown, gr.State(1), gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        tx_n.click(transform_asset, [placed_dropdown, gr.State(-1), gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        ty_p.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(1), gr.State(0), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        ty_n.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(-1), gr.State(0), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        tz_p.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(1), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        tz_n.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(-1), gr.State(0), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])

        rx_p.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(1), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        rx_n.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(-1), gr.State(0), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        ry_p.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(1), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        ry_n.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(-1), gr.State(0), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        rz_p.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(1), move_step, rot_step, logs_box], [placed_dropdown, logs_box])
        rz_n.click(transform_asset, [placed_dropdown, gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(0), gr.State(-1), move_step, rot_step, logs_box], [placed_dropdown, logs_box])

        robot_inputs = [base_x, base_y, base_z, j1, j2, j3, j4, j5, j6, gripper, logs_box]
        for comp in [base_x, base_y, base_z, j1, j2, j3, j4, j5, j6, gripper]:
            comp.change(auto_apply_robot, inputs=robot_inputs, outputs=[logs_box])

        for comp in [cam_x, cam_y, cam_z]:
            comp.change(auto_apply_camera_position, inputs=[cam_x, cam_y, cam_z, logs_box], outputs=[logs_box])

        export_btn.click(export_scene, inputs=[output_prefix, logs_box], outputs=[xml_output_path, json_output_path, logs_box])
        record_pose_btn.click(
            record_pose,
            inputs=[placed_dropdown, pose_name_input, logs_box],
            outputs=[logs_box],
        )
        load_scene_btn.click(
            load_scene,
            inputs=[scene_file, scene_path, logs_box],
            outputs=[placed_dropdown, base_x, base_y, base_z, j1, j2, j3, j4, j5, j6, gripper, cam_x, cam_y, cam_z, logs_box],
        )

        refresh_btn.click(refresh_panel, inputs=[logs_box], outputs=[placed_dropdown, logs_box])

        timer = gr.Timer(1.2)
        timer.tick(refresh_panel, inputs=[logs_box], outputs=[placed_dropdown, logs_box])

    return demo

def main():
    app = build_app()
    app.queue()

    for port in [7862, 7863, 7864, 7865, 7870]:
        try:
            app.launch(
                server_name="localhost",
                server_port=port,
                theme=gr.themes.Soft(),
                allowed_paths=[str(PROJECT_ROOT), str(LOG_ROOT), str(tempfile.gettempdir())],
            )
            return
        except OSError:
            continue

    raise RuntimeError("没有找到可用端口，请先关闭占用 786x 的 Gradio 进程。")

if __name__ == "__main__":
    main()