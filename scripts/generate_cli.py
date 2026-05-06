import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
import random

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import imageio.v2 as imageio
import mujoco
import numpy as np
from openai import OpenAI


from evoma import (
    EvoMAAgentPipeline,
    SegmentSuccessJudgeAgent,
    TaskSuccessJudgeAgent,
    TaskSupervisorAgent,
    format_judge_feedback_for_atomic_skill,
    format_segment_failure_for_atomic_skill,
    parse_evo_phase_segments,
)

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
LOG_ROOT = PROJECT_ROOT / "logs"
EVOMA_PLAN_JSON = LOG_ROOT / "evoma_plan.json"
ATOMIC_OPS_PATH = Path(__file__).resolve().parent / "evoma_atomic_ops.py"
MODEL_ROOT = PROJECT_ROOT / "model"
FINISH_SCENE_CONFIG = LOG_ROOT / "finished_scene.json"
PRE_SCENE_CONFIG = LOG_ROOT / "pre_scene.json"
LEROBOT_HOME = Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


EVOMA_ENABLE_STAGE_JUDGE = _env_bool("EVOMA_ENABLE_STAGE_JUDGE", False)


def _load_atomic_ops_source() -> str:
    """Read evoma_atomic_ops.py; exec'd into user code namespace alongside move_to/ee_pose."""
    try:
        return ATOMIC_OPS_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


_DEFAULT_STAGE_JUDGE_ATOMIC_OPS_NAMES = {
    "move_x",
    "move_y",
    "move_z",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "pick_and_place",
    "push",
    "pull",
    "press",
    "open",
    "close",
    "pour",
    "get_object_abs_pose",
    "recover_grasp_pose_from_offset",
}
_PRIMITIVE_RUNTIME_API_NAMES = {"move_to", "move_ee", "gripper_control", "ee_pose"}


def _extract_atomic_ops_public_names(source: str) -> set[str]:
    """Extract public callable names from ``evoma_atomic_ops.py`` __all__ list."""
    if not source.strip():
        return set(_DEFAULT_STAGE_JUDGE_ATOMIC_OPS_NAMES)
    try:
        tree = ast.parse(source)
    except Exception:
        return set(_DEFAULT_STAGE_JUDGE_ATOMIC_OPS_NAMES)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        names: set[str] = set()
        for elt in node.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.add(elt.value.strip())
        if names:
            return names
    return set(_DEFAULT_STAGE_JUDGE_ATOMIC_OPS_NAMES)


def _called_function_names(code: str) -> set[str]:
    """Collect function names called by this phase body."""
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    return {m.group(1) for m in pattern.finditer(code or "")}


_STAGE_JUDGE_ATOMIC_OPS_NAMES = _extract_atomic_ops_public_names(_load_atomic_ops_source())


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_to_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_err = quat_mul(target, quat_conj(current))
    if q_err[0] < 0:
        q_err = -q_err
    vec = q_err[1:]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vec_norm, q_err[0])
    axis = vec / vec_norm
    return axis * angle


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


def parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    try:
        return [float(x) for x in text.strip().split()]
    except Exception:
        return list(default)


def format_float_list(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def scale_numbers(values: list[float], scale: np.ndarray) -> list[float]:
    if not values:
        return values
    out = []
    for i, v in enumerate(values):
        out.append(float(v) * float(scale[min(i, 2)]))
    return out


def scale_attr(elem: ET.Element, attr: str, scale: np.ndarray) -> None:
    if attr not in elem.attrib:
        return
    vals = parse_float_list(elem.get(attr), [])
    if not vals:
        return
    elem.set(attr, format_float_list(scale_numbers(vals, scale)))


def scale_geom_size(elem: ET.Element, scale: np.ndarray) -> None:
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


def transform_asset_tree_for_scale(root: ET.Element, scale: np.ndarray) -> None:
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
            elem.set("scale", format_float_list(current * scale))


def _absolutize_xml_file_refs(root: ET.Element, source_dir: Path) -> None:
    """Convert all relative file paths in XML to absolute paths.
    
    Handles:
    - compiler meshdir/texturedir (relative to XML source_dir)
    - mesh/texture file attributes (relative to their respective dirs or source_dir)
    - include file attributes (relative to source_dir)
    """
    # First pass: Resolve compiler meshdir/texturedir to absolute paths
    mesh_dir = source_dir
    texture_dir = source_dir
    
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir_attr = compiler.get("meshdir")
        if meshdir_attr:
            meshdir_text = str(meshdir_attr).strip()
            if meshdir_text and not Path(meshdir_text).is_absolute():
                mesh_dir = (source_dir / meshdir_text).resolve()
                compiler.set("meshdir", mesh_dir.as_posix())
            else:
                mesh_dir = Path(meshdir_text).resolve()
        
        texturedir_attr = compiler.get("texturedir")
        if texturedir_attr:
            texturedir_text = str(texturedir_attr).strip()
            if texturedir_text and not Path(texturedir_text).is_absolute():
                texture_dir = (source_dir / texturedir_text).resolve()
                compiler.set("texturedir", texture_dir.as_posix())
            else:
                texture_dir = Path(texturedir_text).resolve()
    
    # Second pass: Resolve mesh/texture files relative to their respective dirs
    for elem in root.iter():
        if elem.tag == "mesh":
            file_attr = elem.get("file")
            if file_attr:
                file_text = str(file_attr).strip()
                if file_text and not Path(file_text).is_absolute():
                    elem.set("file", (mesh_dir / file_text).resolve().as_posix())
        elif elem.tag == "texture":
            file_attr = elem.get("file")
            if file_attr:
                file_text = str(file_attr).strip()
                if file_text and not Path(file_text).is_absolute():
                    elem.set("file", (texture_dir / file_text).resolve().as_posix())
        elif elem.tag == "include":
            file_attr = elem.get("file")
            if file_attr:
                file_text = str(file_attr).strip()
                if file_text and not Path(file_text).is_absolute():
                    elem.set("file", (source_dir / file_text).resolve().as_posix())


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q /= np.linalg.norm(q)
        return q
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    q = s0 * q0 + s1 * q1
    q /= np.linalg.norm(q)
    return q


class EEController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
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
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return self.data.site_xpos[self.site_id].copy(), quat

    def set_gripper(self, value: float):
        self.data.ctrl[self.gripper_act_id] = float(np.clip(value, 0.0, 255.0))

    def solve_step(self, target_pos: np.ndarray, target_quat: np.ndarray) -> float:
        current_pos, current_quat = self.ee_pose()
        pos_err = target_pos - current_pos
        rot_err = quat_to_rotvec(target_quat, current_quat)
        err = np.concatenate([pos_err, rot_err])

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jac = np.vstack([jacp[:, self.dof_span], jacr[:, self.dof_span]])

        lhs = jac @ jac.T + 1e-4 * np.eye(6)
        dq = jac.T @ np.linalg.solve(lhs, 0.7 * err)
        self.data.ctrl[self.act_span] = self.data.qpos[self.jnt_span] + dq
        return float(np.linalg.norm(err))


class CliRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.ee: EEController | None = None
        self._viewer_ready = threading.Event()
        self._busy = False
        self.scene_json_path: Path | None = None
        self.scene_xml_path: Path | None = None
        self.scene_assets: list[dict[str, Any]] = []
        self._cached_background_pair: tuple[Path, Path] | None = None
        self._viewer_ready.set()

    @staticmethod
    def _is_generated_build_runtime_xml(path: Path | None) -> bool:
        if path is None:
            return False
        try:
            resolved = path.resolve()
            log_root = LOG_ROOT.resolve()
        except Exception:
            return False
        if not (resolved.name.startswith("build_runtime_") and resolved.suffix.lower() == ".xml"):
            return False
        if not resolved.parent.name.startswith("evobody_generate_build_"):
            return False
        try:
            resolved.relative_to(log_root)
        except ValueError:
            return False
        return True

    def _cleanup_cached_build_runtime_xml(self) -> None:
        old_xml = self.scene_xml_path
        if not self._is_generated_build_runtime_xml(old_xml):
            return
        old_dir = old_xml.parent
        try:
            if old_dir.exists() and old_dir.is_dir():
                shutil.rmtree(old_dir)
                _log(f"[Scene] removed previous build cache dir: {old_dir}")
        except Exception as err:
            _log(f"[Scene] failed to remove previous build cache dir {old_dir}: {err}")

    @staticmethod
    def _create_timestamped_runtime_dir(prefix: str) -> Path:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx in range(100):
            suffix = stamp if idx == 0 else f"{stamp}_{idx:02d}"
            runtime_dir = LOG_ROOT / f"{prefix}{suffix}"
            try:
                runtime_dir.mkdir(parents=False, exist_ok=False)
                return runtime_dir
            except FileExistsError:
                continue
        raise RuntimeError("Failed to create unique timestamped runtime directory")

    @staticmethod
    def _body_free_joint_qpos_adr(model: mujoco.MjModel, body_id: int) -> int | None:
        body_jnt_num = int(model.body_jntnum[body_id])
        body_jnt_adr = int(model.body_jntadr[body_id])
        for k in range(body_jnt_num):
            jnt_id = body_jnt_adr + k
            if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[jnt_id])
        return None

    def _apply_snapshot_assets(self, assets: list[dict[str, Any]]) -> int:
        if self.model is None or self.data is None:
            return 0
        applied = 0
        for entry in assets:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                continue
            pos = np.asarray(entry.get("pos", [0.0, 0.0, 0.0]), dtype=float)
            quat = np.asarray(entry.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
            if pos.shape != (3,) or quat.shape != (4,):
                continue
            if np.linalg.norm(quat) < 1e-8:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                quat = quat / np.linalg.norm(quat)

            qpos_adr = self._body_free_joint_qpos_adr(self.model, body_id)
            if qpos_adr is not None:
                self.data.qpos[qpos_adr:qpos_adr + 3] = pos
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
            else:
                self.model.body_pos[body_id] = pos
                self.model.body_quat[body_id] = quat
            applied += 1

        if applied > 0:
            mujoco.mj_forward(self.model, self.data)
        return applied

    def _apply_robot_arm_state(self, robot_arm: dict[str, Any] | None) -> bool:
        if robot_arm is None or self.model is None or self.data is None or self.ee is None:
            return False
        joint_qpos = np.asarray(robot_arm.get("joint_qpos", []), dtype=float)
        if joint_qpos.shape != (6,):
            return False
        gripper_qpos = robot_arm.get("gripper_qpos", None)
        if gripper_qpos is None:
            return False

        try:
            grip_qpos_adr = self.model.joint("/ur:2f85:right_driver_joint").qposadr.item()
            grip_act_id = self.model.actuator("/ur:2f85:fingers_actuator").id
        except Exception:
            return False

        self.data.qpos[self.ee.jnt_span] = joint_qpos
        self.data.ctrl[self.ee.act_span] = joint_qpos
        self.data.qpos[grip_qpos_adr] = float(gripper_qpos)
        self.data.ctrl[grip_act_id] = float(np.clip(float(gripper_qpos) * 2550.0, 0.0, 255.0))
        mujoco.mj_forward(self.model, self.data)
        return True

    def _snapshot_robot_arm_state(self) -> dict[str, Any] | None:
        if self.model is None or self.data is None or self.ee is None:
            return None
        grip_qpos_adr = self.model.joint("/ur:2f85:right_driver_joint").qposadr.item()
        ee_pos, ee_quat = self.ee.ee_pose()
        return {
            "joint_qpos": [float(x) for x in self.data.qpos[self.ee.jnt_span].tolist()],
            "gripper_qpos": float(self.data.qpos[grip_qpos_adr]),
            "ee_pos": [float(x) for x in ee_pos.tolist()],
            "ee_quat": [float(x) for x in ee_quat.tolist()],
        }

    def _snapshot_assets_state(self) -> list[dict[str, Any]]:
        if self.model is None or self.data is None:
            return []
        assets: list[dict[str, Any]] = []
        for body_id in range(1, int(self.model.nbody)):
            qpos_adr = self._body_free_joint_qpos_adr(self.model, body_id)
            if qpos_adr is None:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                continue
            pos = self.data.qpos[qpos_adr : qpos_adr + 3]
            quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm < 1e-8:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                quat = quat / quat_norm
            assets.append(
                {
                    "name": str(name),
                    "pos": [float(x) for x in pos.tolist()],
                    "quat": [float(x) for x in quat.tolist()],
                }
            )
        return assets

    @staticmethod
    def _parse_root_body_name(xml_path: Path) -> str:
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

    def _discover_build_assets(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        candidates = []
        candidates.extend((MODEL_ROOT / "object").glob("*.xml"))
        candidates.extend((MODEL_ROOT / "instrument").glob("*.xml"))
        for path in sorted(candidates):
            try:
                root_body = self._parse_root_body_name(path)
            except Exception:
                continue
            key = f"{path.parent.name}/{path.stem}"
            results[key] = {"xml_path": path, "root_body_name": root_body}
        return results

    def _write_scaled_asset_xml(self, source_xml: Path, out_xml: Path, scale: np.ndarray) -> Path:
        tree = ET.parse(source_xml)
        root = tree.getroot()
        transform_asset_tree_for_scale(root, scale)
        _absolutize_xml_file_refs(root, source_xml.parent)
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_xml, encoding="utf-8", xml_declaration=False)
        return out_xml

    def _normalize_asset_xml_refs(self, source_xml: Path, out_xml: Path) -> Path:
        """Normalize file references in asset XML to absolute paths without scaling."""
        tree = ET.parse(source_xml)
        root = tree.getroot()
        _absolutize_xml_file_refs(root, source_xml.parent)
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_xml, encoding="utf-8", xml_declaration=False)
        return out_xml

    def _select_randomized_textures(self, randomize: bool) -> tuple[str, str | None]:
        """Select background and table textures. Returns (bg_texture_line, table_texture_line)."""
        background_dir = PROJECT_ROOT / "assets" / "background_texture"
        background_images = []
        
        if background_dir.exists():
            background_images = list(background_dir.glob("*.png")) + \
                               list(background_dir.glob("*.jpg")) + \
                               list(background_dir.glob("*.jpeg"))
        
        if len(background_images) < 2:
            _log("[Scene] Fewer than 2 background images, using default textures")
            return (
                '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
                None
            )
        
        selected_bg1 = None
        selected_bg2 = None
        
        if randomize:
            random.seed(int(time.time()))
            selected_bg1, selected_bg2 = random.sample(background_images, 2)
            self._cached_background_pair = (selected_bg1, selected_bg2)
        else:
            # Deterministic textures only: do not reuse a pair chosen by a prior
            # randomize=True load in the same process (would look "still random").
            sorted_images = sorted(background_images)
            selected_bg1, selected_bg2 = sorted_images[0], sorted_images[1]
            self._cached_background_pair = (selected_bg1, selected_bg2)
        
        bg1_path = selected_bg1.name
        bg2_path = selected_bg2.name
        bg_texture_line = f'    <texture name="background_tex" type="2d" file="{bg1_path}" width="1024" height="1024"/>'
        table_texture_line = f'    <texture name="table_tex" type="2d" file="{bg2_path}" width="1024" height="1024"/>'
        
        _log(f"[Scene] Background image: {selected_bg1.name}")
        _log(f"[Scene] Table texture: {selected_bg2.name}")
        
        return bg_texture_line, table_texture_line

    def _copy_textures_to_runtime(self, runtime_dir: Path, selected_bg1: Path, selected_bg2: Path) -> None:
        """Copy selected texture images to runtime directory."""
        bg1_dest = runtime_dir / selected_bg1.name
        bg2_dest = runtime_dir / selected_bg2.name
        shutil.copy2(selected_bg1, bg1_dest)
        shutil.copy2(selected_bg2, bg2_dest)
        _log(f"[Scene] Copied textures to {runtime_dir.name}")

    def _load_build_scene_v1(self, scene_json_path: Path, payload: dict[str, Any], randomize_texture: bool = True, cleanup_old_cache: bool = True) -> None:
        # Keep only the latest generated build runtime XML; remove previous cache per evogen run.
        # Set cleanup_old_cache=False to reuse cached scene (e.g., after report generation).
        if cleanup_old_cache:
            self._cleanup_cached_build_runtime_xml()

        # Extract parameters directly, aligned with build_gradio.py's load_scene_config logic
        assets_map = self._discover_build_assets()
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

        cached_xml = self.scene_xml_path
        cached_scene_json = self.scene_json_path
        if (
            not cleanup_old_cache
            and cached_xml is not None
            and cached_scene_json is not None
            and self._is_generated_build_runtime_xml(cached_xml)
            and cached_xml.exists()
        ):
            try:
                same_scene = cached_scene_json.resolve() == scene_json_path.resolve()
            except Exception:
                same_scene = cached_scene_json == scene_json_path
            if same_scene:
                _log(f"[Scene] Reusing cached build runtime xml: {cached_xml}")
                with self.lock:
                    self.model = mujoco.MjModel.from_xml_path(str(cached_xml))
                    self.data = mujoco.MjData(self.model)
                    self.ee = EEController(self.model, self.data)
                    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                    try:
                        self.data.qpos[self.ee.jnt_span] = robot_joint_targets
                        self.data.ctrl[self.ee.act_span] = robot_joint_targets
                        self.data.ctrl[self.ee.gripper_act_id] = float(np.clip(robot_gripper, 0.0, 255.0))
                    except Exception:
                        pass
                    mujoco.mj_forward(self.model, self.data)
                    self.scene_json_path = scene_json_path
                    self.scene_xml_path = cached_xml
                    self.scene_assets = assets
                return

        build_items: list[dict[str, Any]] = []
        used_instance_names: set[str] = set()
        for idx, entry in enumerate(assets, start=1):
            key = str(entry.get("key", "")).strip()
            if not key or key not in assets_map:
                continue
            preferred_name = str(entry.get("name", "")).strip() if isinstance(entry, dict) else ""
            if not preferred_name:
                preferred_name = Path(key).name
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
            build_items.append({
                "id": idx,
                "key": key,
                "instance_name": instance_name,
                "pos": pos,
                "quat": quat,
                "scale": scale,
            })

        runtime_dir = self._create_timestamped_runtime_dir("evobody_generate_build_")
        runtime_suffix = runtime_dir.name[len("evobody_generate_build_"):]
        runtime_xml = runtime_dir / f"build_runtime_{runtime_suffix}.xml"

        base_pos_text = " ".join(f"{x:.6f}" for x in robot_base_pos)
        base_quat_text = " ".join(f"{x:.6f}" for x in robot_base_quat)
        cam_pos_text = " ".join(f"{x:.6f}" for x in camera_pos)
        cam_quat_text = " ".join(f"{x:.6f}" for x in camera_quat)
        
        # Apply randomized textures after loading (select them here, copy after XML dir created)
        bg_texture_line, table_texture_line = self._select_randomized_textures(randomize_texture)

        lines = [
            '<mujoco model="builder_generate">',
            '  <option integrator="implicitfast" impratio="10" cone="elliptic" noslip_iterations="2">',
            '    <flag multiccd="enable"/>',
            '  </option>',
            '  <visual>',
            '    <global azimuth="220" elevation="-30" offwidth="1280" offheight="960"/>',
            '  </visual>',
            '  <asset>',
            '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
            bg_texture_line,
            '    <material name="background_mat" texture="background_tex" texrepeat="4 4"/>',
            table_texture_line,
            '    <material name="table_mat" texture="table_tex" texrepeat="1 1"/>',
            '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>',
            '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>',
            f'    <model name="desk_model" file="{(MODEL_ROOT / "misc" / "simple_table.xml").as_posix()}" content_type="text/xml"/>',
            f'    <model name="ur5e_model" file="{(MODEL_ROOT / "robot" / "ur5e_gripper.xml").as_posix()}" content_type="text/xml"/>',
        ]

        scaled_asset_dir = runtime_dir / "runtime_assets"
        for item in build_items:
            asset_def = assets_map[item["key"]]
            model_file = asset_def["xml_path"]
            scale = item["scale"]
            if not np.allclose(scale, np.ones(3, dtype=float)):
                sx, sy, sz = [float(x) for x in scale.tolist()]
                scaled_name = f"{asset_def['xml_path'].stem}__scaled__id{item['id']}__s_{sx:.4f}_{sy:.4f}_{sz:.4f}.xml"
                model_file = self._write_scaled_asset_xml(asset_def["xml_path"], scaled_asset_dir / scaled_name, scale)
            model_name = _sanitize_mj_name(f"{item['instance_name']}_model", fallback="asset_model")
            lines.append(
                f'    <model name="{model_name}" file="{model_file.as_posix()}" content_type="text/xml"/>'
            )

        lines.extend(
            [
                '  </asset>',
                '  <worldbody>',
                '    <light directional="true" diffuse="0.8 0.8 0.8" ambient="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>',
                '    <geom name="floor" pos="0 0 0" size="2.5 2.5 0.05" type="plane" material="groundplane"/>',
                '    <geom name="background_wall_front" type="plane" pos="0 2.5 1.5" quat="0.707 0.707 0 0" size="4 3 0.1" material="background_mat" contype="0" conaffinity="0"/>',
                '    <geom name="background_wall_back" type="plane" pos="0 -2.5 1.5" quat="0.707 -0.707 0 0" size="4 3 0.1" material="background_mat" contype="0" conaffinity="0"/>',
                '    <geom name="background_wall_left" type="plane" pos="-2.5 0 1.5" quat="0.707 0 -0.707 0" size="4 3 0.1" material="background_mat" contype="0" conaffinity="0"/>',
                '    <geom name="background_wall_right" type="plane" pos="2.5 0 1.5" quat="0.707 0 0.707 0" size="4 3 0.1" material="background_mat" contype="0" conaffinity="0"/>',
                '    <body name="desk" pos="0 0 0" quat="1 0 0 1">',
                '      <attach model="desk_model" body="vention table" prefix="desk/"/>',
                '      <geom name="table_surface_overlay" type="plane" pos="0 0 0.8241" size="1.2 0.80 0.01" material="table_mat" contype="0" conaffinity="0" group="1"/>',
                f'      <camera name="table_cam_front" pos="{cam_pos_text}" quat="{cam_quat_text}" fovy="45" resolution="1280 960"/>',
                '    </body>',
                f'    <body name="ur5e_center" pos="{base_pos_text}" quat="{base_quat_text}">',
                '      <attach model="ur5e_model" body="world" prefix="/ur:"/>',
                '    </body>',
            ]
        )

        for item in build_items:
            asset_def = assets_map[item["key"]]
            pos_text = " ".join(f"{x:.6f}" for x in item["pos"])
            quat_text = " ".join(f"{x:.6f}" for x in item["quat"])
            body_name = item["instance_name"]
            joint_name = _sanitize_mj_name(f"{body_name}_joint", fallback="asset_joint")
            model_name = _sanitize_mj_name(f"{body_name}_model", fallback="asset_model")
            lines.extend(
                [
                    f'    <body name="{body_name}" pos="{pos_text}" quat="{quat_text}">',
                    f'      <joint name="{joint_name}" type="free"/>',
                    f'      <attach model="{model_name}" body="{asset_def["root_body_name"]}" prefix="{body_name}/"/>',
                    '    </body>',
                ]
            )

        lines.extend(['  </worldbody>', '</mujoco>'])
        runtime_xml.write_text("\n".join(lines), encoding="utf-8")
        
        # Copy selected textures to runtime directory
        background_dir = PROJECT_ROOT / "assets" / "background_texture"
        if background_dir.exists():
            all_images = list(background_dir.glob("*.png")) + \
                        list(background_dir.glob("*.jpg")) + \
                        list(background_dir.glob("*.jpeg"))
            if len(all_images) >= 2 and self._cached_background_pair is not None:
                selected_bg1, selected_bg2 = self._cached_background_pair
                if selected_bg1.exists() and selected_bg2.exists():
                    self._copy_textures_to_runtime(runtime_dir, selected_bg1, selected_bg2)

        with self.lock:
            self.model = mujoco.MjModel.from_xml_path(str(runtime_xml))
            self.data = mujoco.MjData(self.model)
            self.ee = EEController(self.model, self.data)
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            try:
                self.data.qpos[self.ee.jnt_span] = robot_joint_targets
                self.data.ctrl[self.ee.act_span] = robot_joint_targets
                self.data.ctrl[self.ee.gripper_act_id] = float(np.clip(robot_gripper, 0.0, 255.0))
            except Exception:
                pass

            mujoco.mj_forward(self.model, self.data)
            self.scene_json_path = scene_json_path
            self.scene_xml_path = runtime_xml
            self.scene_assets = assets

    def load_scene_config(self, load_path: Path, randomize_texture: bool = True, cleanup_old_cache: bool = True) -> None:
        payload = json.loads(load_path.read_text(encoding="utf-8"))
        payload_format = str(payload.get("format", "")).strip()

        # Align with build_gradio JSON loading: consume robot/camera/assets directly.
        if payload_format in {"evobody_build_scene_v1", "evobody_build_scene_v2", "evobody_manual_scene_v1"} or (
            not payload_format and isinstance(payload.get("assets"), list) and not payload.get("scene_xml")
        ):
            self._load_build_scene_v1(load_path, payload, randomize_texture=randomize_texture, cleanup_old_cache=cleanup_old_cache)
            robot_arm = payload.get("robot_arm") if isinstance(payload.get("robot_arm"), dict) else None
            if robot_arm is not None:
                with self.lock:
                    self._apply_robot_arm_state(robot_arm)
            return

        scene_xml_text = str(payload.get("scene_xml", "")).strip()
        if not scene_xml_text:
            raise ValueError(
                "Scene json cannot be loaded. Unsupported format and missing 'scene_xml'. "
                f"payload_format={payload_format!r}."
            )

        scene_xml = Path(scene_xml_text).expanduser()
        if not scene_xml.is_absolute():
            scene_xml = PROJECT_ROOT / scene_xml
        if not scene_xml.exists():
            raise FileNotFoundError(f"scene_xml does not exist: {scene_xml}")

        with self.lock:
            self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
            self.data = mujoco.MjData(self.model)
            self.ee = EEController(self.model, self.data)
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)

            assets = payload.get("assets", []) if isinstance(payload.get("assets"), list) else []
            self._apply_snapshot_assets(assets)
            self._apply_robot_arm_state(payload.get("robot_arm"))
            mujoco.mj_forward(self.model, self.data)

            self.scene_json_path = load_path
            self.scene_xml_path = scene_xml
            self.scene_assets = assets

    def save_front_image(self, save_path: Path) -> None:
        with self.lock:
            if self.model is None or self.data is None:
                raise RuntimeError("Runtime model/data is not initialized")
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "table_cam_front")
            if camera_id < 0:
                raise ValueError("Camera 'table_cam_front' not found")
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id

            renderer = mujoco.Renderer(self.model, 256, 256)
            try:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                renderer._scene_option.sitegroup[:] = False
                renderer.update_scene(self.data)
                mujoco.mjv_updateCamera(self.model, self.data, camera, renderer._scene)
                image = renderer.render().astype(np.uint8)
            finally:
                renderer.close()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(save_path, image)

    def save_wrist_image(self, save_path: Path) -> None:
        with self.lock:
            if self.model is None or self.data is None:
                raise RuntimeError("Runtime model/data is not initialized")
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "/ur:wrist_cam")
            if camera_id < 0:
                raise ValueError("Camera '/ur:wrist_cam' not found")
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id

            renderer = mujoco.Renderer(self.model, 256, 256)
            try:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                renderer._scene_option.sitegroup[:] = False
                renderer.update_scene(self.data)
                mujoco.mjv_updateCamera(self.model, self.data, camera, renderer._scene)
                image = renderer.render().astype(np.uint8)
            finally:
                renderer.close()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(save_path, image)

    def save_scene_config(self, save_path: Path) -> None:
        with self.lock:
            if self.scene_xml_path is None:
                raise RuntimeError("No loaded scene xml to save")
            try:
                scene_xml_text = str(self.scene_xml_path.relative_to(PROJECT_ROOT))
            except Exception:
                scene_xml_text = str(self.scene_xml_path)
            payload = {
                "format": "evobody_scene_snapshot_v2",
                "scene_mode": "xml_override",
                "scene_xml": scene_xml_text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "assets": self._snapshot_assets_state(),
                "robot_arm": self._snapshot_robot_arm_state(),
            }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


RUNTIME = CliRuntime()


def _normalize_task_dir_name(name: str | None) -> str:
    text = " ".join((name or "").split()).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "manual_execution"


def _plan_summary_for_state_action() -> str | None:
    """Short plan summary from logs/evoma_plan.json (``summary`` field)."""
    if not EVOMA_PLAN_JSON.exists():
        return None
    try:
        plan = json.loads(EVOMA_PLAN_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None
    s = str(plan.get("summary", "")).strip()
    return s or None


def _primitive_log_dir_name_from_plan() -> str | None:
    """Basename under logs/ from first atomic task's ``primitive`` in evoma_plan.json (TaskSupervisorAgent output)."""
    if not EVOMA_PLAN_JSON.exists():
        return None
    try:
        plan = json.loads(EVOMA_PLAN_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None
    tasks = TaskSupervisorAgent._atomic_tasks_from_payload(plan)
    if not tasks:
        return None
    prim = str(tasks[0].get("primitive", "")).strip()
    if not prim:
        return None
    return _normalize_task_dir_name(prim)


def _lerobot_repo_id_from_plan() -> str | None:
    """LeRobot ``repo_id`` as ``<primitive>/<summary_slug>`` from logs/evoma_plan.json.

    ``summary_slug`` is the plan ``summary`` lowercased with non-alphanumeric runs
    replaced by underscores (same as :func:`_normalize_task_dir_name`). Call after
    the plan on disk reflects the current run.
    """
    prim = _primitive_log_dir_name_from_plan()
    if not prim:
        return None
    summary_raw = _plan_summary_for_state_action()
    if not summary_raw:
        return None
    summary_seg = _normalize_task_dir_name(summary_raw)
    if not summary_seg:
        return None
    return f"{prim}/{summary_seg}"


def _lerobot_task_prompt_from_plan(cli_task: str) -> str:
    """Dataset task string: plan ``summary`` lowercased + underscored, else normalized CLI ``--task``."""
    s = _plan_summary_for_state_action()
    if s:
        return _normalize_task_dir_name(s)
    return _normalize_task_prompt(cli_task)


class ExecutionRecorder:
    FRAME_WIDTH = 256
    FRAME_HEIGHT = 256

    def __init__(
        self,
        runtime: CliRuntime,
        task_prompt: str | None = None,
        enabled: bool = True,
        render_engine: str = "mujoco",
    ):
        self.runtime = runtime
        self.enabled = enabled
        self.render_engine = str(render_engine or "mujoco").strip().lower()
        if self.render_engine not in {"mujoco", "blender"}:
            self.render_engine = "mujoco"
        sub_norm = _normalize_saved_subtask_prompt(task_prompt)
        self._log_parent_fallback = _normalize_task_dir_name(sub_norm)
        plan_summary = _plan_summary_for_state_action()
        self.task_prompt = (
            _normalize_task_dir_name(plan_summary) if plan_summary is not None else sub_norm
        )
        self.run_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.wrist_frames_dir: Path | None = None
        self.renderer: mujoco.Renderer | None = None
        self.cameras: dict[str, mujoco.MjvCamera] = {}
        self.blender_renderer: BlenderFrameRenderer | None = None
        self.records: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []
        self.frame_idx = 0
        self._prev_ee_pos: np.ndarray | None = None
        self._prev_ee_quat: np.ndarray | None = None
        self.qpos_records: list[np.ndarray] = []

        if not enabled:
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_parent = _primitive_log_dir_name_from_plan() or self._log_parent_fallback
        self.run_dir = LOG_ROOT / log_parent / stamp
        self.frames_dir = self.run_dir / "frames"
        self.wrist_frames_dir = self.run_dir / "wrist_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_frames_dir.mkdir(parents=True, exist_ok=True)

        model = runtime.model
        data = runtime.data
        if model is None or data is None:
            raise RuntimeError("Runtime model/data is not initialized")

        if self.render_engine == "blender":
            scene_xml = runtime.scene_xml_path
            if scene_xml is None:
                raise RuntimeError(
                    "Cannot start Blender live renderer: runtime has no scene_xml_path"
                )
            self.blender_renderer = BlenderFrameRenderer(
                scene_xml=Path(scene_xml),
                width=self.FRAME_WIDTH,
                height=self.FRAME_HEIGHT,
            )
        else:
            self.renderer = mujoco.Renderer(model, self.FRAME_WIDTH, self.FRAME_HEIGHT)
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

    @staticmethod
    def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
        # MuJoCo quaternion ordering: [w, x, y, z]
        w, x, y, z = [float(v) for v in quat]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.asarray([roll, pitch, yaw], dtype=float)

    def _capture(self, key: str) -> np.ndarray | None:
        camera = self.cameras.get(key)
        if camera is None or self.renderer is None:
            return None
        model = self.runtime.model
        data = self.runtime.data
        mujoco.mjv_updateCamera(model, data, camera, self.renderer._scene)
        return self.renderer.render().astype(np.uint8)

    @staticmethod
    def _read_rendered_png(path: Path) -> np.ndarray:
        img = imageio.imread(path)
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr.astype(np.uint8)

    def _state_action(self) -> tuple[list[float], list[float]]:
        model = self.runtime.model
        data = self.runtime.data
        jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        grip_jnt = model.joint("/ur:2f85:right_driver_joint").qposadr.item()
        grip_act = model.actuator("/ur:2f85:fingers_actuator").id
        state = np.concatenate([data.qpos[jnt_adr:jnt_adr + 6], [data.qpos[grip_jnt]]]).tolist()
        ee = self.runtime.ee
        cur_pos, cur_quat = ee.ee_pose()
        cur_pos = np.asarray(cur_pos, dtype=float)
        cur_quat = np.asarray(cur_quat, dtype=float)
        if self._prev_ee_pos is None or self._prev_ee_quat is None:
            dpos = np.zeros(3, dtype=float)
            deuler = np.zeros(3, dtype=float)
        else:
            dpos = cur_pos - self._prev_ee_pos
            q_delta = quat_mul(cur_quat, quat_conj(self._prev_ee_quat))
            if q_delta[0] < 0:
                q_delta = -q_delta
            q_delta = q_delta / max(1e-12, float(np.linalg.norm(q_delta)))
            deuler = self._quat_to_euler_xyz(q_delta)
        self._prev_ee_pos = cur_pos.copy()
        self._prev_ee_quat = cur_quat.copy()
        action = np.concatenate([dpos, deuler, [data.ctrl[grip_act]]]).astype(float).tolist()
        return state, action

    def record(self, tag: str = "step") -> None:
        if not self.enabled:
            return
        data = self.runtime.data
        frame_path = self.frames_dir / f"{self.frame_idx:06d}.png"
        wrist_frame_path = self.wrist_frames_dir / f"{self.frame_idx:06d}.png"

        if self.render_engine == "blender":
            if self.blender_renderer is None:
                raise RuntimeError("Blender renderer not initialized")
            qpos_snapshot = np.asarray(data.qpos, dtype=np.float64).copy()
            self.blender_renderer.render(
                qpos_snapshot,
                front_path=frame_path,
                wrist_path=wrist_frame_path,
            )
            img = self._read_rendered_png(frame_path)
            if not wrist_frame_path.exists():
                imageio.imwrite(wrist_frame_path, img)
        else:
            self.renderer.update_scene(data)
            img = self._capture("image")
            if img is None:
                raise RuntimeError("Failed to capture image from camera 'table_cam_front'")
            wrist_img = self._capture("wrist_image")
            if wrist_img is None:
                wrist_img = img
            imageio.imwrite(frame_path, img)
            imageio.imwrite(wrist_frame_path, wrist_img)

        self.frames.append(img)
        state, action = self._state_action()
        self.records.append({
            "frame": frame_path.name,
            "wrist_frame": wrist_frame_path.name,
            "tag": tag,
            "time": float(data.time),
            "state": state,
            "action": action,
        })
        self.qpos_records.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.frame_idx += 1

    def close(self) -> str | None:
        if not self.enabled:
            return None
        if self.runtime.model is not None:
            try:
                mujoco.mj_saveModel(self.runtime.model, str(self.run_dir / "model.mjb"), None)
            except Exception as err:
                _log(f"[Blender] warning: failed to export model.mjb: {err}")
        scene_xml_src = self.runtime.scene_xml_path
        if scene_xml_src is not None and scene_xml_src.exists():
            shutil.copy2(scene_xml_src, self.run_dir / "scene.xml")
            scene_source_dir = self.run_dir / "scene_source"
            if scene_source_dir.exists():
                shutil.rmtree(scene_source_dir, ignore_errors=True)
            # Preserve sibling resources referenced by relative paths in scene.xml.
            shutil.copytree(scene_xml_src.parent, scene_source_dir, dirs_exist_ok=True)
            # Normalize source entry name for render_blender directory loader.
            shutil.copy2(scene_xml_src, scene_source_dir / "scene.xml")
        if self.qpos_records:
            np.save(self.run_dir / "qpos.npy", np.asarray(self.qpos_records, dtype=np.float64))
        video_path = self.run_dir / "execution.mp4"
        if self.frames:
            imageio.mimwrite(video_path, self.frames, fps=40)
        (self.run_dir / "state_action.json").write_text(
            json.dumps({"task_prompt": self.task_prompt, "records": self.records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.blender_renderer is not None:
            try:
                self.blender_renderer.close()
            except Exception as exc:
                _log(f"[Blender] live worker close warning: {exc}")
            self.blender_renderer = None
        return str(video_path) if video_path.exists() else None


def _build_atomic_exec_globals_for_run(recorder: ExecutionRecorder, _step) -> dict[str, Any]:
    def ee_pose():
        with RUNTIME.lock:
            pos, quat = RUNTIME.ee.ee_pose()
            return pos.copy(), quat.copy()

    def move_to(pos, quat=None, num_steps: int = 100):
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

    def move_ee(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, steps=120):
        with RUNTIME.lock:
            cur_pos, cur_quat = RUNTIME.ee.ee_pose()
        target_pos = cur_pos + np.array([dx, dy, dz], dtype=float)
        dquat = quat_from_euler(droll, dpitch, dyaw)
        target_quat = quat_mul(dquat, cur_quat)
        target_quat /= np.linalg.norm(target_quat)
        move_to(target_pos, target_quat, int(steps))

    def gripper_control(value: float, delay: int = 50):
        with RUNTIME.lock:
            RUNTIME.ee.set_gripper(value)
        for _ in range(max(1, int(delay))):
            _step("gripper")

    api = {
        "np": np,
        "ee_pose": ee_pose,
        "move_to": move_to,
        "move_ee": move_ee,
        "gripper_control": gripper_control,
        "set_gripper": lambda value: gripper_control(value, delay=1),
        "sleep": time.sleep,
        "print": print,
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
    exec_globals: dict[str, Any] = {"__builtins__": safe_builtins}
    exec_globals.update(api)
    atomic_ops_source = _load_atomic_ops_source()
    if atomic_ops_source:
        exec(compile(atomic_ops_source, str(ATOMIC_OPS_PATH), "exec"), exec_globals)
    return exec_globals


def execute_code_phased_with_segment_judges(
    *,
    prologue: str,
    segments: list[dict[str, str]],
    task_prompt: str | None,
    atomic_task_summary: str,
    segment_judge: SegmentSuccessJudgeAgent,
    skip_judge_prefix: int = 0,
    save_video: bool = True,
    render_engine: str = "mujoco",
) -> tuple[bool, int | None, str, str | None]:
    """
    Run ``prologue`` then each phase ``body`` in one continuous rollout; after each
    phase, capture the table view and call ``SegmentSuccessJudgeAgent`` (stage judge).

    Returns ``(all_segments_ok, failed_index_or_none, last_judge_json, video_path_or_none)``.
    """
    recorder = ExecutionRecorder(
        RUNTIME,
        task_prompt=task_prompt,
        enabled=save_video,
        render_engine=render_engine,
    )
    segment_front_view_path = LOG_ROOT / "segment_judge_current.png"
    segment_wrist_view_path = LOG_ROOT / "segment_judge_wrist_current.png"
    segment_judge_json_path = LOG_ROOT / "segment_judge_current.json"
    RUNTIME._viewer_ready.wait(timeout=10)
    with RUNTIME.lock:
        RUNTIME._busy = True
    last_judge = "{}"
    failed_idx: int | None = None

    def _step(tag: str):
        with RUNTIME.lock:
            mujoco.mj_step(RUNTIME.model, RUNTIME.data)
        recorder.record(tag)

    try:
        g = _build_atomic_exec_globals_for_run(recorder, _step)
        if prologue.strip():
            _log("[stage judge] --- prologue (before first EVO_PHASE) ---\n" + prologue.strip())
            exec(compile(prologue, "<atomic_prologue>", "exec"), g)
        n_seg = len(segments)
        for idx, seg in enumerate(segments):
            body = (seg.get("body") or "").strip()
            slug_raw = str(seg.get("slug", f"p{idx}"))
            goal_raw = str(seg.get("goal", ""))
            marker = f"# === EVO_PHASE: {slug_raw} | {goal_raw} ==="
            _log(
                f"[stage judge] segment {idx + 1}/{n_seg} — {marker}\n"
                f"[stage judge] code body:\n{body if body else '(no statements; marker only)'}"
            )
            if body:
                exec(compile(body, f"<evoma_phase_{idx}>", "exec"), g)
            if idx == n_seg - 1:
                _log(
                    f"[stage judge] skip final restore segment {idx + 1}/{n_seg} "
                    f"(slug={slug_raw})"
                )
                continue
            if idx < max(0, int(skip_judge_prefix)):
                _log(
                    f"[stage judge] skip previously-approved segment {idx + 1}/{n_seg} "
                    f"(slug={slug_raw})"
                )
                continue
            called_names = _called_function_names(body)
            stage_atomic_ops_calls = sorted(
                name for name in called_names if name in _STAGE_JUDGE_ATOMIC_OPS_NAMES
            )
            if not stage_atomic_ops_calls:
                _log(
                    f"[stage judge] skip segment {idx + 1}/{n_seg} (slug={slug_raw}); "
                    "no evoma_atomic_ops.py function call detected"
                )
                continue
            primitive_calls = sorted(name for name in called_names if name in _PRIMITIVE_RUNTIME_API_NAMES)
            _save_front(segment_front_view_path)
            _save_wrist(segment_wrist_view_path)
            scoped_atomic_task_summary = (
                f"{atomic_task_summary}\n\n"
                "Stage judge scope: evaluate only the behavior semantics of functions defined in "
                "scripts/evoma_atomic_ops.py that are called in this stage.\n"
                f"Atomic-ops functions used in this stage: {', '.join(stage_atomic_ops_calls)}.\n"
                "Do not evaluate primitive runtime APIs (move_to, move_ee, gripper_control, ee_pose) "
                "as pass/fail criteria."
            )
            if primitive_calls:
                scoped_atomic_task_summary += (
                    f"\nPrimitive runtime API calls present (ignore for judging): {', '.join(primitive_calls)}."
                )
            ok, last_judge = segment_judge.judge_segment(
                phase_slug=str(seg.get("slug", f"p{idx}")),
                phase_goal=str(seg.get("goal", "")),
                atomic_task_summary=scoped_atomic_task_summary,
                image_path=str(segment_front_view_path),
                wrist_image_path=str(segment_wrist_view_path),
                json_out=segment_judge_json_path,
            )
            if not ok:
                _log(
                    f"[stage judge] FAIL at segment {idx + 1}/{n_seg} "
                    f"(slug={slug_raw}) stage goal: {goal_raw}"
                )
                failed_idx = idx
                break
    except Exception as exc:
        last_judge = json.dumps(
            {
                "verdict": "FAIL",
                "reason": f"segment execution error: {exc}",
                "analysis": "",
                "task_prompt": "phased_exec",
                "image_path": "",
            },
            ensure_ascii=False,
            indent=2,
        )
        failed_idx = failed_idx if failed_idx is not None else 0
    finally:
        with RUNTIME.lock:
            RUNTIME._busy = False
    video_path = recorder.close()
    return failed_idx is None, failed_idx, last_judge, video_path


def execute_code_with_recording(
    code: str,
    task_prompt: str | None = None,
    save_video: bool = True,
    render_engine: str = "mujoco",
) -> str | None:
    recorder = ExecutionRecorder(
        RUNTIME,
        task_prompt=task_prompt,
        enabled=save_video,
        render_engine=render_engine,
    )
    RUNTIME._viewer_ready.wait(timeout=10)
    with RUNTIME.lock:
        RUNTIME._busy = True
    try:
        def _step(tag: str):
            with RUNTIME.lock:
                mujoco.mj_step(RUNTIME.model, RUNTIME.data)
            recorder.record(tag)

        exec_globals = _build_atomic_exec_globals_for_run(recorder, _step)
        exec(code, exec_globals)
    finally:
        with RUNTIME.lock:
            RUNTIME._busy = False
    return recorder.close()


def _apply_robot_arm_perturbation(enable: bool):
    if not enable:
        return
    lows = np.array([-0.1, 0.0, -0.2, -0.1, 0.0, -0.2], dtype=float)
    highs = np.array([0.1, 0.3, 0.2, 0.1, 0.3, 0.2], dtype=float)
    with RUNTIME.lock:
        model = RUNTIME.model
        data = RUNTIME.data
        if model is None or data is None or RUNTIME.ee is None:
            return
        perturbation = np.random.uniform(lows, highs)
        data.qpos[RUNTIME.ee.jnt_span] += perturbation
        data.ctrl[RUNTIME.ee.act_span] += perturbation
        mujoco.mj_forward(model, data)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)


_BLENDER_DEPS_READY: dict[str, Path] = {}


def _discover_blender_python(blender_bin: str) -> Path | None:
    cmd = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python-expr",
        "import sys;print(sys.executable)",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        txt = line.strip()
        if txt.lower().endswith(("python.exe", "/python", "\\python")):
            p = Path(txt)
            if p.exists():
                return p
    return None


def _blender_deps_target_dir(blender_python: Path) -> Path:
    major, minor = sys.version_info[:2]
    match = re.search(r"python(\d+)\.(\d+)", str(blender_python).replace("\\", "/").lower())
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
    return PROJECT_ROOT / ".cache" / "blender_pydeps" / f"py{major}{minor}"


def _ensure_blender_python_deps(blender_bin: str) -> Path | None:
    ready = _BLENDER_DEPS_READY.get(blender_bin)
    if ready is not None and ready.exists():
        return ready
    blender_python = _discover_blender_python(blender_bin)
    if blender_python is None:
        _log("[Blender] failed to discover Blender bundled python")
        return None
    deps_dir = _blender_deps_target_dir(blender_python)
    deps_dir.mkdir(parents=True, exist_ok=True)
    check_env = os.environ.copy()
    check_env["PYTHONPATH"] = str(deps_dir)
    check_env["PYTHONNOUSERSITE"] = "1"
    check_cmd = [
        str(blender_python),
        "-c",
        "import mujoco, numpy, scipy, trimesh, zstandard; print('ok')",
    ]
    check_rc = subprocess.run(check_cmd, cwd=str(PROJECT_ROOT), env=check_env, check=False).returncode
    if check_rc == 0:
        _BLENDER_DEPS_READY[blender_bin] = deps_dir
        return deps_dir

    _log(f"[Blender] installing Python deps into project cache: {deps_dir}")
    pip_cmd = [
        str(blender_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(deps_dir),
        "numpy<2.5",
        "mujoco==3.3.0",
        "scipy",
        "trimesh",
        "zstandard",
    ]
    pip_rc = subprocess.run(pip_cmd, cwd=str(PROJECT_ROOT), env=check_env, check=False).returncode
    if pip_rc != 0:
        _log(f"[Blender] dependency installation failed (rc={pip_rc})")
        return None
    check_rc = subprocess.run(check_cmd, cwd=str(PROJECT_ROOT), env=check_env, check=False).returncode
    if check_rc != 0:
        _log("[Blender] dependency verification failed after installation")
        return None
    _BLENDER_DEPS_READY[blender_bin] = deps_dir
    return deps_dir


class BlenderFrameRenderer:
    """Persistent Blender subprocess that renders camera frames on demand.

    The worker is launched once per :class:`ExecutionRecorder` (i.e. per
    rollout) so we pay Blender's startup cost a single time. Each ``record``
    pushes the current ``qpos`` to the worker over stdin and blocks until the
    worker writes the rendered PNG(s) and returns ``OK``.
    """

    PROTOCOL_PREFIX = "<<RWORKER>>"
    READY_TIMEOUT_S = 180.0
    RESPONSE_TIMEOUT_S = 240.0

    def __init__(
        self,
        scene_xml: Path,
        *,
        width: int,
        height: int,
        main_camera: str = "table_cam_front",
        wrist_camera: str = "/ur:wrist_cam",
    ):
        self.scene_xml = Path(scene_xml)
        self.width = int(width)
        self.height = int(height)
        self.main_camera = str(main_camera)
        self.wrist_camera = str(wrist_camera)
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        blender_bin = os.getenv("BLENDER_BIN", "blender").strip() or "blender"
        if shutil.which(blender_bin) is None:
            raise RuntimeError(
                f"Blender executable not found ({blender_bin}); "
                "set BLENDER_BIN or install Blender to use --render-engine blender"
            )
        deps_dir = _ensure_blender_python_deps(blender_bin)
        if deps_dir is None:
            raise RuntimeError("Required Blender Python deps unavailable for live renderer")
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"Scene XML for Blender worker does not exist: {self.scene_xml}")

        script_path = PROJECT_ROOT / "scripts" / "render_blender.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(deps_dir)
        env["PYTHONNOUSERSITE"] = "1"
        cmd = [
            blender_bin,
            "--background",
            "--python",
            str(script_path),
            "--",
            "--worker",
            "--source",
            str(self.scene_xml),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--camera",
            self.main_camera,
            "--wrist-camera",
            self.wrist_camera,
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )

        # Drain stderr in background; forward each line via _log to keep the worker visible.
        def _pump_stderr(proc: subprocess.Popen) -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                msg = line.rstrip()
                if msg:
                    _log(f"[Blender worker] {msg}")

        self._stderr_thread = threading.Thread(target=_pump_stderr, args=(self._proc,), daemon=True)
        self._stderr_thread.start()

        self._wait_for_marker("READY", timeout=self.READY_TIMEOUT_S, context="startup")
        _log(
            f"[Blender] live worker ready (scene={self.scene_xml.name}, "
            f"size={self.width}x{self.height})"
        )

    def _wait_for_marker(self, expected: str, *, timeout: float, context: str) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Blender worker stdout closed during {context} (rc={rc}) before '{expected}'"
                )
            text = line.rstrip()
            if not text.startswith(self.PROTOCOL_PREFIX):
                continue
            body = text[len(self.PROTOCOL_PREFIX) :].strip()
            if body == expected:
                return
            if body.startswith("ERR"):
                raise RuntimeError(f"Blender worker error during {context}: {body}")
            # Other markers (e.g. BYE) are unexpected here; surface them.
            raise RuntimeError(
                f"Unexpected Blender worker marker during {context}: '{body}' (wanted '{expected}')"
            )
        raise TimeoutError(f"Blender worker did not emit '{expected}' within {timeout:.0f}s ({context})")

    def render(
        self,
        qpos: np.ndarray,
        front_path: Path | None,
        wrist_path: Path | None = None,
    ) -> None:
        if self._proc is None or self._proc.poll() is not None or self._proc.stdin is None:
            raise RuntimeError("Blender worker is not running")
        request = {
            "cmd": "render",
            "qpos": [float(x) for x in np.asarray(qpos, dtype=np.float64).ravel().tolist()],
            "front": str(front_path) if front_path is not None else "",
            "wrist": str(wrist_path) if wrist_path is not None else "",
        }
        line = f"{self.PROTOCOL_PREFIX} {json.dumps(request, ensure_ascii=False)}\n"
        with self._stdout_lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            self._wait_for_marker("OK", timeout=self.RESPONSE_TIMEOUT_S, context="render")

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(
                        f"{self.PROTOCOL_PREFIX} {json.dumps({'cmd': 'exit'})}\n"
                    )
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass


def _enhance_run_with_blender(run_dir: Path, fps: int) -> bool:
    """Render run frames with Blender and rebuild execution.mp4 via ffmpeg."""
    blender_bin = os.getenv("BLENDER_BIN", "blender").strip() or "blender"
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
    if shutil.which(blender_bin) is None:
        _log(f"[Blender] skip: executable not found ({blender_bin})")
        return False
    if shutil.which(ffmpeg_bin) is None:
        _log(f"[Blender] skip: executable not found ({ffmpeg_bin})")
        return False
    deps_dir = _ensure_blender_python_deps(blender_bin)
    if deps_dir is None:
        _log("[Blender] skip: required Python deps unavailable in Blender runtime")
        return False

    qpos_path = run_dir / "qpos.npy"
    scene_xml = run_dir / "scene.xml"
    model_mjb = run_dir / "model.mjb"
    scene_source_dir = run_dir / "scene_source"
    if model_mjb.exists():
        source_for_blender = run_dir
    else:
        source_for_blender = scene_source_dir if (scene_source_dir / "scene.xml").exists() else run_dir
    if not qpos_path.exists() or not scene_xml.exists():
        _log(f"[Blender] skip: missing scene.xml/qpos.npy under {run_dir}")
        return False

    script_path = PROJECT_ROOT / "scripts" / "render_blender.py"
    scene_blend = run_dir / "scene.blend"
    enhanced_frames = run_dir / "frames_enhanced"
    enhanced_frames.mkdir(parents=True, exist_ok=True)
    render_engine = (os.getenv("BLENDER_RENDER_ENGINE", "BLENDER_EEVEE_NEXT").strip() or "BLENDER_EEVEE_NEXT").upper()
    fast_mode = str(os.getenv("BLENDER_FAST_MODE", "1")).strip().lower() in {"1", "true", "yes", "on"}
    render_width = 512
    render_height = 512
    seed_frame = run_dir / "frames" / "000000.png"
    if seed_frame.exists():
        try:
            seed_img = imageio.imread(seed_frame)
            if seed_img.ndim >= 2:
                render_height = int(seed_img.shape[0])
                render_width = int(seed_img.shape[1])
        except Exception:
            pass
    blender_env = os.environ.copy()
    blender_env["PYTHONPATH"] = str(deps_dir)
    blender_env["PYTHONNOUSERSITE"] = "1"

    build_blend_cmd = [
        blender_bin,
        "--background",
        "--python",
        str(script_path),
        "--",
        str(source_for_blender),
        "--qpos-npy",
        str(qpos_path),
        "--blend-out",
        str(scene_blend),
        "--blend-only",
        "--fps",
        str(int(max(1, fps))),
        "--width",
        str(int(max(64, render_width))),
        "--height",
        str(int(max(64, render_height))),
        "--camera",
        "table_cam_front",
    ]
    rc = subprocess.run(build_blend_cmd, cwd=str(PROJECT_ROOT), env=blender_env, check=False).returncode
    if rc != 0:
        _log(f"[Blender] scene.blend generation failed (rc={rc})")
        return False
    if not scene_blend.exists() or scene_blend.stat().st_size <= 0:
        _log(f"[Blender] scene.blend missing after generation: {scene_blend}")
        return False

    speed_tune_expr = ""
    if fast_mode:
        speed_tune_expr = (
            "import bpy;"
            "s=bpy.context.scene;"
            "e=str(s.render.engine);"
            "s.render.use_simplify=True;"
            "s.render.simplify_subdivision=0;"
            "s.render.simplify_child_particles=0.0;"
            "s.render.resolution_percentage=100;"
            "s.cycles.samples=16 if hasattr(s,'cycles') else 0;"
            "setattr(s.cycles,'use_denoising',False) if hasattr(s,'cycles') else None;"
            "setattr(s.cycles,'max_bounces',2) if hasattr(s,'cycles') else None;"
            "setattr(s.eevee,'taa_render_samples',8) if hasattr(s,'eevee') else None;"
            "setattr(s.eevee,'use_gtao',False) if hasattr(s,'eevee') else None;"
            "setattr(s.eevee,'use_bloom',False) if hasattr(s,'eevee') else None;"
            "setattr(s.eevee,'use_ssr',False) if hasattr(s,'eevee') else None;"
        )

    render_cmd = [
        blender_bin,
        "-b",
        str(scene_blend),
        "--render-output",
        str(enhanced_frames / "####.png"),
        "--engine",
        render_engine,
    ]
    if speed_tune_expr:
        render_cmd.extend(["--python-expr", speed_tune_expr])
    render_cmd.extend([
        "--render-anim",
        "--render-format",
        "PNG",
    ])
    rc = subprocess.run(render_cmd, cwd=str(PROJECT_ROOT), env=blender_env, check=False).returncode
    if rc != 0:
        _log(f"[Blender] render-anim failed (rc={rc})")
        return False

    src_frames = sorted(enhanced_frames.glob("*.png"))
    if not src_frames:
        _log("[Blender] render-anim produced no frames")
        return False
    final_frames = run_dir / "frames"
    final_frames.mkdir(parents=True, exist_ok=True)
    for old in final_frames.glob("*.png"):
        old.unlink(missing_ok=True)
    for idx, src in enumerate(src_frames):
        shutil.copy2(src, final_frames / f"{idx:06d}.png")

    mp4_path = run_dir / "execution.mp4"
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(int(max(1, fps))),
        "-i",
        str(enhanced_frames / "%04d.png"),
        "-filter:v",
        "format=rgba,premultiply=inplace=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    rc = subprocess.run(ffmpeg_cmd, cwd=str(PROJECT_ROOT), check=False).returncode
    if rc != 0:
        _log(f"[Blender] ffmpeg compose failed (rc={rc})")
        return False
    return True


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _normalize_task_prompt(prompt: str | None) -> str:
    text = " ".join((prompt or "").replace("\x00", " ").split()).strip()
    if not text:
        return "manual_execution"
    return text.replace("/", "_").replace("\\", "_")[:120]


def _normalize_saved_subtask_prompt(prompt: str | None) -> str:
    text = " ".join((prompt or "").replace("\x00", " ").split()).strip().lower()
    if not text:
        return "manual_execution"
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "manual_execution"


def _sanitize_mj_name(text: str | None, fallback: str = "asset") -> str:
    raw = " ".join((text or "").replace("\x00", " ").split()).strip()
    if not raw:
        return fallback
    name = re.sub(r"[^a-zA-Z0-9_./:-]+", "_", raw)
    name = name.replace("/", "_")
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        return fallback
    if not re.match(r"^[a-zA-Z_]", name):
        name = f"{fallback}_{name}"
    return name


def _repo_id_from_prompt(prompt: str | None) -> str:
    text = _normalize_task_prompt(prompt).lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "manual_execution"


def _extract_task_description_from_code(code: str | None) -> str | None:
    if not code:
        return None
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Prefer first non-empty line in comments like: "# place xxx"
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        line = line.strip("[](){} ").strip()
        return line or None
    return None


def _safe_load_scene(scene_json_path: Path, randomize_texture: bool = True, cleanup_old_cache: bool = True):
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON does not exist: {scene_json_path}")
    RUNTIME.load_scene_config(scene_json_path, randomize_texture=randomize_texture, cleanup_old_cache=cleanup_old_cache)


def _save_front(path: Path):
    RUNTIME.save_front_image(path)


def _save_wrist(path: Path):
    RUNTIME.save_wrist_image(path)


def _cleanup_run_dir(run_dir: Path | None) -> None:
    if run_dir is None:
        return
    try:
        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir)
            _log(f"[Cleanup] removed failed run dir: {run_dir}")
    except Exception as err:
        _log(f"[Cleanup] failed to remove run dir {run_dir}: {err}")


def _prepare_report_for_attempt(
    pipeline: EvoMAAgentPipeline,
    task_prompt: str,
    scene_json_path: Path,
) -> str:
    # Randomize texture once at the start of each attempt before report generation.
    # cleanup_old_cache=True to clear any previous cached scene and generate a fresh one.
    _safe_load_scene(scene_json_path, randomize_texture=True, cleanup_old_cache=True)
    _apply_robot_arm_perturbation(True)
    RUNTIME.save_scene_config(PRE_SCENE_CONFIG)
    _save_front(LOG_ROOT / "current_view.png")
    report = pipeline.run_report_generation(task_prompt)
    _log(f"Attempt report generated at: {LOG_ROOT / 'report.txt'}")
    return report


def _prepare_scene_perturbations_only(scene_json_path: Path) -> None:
    """Randomize background (texture) + arm perturbation without calling the report LLM."""
    _safe_load_scene(scene_json_path, randomize_texture=True, cleanup_old_cache=True)
    _apply_robot_arm_perturbation(True)
    RUNTIME.save_scene_config(PRE_SCENE_CONFIG)
    _save_front(LOG_ROOT / "current_view.png")


def _format_regeneration_feedback(judge_text: str, attempted_code: str | None) -> str:
    """User-visible hint for AtomicActionSkill after a failed judge on reused or fresh code."""
    code_block = (attempted_code or "").strip()
    judge_block = format_judge_feedback_for_atomic_skill(judge_text)
    return (
        "Last call failed: the simulation execution was judged unsuccessful. "
        "Revise the atomic-action code using the judge **reason** and **analysis** in the section below, "
        "together with the previous code. Keep scene and atomic-task constraints consistent.\n\n"
        f"{judge_block}\n\n"
        f"Previous executed code:\n```python\n{code_block}\n```\n"
    )


def _merge_atomic_feedback(*parts: str | None) -> str | None:
    texts = [str(p).strip() for p in parts if p and str(p).strip()]
    return "\n\n---\n\n".join(texts) if texts else None


def _first_atomic_task_json_text() -> str:
    if not EVOMA_PLAN_JSON.exists():
        return ""
    try:
        plan = json.loads(EVOMA_PLAN_JSON.read_text(encoding="utf-8"))
        tasks = TaskSupervisorAgent._atomic_tasks_from_payload(plan)
        if not tasks:
            return ""
        return json.dumps(tasks[0], ensure_ascii=False, indent=2)
    except Exception:
        return ""


def _segment_signature(seg: dict[str, str]) -> str:
    raw = (
        str(seg.get("slug", "")),
        str(seg.get("goal", "")),
        str(seg.get("body", "")).strip(),
    )
    joined = "\n<SEP>\n".join(raw)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _shared_prefix_len(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _run_once(
    pipeline: EvoMAAgentPipeline,
    judge: TaskSuccessJudgeAgent,
    task_prompt: str,
    scene_json_path: Path,
    *,
    fixed_code: str | None = None,
    failure_feedback: str | None = None,
    enable_judge: bool = True,
    render_engine: str = "mujoco",
) -> tuple[bool, Path | None, str, str | None]:
    """Run execute + judge. If ``fixed_code`` is set, skip LLM atomic generation and reuse that snippet."""
    run_dir = None
    code_used: str | None = None
    max_seg = max(1, int(os.getenv("EVOMA_MAX_SEGMENT_REGEN", "8")))
    segment_judge = SegmentSuccessJudgeAgent(client=judge.client, model=judge.model) if EVOMA_ENABLE_STAGE_JUDGE else None
    try:
        # Keep texture fixed after outer perturbation / report prep to avoid report/actioncode mismatch.
        _safe_load_scene(scene_json_path, randomize_texture=False, cleanup_old_cache=False)

        # Match the default perturbation behavior in Gradio auto-run.
        _apply_robot_arm_perturbation(True)

        RUNTIME.save_scene_config(PRE_SCENE_CONFIG)
        _save_front(LOG_ROOT / "current_view.png")

        already_executed = False

        if fixed_code is not None and str(fixed_code).strip():
            code_used = str(fixed_code).strip()
            (LOG_ROOT / "atomic_actions.py").write_text(code_used, encoding="utf-8")
        else:
            if not (LOG_ROOT / "report.txt").exists():
                raise RuntimeError("Missing logs/report.txt. Please generate report once before loop.")

            segment_hint: str | None = failure_feedback
            approved_sig_prefix: list[str] = []
            if not EVOMA_ENABLE_STAGE_JUDGE:
                gen = pipeline.run_atomic_action_generation(failure_feedback=segment_hint)
                if not gen or not str(gen).strip():
                    raise RuntimeError("Atomic action generation returned empty code.")
                code_used = str(gen).strip()
                (LOG_ROOT / "atomic_actions.py").write_text(code_used, encoding="utf-8")
            else:
                for seg_try in range(max_seg):
                # Hard reset scene before each segment-fix regeneration trial so execution
                # always restarts from a freshly reloaded task state.
                    _safe_load_scene(scene_json_path, randomize_texture=False, cleanup_old_cache=False)
                    _apply_robot_arm_perturbation(True)
                    RUNTIME.save_scene_config(PRE_SCENE_CONFIG)
                    _save_front(LOG_ROOT / "current_view.png")

                    gen = pipeline.run_atomic_action_generation(failure_feedback=segment_hint)
                    if not gen or not str(gen).strip():
                        raise RuntimeError("Atomic action generation returned empty code.")
                    code_used = str(gen).strip()
                    (LOG_ROOT / "atomic_actions.py").write_text(code_used, encoding="utf-8")

                    prologue_try, segments_try = parse_evo_phase_segments(code_used)
                    if not segments_try:
                        break
                    sigs_try = [_segment_signature(s) for s in segments_try]
                    skip_prefix = _shared_prefix_len(approved_sig_prefix, sigs_try)

                    atomic_summary = _first_atomic_task_json_text()
                    sub_raw_try = _extract_task_description_from_code(code_used) or task_prompt
                    sub_norm_try = _normalize_saved_subtask_prompt(sub_raw_try)
                    summary_line = atomic_summary or sub_raw_try

                    ok_phased, fail_i, seg_judge_txt, video_path = execute_code_phased_with_segment_judges(
                        prologue=prologue_try,
                        segments=segments_try,
                        task_prompt=sub_norm_try,
                        atomic_task_summary=summary_line,
                        segment_judge=segment_judge,
                        skip_judge_prefix=skip_prefix,
                        save_video=True,
                        render_engine=render_engine,
                    )
                    if ok_phased:
                        approved_sig_prefix = sigs_try
                        # Keep this successful phased rollout data; no need to replay.
                        already_executed = bool(video_path)
                        if video_path:
                            run_dir = Path(video_path).parent
                        break

                    _log(f"Stage judge failed segment={fail_i}; atomic regen {seg_try + 1}/{max_seg}")
                    if video_path:
                        failed_run_dir = Path(video_path).parent
                        src_mp4 = failed_run_dir / "execution.mp4"
                        if src_mp4.is_file():
                            LOG_ROOT.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_mp4, LOG_ROOT / "fail.mp4")
                        _cleanup_run_dir(failed_run_dir)
                    fi = int(fail_i) if fail_i is not None and fail_i >= 0 else 0
                    approved_sig_prefix = sigs_try[:fi]
                    segment_hint = _merge_atomic_feedback(
                        failure_feedback,
                        format_segment_failure_for_atomic_skill(
                            full_code=code_used,
                            failed_index=fi,
                            segments=segments_try,
                            judge_text=seg_judge_txt,
                        ),
                    )
                else:
                    raise RuntimeError(
                        f"Exceeded EVOMA_MAX_SEGMENT_REGEN={max_seg} segment-level retries without phased success."
                    )

        subtask_prompt_raw = _extract_task_description_from_code(code_used) or task_prompt
        subtask_prompt_norm = _normalize_saved_subtask_prompt(subtask_prompt_raw)

        prologue, segments = parse_evo_phase_segments(code_used or "")
        if segments and EVOMA_ENABLE_STAGE_JUDGE and segment_judge is not None:
            if not already_executed:
                atomic_summary = _first_atomic_task_json_text()
                summary_line = atomic_summary or subtask_prompt_raw
                ok_phased, fail_i, seg_judge_txt, video_path = execute_code_phased_with_segment_judges(
                    prologue=prologue,
                    segments=segments,
                    task_prompt=subtask_prompt_norm,
                    atomic_task_summary=summary_line,
                    segment_judge=segment_judge,
                    save_video=True,
                    render_engine=render_engine,
                )
                if not ok_phased:
                    if video_path:
                        run_dir = Path(video_path).parent
                        src_mp4 = run_dir / "execution.mp4"
                        if src_mp4.is_file():
                            LOG_ROOT.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_mp4, LOG_ROOT / "fail.mp4")
                    return False, run_dir, seg_judge_txt, code_used
                if video_path:
                    run_dir = Path(video_path).parent
        elif code_used and code_used.strip():
            video_path = execute_code_with_recording(
                code_used,
                task_prompt=subtask_prompt_norm,
                save_video=True,
                render_engine=render_engine,
            )
            if video_path:
                run_dir = Path(video_path).parent

        RUNTIME.save_scene_config(FINISH_SCENE_CONFIG)
        finished_front = LOG_ROOT / "finished_view.png"
        finished_wrist = LOG_ROOT / "finished_view_wrist.png"
        _save_front(finished_front)
        _save_wrist(finished_wrist)

        judge_task_prompt = subtask_prompt_raw
        if bool(enable_judge):
            success, judge_text = judge.judge(
                task_prompt=judge_task_prompt,
                image_path=str(finished_front),
                wrist_image_path=str(finished_wrist),
            )
        else:
            success, judge_text = True, "SKIPPED: judge disabled by --no-judge"
        if not success and run_dir is not None:
            src_mp4 = run_dir / "execution.mp4"
            if src_mp4.is_file():
                LOG_ROOT.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_mp4, LOG_ROOT / "fail.mp4")
        return success, run_dir, judge_text, code_used
    except Exception:
        _cleanup_run_dir(run_dir)
        raise


def _convert_runs_to_lerobot(
    run_dirs: list[Path],
    repo_id: str,
    lerobot_root: Path,
    task_name: str,
    overwrite: bool,
) -> Path:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as err:
        raise RuntimeError(f"Import LeRobotDataset failed: {err}") from err

    if not run_dirs:
        raise RuntimeError("No successful run dirs to convert")

    sample_payload = json.loads((run_dirs[0] / "state_action.json").read_text(encoding="utf-8"))
    sample_records = sample_payload.get("records", [])
    if not sample_records:
        raise RuntimeError(f"Empty recording: {run_dirs[0]}")

    sample_frames_dir = run_dirs[0] / "frames"
    first_frame_path = sample_frames_dir / sample_records[0]["frame"]
    first_img = imageio.imread(first_frame_path)
    if first_img.ndim != 3:
        raise RuntimeError(f"Unexpected frame dim: {first_img.shape}")
    if first_img.shape[2] > 3:
        first_img = first_img[:, :, :3]
    image_shape = tuple(first_img.shape)

    state_dim = len(sample_records[0]["state"])
    action_dim = len(sample_records[0]["action"])

    times = [float(r.get("time", 0.0)) for r in sample_records]
    dt = None
    if len(times) >= 2:
        deltas = [times[i + 1] - times[i] for i in range(len(times) - 1) if times[i + 1] > times[i]]
        if deltas:
            dt = float(np.median(np.asarray(deltas)))
    raw_fps = int(round(1.0 / dt)) if dt and dt > 1e-6 else 20
    fps = int(np.clip(raw_fps, 1, 240))

    output_path = lerobot_root / repo_id
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output path exists: {output_path}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
        "observation.images.base_0_rgb": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_wrist_0_rgb": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist_0_rgb": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    for idx, run_dir in enumerate(run_dirs, start=1):
        payload = json.loads((run_dir / "state_action.json").read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            _log(f"[Convert] skip empty run: {run_dir}")
            continue
        frames_dir = run_dir / "frames"
        wrist_frames_dir = run_dir / "wrist_frames"
        current_task = task_name or payload.get("task_prompt") or "manual_execution"
        for rec in records:
            frame_path = frames_dir / rec["frame"]
            img = imageio.imread(frame_path)
            if img.shape[2] > 3:
                img = img[:, :, :3]

            wrist_name = rec.get("wrist_frame", rec["frame"])
            wrist_path = wrist_frames_dir / wrist_name

            wrist_img = imageio.imread(wrist_path) if wrist_path.exists() else img
            if wrist_img.shape[2] > 3:
                wrist_img = wrist_img[:, :, :3]

            frame = {
                "observation.state": np.asarray(rec["state"], dtype=np.float32),
                "action": np.asarray(rec["action"], dtype=np.float32),
                "observation.images.base_0_rgb": img.astype(np.uint8),
                "observation.images.left_wrist_0_rgb": wrist_img.astype(np.uint8),
                "observation.images.right_wrist_0_rgb": wrist_img.astype(np.uint8),
                "task": str(current_task),
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        _log(f"[Convert] episode {idx}/{len(run_dirs)} done: {run_dir.name}")

    return output_path


def _validate_success_run_videos(run_dirs: list[Path]) -> None:
    """Ensure each successful run folder contains a non-empty MP4 before conversion."""
    missing_or_empty: list[str] = []
    for run_dir in run_dirs:
        mp4_path = run_dir / "execution.mp4"
        if not mp4_path.exists() or not mp4_path.is_file() or mp4_path.stat().st_size <= 0:
            missing_or_empty.append(str(mp4_path))
    if missing_or_empty:
        raise RuntimeError(
            "Missing or empty MP4 in successful run folders; "
            f"conversion aborted. Offending files: {missing_or_empty}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoBody automatic trajectory generation CLI")
    parser.add_argument("--scene-json", required=True, help="Path to scene json file")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--success-target", type=int, default=1, help="Number of successful runs to collect")
    parser.add_argument("--max-attempts", type=int, default=0, help="Max attempts; 0 means unlimited")
    parser.add_argument(
        "--repo-id",
        default="",
        help="LeRobot repo_id; if empty, uses logs/evoma_plan.json: <primitive>/<summary_slug>",
    )
    parser.add_argument("--lerobot-root", default=str(LEROBOT_HOME), help="LeRobot dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dataset if exists")
    parser.add_argument("--result-json-out", default="", help="Optional path to write generation summary json")
    parser.add_argument(
        "--judge",
        dest="enable_judge",
        action="store_true",
        default=True,
        help="Enable VLM judge for generation filtering (default: enabled)",
    )
    parser.add_argument(
        "--no-judge",
        dest="enable_judge",
        action="store_false",
        help="Disable VLM judge and accept each run as success if artifacts exist",
    )
    parser.add_argument(
        "--render-engine",
        choices=("mujoco", "blender"),
        default="mujoco",
        help="Render engine for execution video (default: mujoco)",
    )
    return parser.parse_args()


def _write_result_json(path_text: str | None, payload: dict[str, Any]) -> None:
    if not path_text or not path_text.strip():
        return
    out_path = Path(path_text.strip()).expanduser()
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    # Create logs directory if it doesn't exist
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    
    args = parse_args()

    scene_json_path = Path(args.scene_json).expanduser()
    if not scene_json_path.is_absolute():
        scene_json_path = PROJECT_ROOT / scene_json_path

    api_base_url = _require_env("API_BASE_URL")
    api_key = _require_env("API_KEY")
    model_name = _require_env("MODEL_NAME")

    os.environ["API_BASE_URL"] = api_base_url
    os.environ["API_KEY"] = api_key
    os.environ["MODEL_NAME"] = model_name

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    pipeline = EvoMAAgentPipeline(client=client, model=model_name)
    judge = TaskSuccessJudgeAgent(client=client, model=model_name)

    success_target = max(1, int(args.success_target))
    max_attempts = int(args.max_attempts)
    lerobot_root = Path(args.lerobot_root).expanduser()

    _log(f"Start auto generation: target_success={success_target}, model={model_name}")
    _log(f"Scene JSON: {scene_json_path}")
    _log(f"Stage judge enabled: {EVOMA_ENABLE_STAGE_JUDGE}")
    _log(f"Task judge enabled: {bool(args.enable_judge)}")
    _log(f"Render engine: {args.render_engine}")

    successes = 0
    attempts = 0
    successful_runs: list[Path] = []
    # After at least one judge success with agent-produced code, reuse it with scene/arm perturbations only
    # until judge fails; then pass failure context back to AtomicActionSkill.
    locked_success_code: str | None = None
    pending_regeneration_hint: str | None = None

    while successes < success_target:
        if max_attempts > 0 and attempts >= max_attempts:
            _log(f"Reached max attempts: {max_attempts}")
            break

        attempts += 1
        regen_hint = pending_regeneration_hint

        try:
            if locked_success_code:
                _log(
                    f"Attempt {attempts}: reusing last successful code (background + arm perturbation only; "
                    "no report / no atomic LLM)"
                )
                _prepare_scene_perturbations_only(scene_json_path)
            else:
                _log(f"Attempt {attempts}: generating report with randomized texture")
                _prepare_report_for_attempt(
                    pipeline=pipeline,
                    task_prompt=args.task,
                    scene_json_path=scene_json_path,
                )
        except Exception as err:
            _log(f"Attempt {attempts} report / scene prep failed: {err}")
            RUNTIME._cleanup_cached_build_runtime_xml()
            continue

        _log(f"Attempt {attempts}: execute + judge")
        try:
            success, run_dir, judge_text, code_used = _run_once(
                pipeline=pipeline,
                judge=judge,
                task_prompt=args.task,
                scene_json_path=scene_json_path,
                fixed_code=locked_success_code,
                failure_feedback=regen_hint if locked_success_code is None else None,
                enable_judge=bool(args.enable_judge),
                render_engine=str(args.render_engine),
            )
        except Exception as err:
            err_text = str(err)
            _log(f"Attempt {attempts} failed by exception: {err_text}")
            # Clean up cached scene when code execution fails
            RUNTIME._cleanup_cached_build_runtime_xml()
            if "OpenGL platform library has not been loaded" in err_text or "mjr_makeContext" in err_text:
                _log(
                    "Fatal OpenGL initialization error. Stop retrying to avoid process abort. "
                    "Please verify MUJOCO_GL backend (egl/osmesa) and system GL dependencies."
                )
                sys.exit(4)
            continue

        try:
            j = json.loads(judge_text) if judge_text else {}
            head = f"{j.get('verdict', '')}: {(j.get('reason') or '')[:160]}"
        except Exception:
            head = judge_text.splitlines()[0] if judge_text else ""
        _log(f"Attempt {attempts} judge result: {head}")

        if success and run_dir and (run_dir / "state_action.json").exists() and (run_dir / "frames").exists():
            successes += 1
            successful_runs.append(run_dir)
            locked_success_code = code_used or locked_success_code
            pending_regeneration_hint = None
            _log(f"Success collected: {successes}/{success_target}; run={run_dir}")
        elif success:
            _log("Judge says SUCCESS but run data missing; not counted")
            _cleanup_run_dir(run_dir)
        else:
            _log(f"Attempt {attempts} not successful. {judge_text}")
            _cleanup_run_dir(run_dir)
            pending_regeneration_hint = _format_regeneration_feedback(judge_text, code_used)
            locked_success_code = None

        # Clean up cached scene after attempt completes (success or failure for judge)
        RUNTIME._cleanup_cached_build_runtime_xml()

    if successes < success_target:
        _log(f"Finished without enough successes: {successes}/{success_target}")
        _write_result_json(
            args.result_json_out,
            {
                "status": "FAIL",
                "reason": "insufficient_successes",
                "task": args.task,
                "scene_json": str(scene_json_path),
                "attempts": attempts,
                "successes": successes,
                "success_target": success_target,
                "created_at": _now(),
                "enable_judge": bool(args.enable_judge),
            },
        )
        sys.exit(2)

    _log("All target successes collected, converting to LeRobot dataset")
    repo_id = args.repo_id.strip() or _lerobot_repo_id_from_plan() or _repo_id_from_prompt(args.task)
    lerobot_task = _lerobot_task_prompt_from_plan(args.task)
    _log(f"LeRobot repo_id={repo_id!r}, dataset task from plan summary where available")
    try:
        _validate_success_run_videos(successful_runs)
        out = _convert_runs_to_lerobot(
            run_dirs=successful_runs,
            repo_id=repo_id,
            lerobot_root=lerobot_root,
            task_name=lerobot_task,
            overwrite=bool(args.overwrite),
        )
    except Exception as err:
        _log(f"Convert failed: {err}")
        _write_result_json(
            args.result_json_out,
            {
                "status": "FAIL",
                "reason": "convert_failed",
                "error": str(err),
                "task": args.task,
                "scene_json": str(scene_json_path),
                "attempts": attempts,
                "successes": successes,
                "success_target": success_target,
                "created_at": _now(),
                "enable_judge": bool(args.enable_judge),
            },
        )
        sys.exit(3)

    _write_result_json(
        args.result_json_out,
        {
            "status": "SUCCESS",
            "task": args.task,
            "lerobot_task": lerobot_task,
            "scene_json": str(scene_json_path),
            "attempts": attempts,
            "successes": successes,
            "success_target": success_target,
            "repo_id": repo_id,
            "dataset_path": str(out),
            "successful_runs": [str(p) for p in successful_runs],
            "created_at": _now(),
            "enable_judge": bool(args.enable_judge),
        },
    )

    _log(f"Done. attempts={attempts}, successes={successes}, dataset={out}")


if __name__ == "__main__":
    main()
