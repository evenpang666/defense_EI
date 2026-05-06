import argparse
import json
import os
import re
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mujoco_render_env

mujoco_render_env.ensure_mujoco_gl_environment()

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from openai import OpenAI



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
MODEL_ROOT = PROJECT_ROOT / "model"
DEFAULT_SCENE_JSON = LOG_ROOT / "saved_scene.json"
DEFAULT_EVAL_ROOT = LOG_ROOT / "pi05_eval_cli"
DEFAULT_BANK_DIR = PROJECT_ROOT / "checkpoints" / "pi05_adapter_bank_pt"
DEFAULT_BASE_MODEL_PATH = "checkpoints/pi05_base"
EVAL_CURRENT_IMAGE_NAME = "current_view_eval.png"
EVAL_FINISHED_IMAGE_NAME = "finished_view_eval.png"
EVAL_FINISHED_WRIST_IMAGE_NAME = "finished_view_wrist_eval.png"
EVAL_PRE_SCENE_NAME = "pre_scene_eval.json"
EVAL_FINISHED_SCENE_NAME = "finished_scene_eval.json"


@dataclass
class LoadedScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene_json_path: Path
    scene_xml_path: Path


@dataclass
class BuildAssetDef:
    key: str
    xml_path: Path
    root_body_name: str


class ContinualPolicyLeRobot:
    def __init__(
        self,
        config_name: str,
        base_model_path: Path | str,
        bank_dir: Path,
        checkpoint_assets_dir: Path | str,
        use_norm_stats: bool,
        num_steps: int,
        default_prompt: str | None,
    ):
        del config_name, bank_dir, checkpoint_assets_dir, use_norm_stats
        self._num_steps = int(num_steps)
        self._default_prompt = default_prompt or ""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lerobot_src = PROJECT_ROOT / "third_party" / "lerobot" / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))

        try:
            # LeRobot 0.4.x exposes PreTrainedConfig under lerobot.configs.policies.
            from lerobot.configs.policies import PreTrainedConfig
        except ImportError:
            try:
                # Backward compatibility for older LeRobot releases.
                from lerobot.configs import PreTrainedConfig
            except ImportError:
                # Compatibility fallback for alternative package layouts.
                from lerobot.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.constants import OBS_STATE

        resolved_base_model_path = _resolve_base_model_ref(base_model_path)
        model_ref = str(resolved_base_model_path)
        cfg = PreTrainedConfig.from_pretrained(model_ref)
        cfg.device = str(self._device)
        cfg.n_action_steps = int(self._num_steps)
        self._policy_type = str(getattr(cfg, "type", "")).strip().lower()
        policy_cls = get_policy_class(cfg.type)
        self._policy = policy_cls.from_pretrained(pretrained_name_or_path=model_ref, config=cfg)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=model_ref,
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )
        self._obs_state_key = OBS_STATE
        self._visual_keys = [k for k in cfg.input_features.keys() if "observation.image" in str(k)]

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        front = np.asarray(obs.get("observation/image"), dtype=np.uint8)
        wrist = np.asarray(obs.get("observation/wrist_image", front), dtype=np.uint8)
        state = np.asarray(obs.get("observation/state"), dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]

        def _to_bchw_image(img: np.ndarray) -> torch.Tensor:
            if img.ndim != 3:
                raise ValueError(f"Expected HWC image, got shape={img.shape}")
            # Policies consume BCHW tensors; SmolVLA resize path requires floating tensors.
            chw = np.transpose(img, (2, 0, 1))
            if self._policy_type == "smolvla":
                tensor = torch.as_tensor(chw[None, ...], dtype=torch.float32)
                # MuJoCo render frames are uint8 [0,255]; normalize only when needed.
                if float(tensor.max().item()) > 1.0:
                    tensor = tensor / 255.0
                return tensor
            return torch.as_tensor(chw[None, ...], dtype=torch.uint8)

        batch: dict[str, Any] = {
            self._obs_state_key: torch.as_tensor(state, dtype=torch.float32),
            "task": [str(obs.get("prompt") or self._default_prompt or "")],
        }
        if not self._visual_keys:
            batch["observation.image"] = _to_bchw_image(front)
        else:
            for key in self._visual_keys:
                src = wrist if "wrist" in key else front
                batch[key] = _to_bchw_image(src)

        proc = self._preprocessor(batch)
        pred = self._policy.select_action(proc)
        pred = self._postprocessor(pred)
        actions = np.asarray(pred.detach().cpu().numpy(), dtype=np.float32)
        return {"actions": actions}


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_base_model_ref(model_ref: Path | str) -> Path | str:
    text = str(model_ref).strip().replace("\\", "/")
    if not text:
        return text

    candidate = Path(text).expanduser()
    if candidate.exists():
        return candidate.resolve()

    local_base = (PROJECT_ROOT / "checkpoints" / "pi05_base").resolve()
    if local_base.exists() and text in {"lerobot/pi05_base", "pi05_base"}:
        return local_base

    if text.startswith(("gs://", "s3://", "http://", "https://")):
        return text

    if "/" in text:
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(repo_id=text, repo_type="model")).resolve()
        except Exception as err:
            raise RuntimeError(
                f"Failed to resolve base model repo id {text!r} to a local snapshot path."
            ) from err

    return text


def _is_remote_uri(path_text: str) -> bool:
    text = (path_text or "").strip().lower()
    return text.startswith("gs://") or text.startswith("s3://") or text.startswith("http://") or text.startswith("https://")


def _body_free_joint_qpos_adr(model: mujoco.MjModel, body_id: int) -> int | None:
    body_jnt_num = int(model.body_jntnum[body_id])
    body_jnt_adr = int(model.body_jntadr[body_id])
    for k in range(body_jnt_num):
        jnt_id = body_jnt_adr + k
        if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jnt_id])
    return None


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


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_to_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_err = _quat_mul(target, _quat_conj(current))
    if q_err[0] < 0:
        q_err = -q_err
    vec = q_err[1:]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(vec_norm, q_err[0])
    axis = vec / vec_norm
    return axis * angle


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_axisAngle2Quat(out, axis, angle)
    return out


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    qx = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), roll)
    qy = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=float), pitch)
    qz = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=float), yaw)
    q = _quat_mul(qz, _quat_mul(qy, qx))
    return q / max(1e-12, float(np.linalg.norm(q)))


def _ee_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), quat


def _solve_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    jnt_span: slice,
    dof_span: slice,
    act_span: slice,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> None:
    cur_pos, cur_quat = _ee_pose(model, data, site_id)
    pos_err = np.asarray(target_pos, dtype=float) - cur_pos
    rot_err = _quat_to_rotvec(np.asarray(target_quat, dtype=float), cur_quat)
    err = np.concatenate([pos_err, rot_err])

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp[:, dof_span], jacr[:, dof_span]])

    lhs = jac @ jac.T + 1e-4 * np.eye(6)
    dq = jac.T @ np.linalg.solve(lhs, 0.7 * err)
    data.ctrl[act_span] = data.qpos[jnt_span] + dq


def _apply_snapshot_assets(model: mujoco.MjModel, data: mujoco.MjData, assets: list[dict[str, Any]]) -> int:
    applied = 0
    for entry in assets:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue

        pos = np.asarray(entry.get("pos", [0.0, 0.0, 0.0]), dtype=float)
        quat = np.asarray(entry.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        if pos.shape != (3,) or quat.shape != (4,):
            continue

        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            quat = quat / quat_norm

        qpos_adr = _body_free_joint_qpos_adr(model, body_id)
        if qpos_adr is not None:
            data.qpos[qpos_adr : qpos_adr + 3] = pos
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
        else:
            model.body_pos[body_id] = pos
            model.body_quat[body_id] = quat
        applied += 1

    if applied > 0:
        mujoco.mj_forward(model, data)
    return applied


def _apply_robot_arm_state(model: mujoco.MjModel, data: mujoco.MjData, robot_arm: dict[str, Any] | None) -> bool:
    if robot_arm is None:
        return False

    joint_qpos = np.asarray(robot_arm.get("joint_qpos", []), dtype=float)
    if joint_qpos.shape != (6,):
        return False

    try:
        jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        act_id = model.actuator("/ur:shoulder_pan").id
        grip_qpos_adr = model.joint("/ur:2f85:right_driver_joint").qposadr.item()
        grip_act_id = model.actuator("/ur:2f85:fingers_actuator").id
    except Exception:
        return False

    gripper_qpos = robot_arm.get("gripper_qpos")
    if gripper_qpos is None:
        return False

    data.qpos[jnt_adr : jnt_adr + 6] = joint_qpos
    data.ctrl[act_id : act_id + 6] = joint_qpos
    data.qpos[grip_qpos_adr] = float(gripper_qpos)
    data.ctrl[grip_act_id] = float(np.clip(float(gripper_qpos) * 2550.0, 0.0, 255.0))
    mujoco.mj_forward(model, data)
    return True


def _snapshot_robot_arm_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any] | None:
    try:
        jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        grip_qpos_adr = model.joint("/ur:2f85:right_driver_joint").qposadr.item()
        site_id = model.site("/ur:2f85:pinch").id
    except Exception:
        return None

    ee_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(ee_quat, data.site_xmat[site_id])
    ee_pos = data.site_xpos[site_id].copy()
    return {
        "joint_qpos": [float(x) for x in data.qpos[jnt_adr : jnt_adr + 6].tolist()],
        "gripper_qpos": float(data.qpos[grip_qpos_adr]),
        "ee_pos": [float(x) for x in ee_pos.tolist()],
        "ee_quat": [float(x) for x in ee_quat.tolist()],
    }


def _collect_free_body_assets(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for body_id in range(1, int(model.nbody)):
        qpos_adr = _body_free_joint_qpos_adr(model, body_id)
        if qpos_adr is None:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name:
            continue
        pos = data.qpos[qpos_adr : qpos_adr + 3]
        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
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


def _save_scene_snapshot(scene_xml_path: Path, model: mujoco.MjModel, data: mujoco.MjData, out_path: Path) -> None:
    try:
        scene_xml_text = str(scene_xml_path.relative_to(PROJECT_ROOT))
    except Exception:
        scene_xml_text = str(scene_xml_path)

    payload = {
        "format": "evobody_scene_snapshot_v2",
        "scene_mode": "xml_override",
        "scene_xml": scene_xml_text,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "assets": _collect_free_body_assets(model, data),
        "robot_arm": _snapshot_robot_arm_state(model, data),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    if not values:
        return values
    out = []
    for i, v in enumerate(values):
        out.append(float(v) * float(scale[min(i, 2)]))
    return out


def _scale_attr(elem: ET.Element, attr: str, scale: np.ndarray) -> None:
    if attr not in elem.attrib:
        return
    vals = _parse_float_list(elem.get(attr), [])
    if not vals:
        return
    elem.set(attr, _format_float_list(_scale_numbers(vals, scale)))


def _scale_geom_size(elem: ET.Element, scale: np.ndarray) -> None:
    if "size" not in elem.attrib:
        return
    vals = _parse_float_list(elem.get("size"), [])
    if not vals:
        return
    gtype = (elem.get("type") or "").lower()
    if gtype in {"sphere"} and len(vals) == 1:
        uniform = float(np.mean(scale))
        elem.set("size", f"{vals[0] * uniform:.6f}")
        return
    if gtype in {"capsule", "cylinder"} and len(vals) == 2:
        radius_scale = float((scale[0] + scale[1]) / 2.0)
        elem.set("size", _format_float_list([vals[0] * radius_scale, vals[1] * scale[2]]))
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
            current = np.asarray(current[:3], dtype=float)
            elem.set("scale", _format_float_list(current * scale))


def _rewrite_asset_file_paths(root: ET.Element, source_xml: Path) -> None:
    source_dir = source_xml.parent
    assets_dir = PROJECT_ROOT / "assets"

    def _resolve_existing_path(file_attr: str) -> Path:
        raw = Path(file_attr).expanduser()
        if raw.is_absolute():
            return raw

        candidates = [
            (source_dir / raw),
            (assets_dir / raw),
            (PROJECT_ROOT / raw),
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        return candidates[0].resolve()

    for elem in root.iter():
        if elem.tag not in {"mesh", "texture", "hfield"}:
            continue
        file_attr = (elem.get("file") or "").strip()
        if not file_attr:
            continue
        file_path = _resolve_existing_path(file_attr)
        elem.set("file", file_path.as_posix())


def _write_scaled_asset_xml(source_xml: Path, out_xml: Path, scale: np.ndarray) -> Path:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    _transform_asset_tree_for_scale(root, scale)
    _rewrite_asset_file_paths(root, source_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    return out_xml


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


def _select_randomized_textures(randomize: bool) -> tuple[str, str, tuple[Path, Path] | None]:
    background_dir = PROJECT_ROOT / "assets" / "background_texture"
    background_images = []

    if background_dir.exists():
        background_images = list(background_dir.glob("*.png")) + \
                           list(background_dir.glob("*.jpg")) + \
                           list(background_dir.glob("*.jpeg"))

    if len(background_images) < 2:
        return (
            '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
            '    <texture type="2d" name="table_tex" builtin="checker" rgb1="0.6 0.7 0.8" rgb2="0.4 0.5 0.6" width="300" height="300"/>',
            None,
        )

    if randomize:
        random.seed(int(time.time()))
        selected_bg1, selected_bg2 = random.sample(background_images, 2)
    else:
        sorted_images = sorted(background_images)
        selected_bg1, selected_bg2 = sorted_images[0], sorted_images[1]

    bg_texture_line = f'    <texture name="background_tex" type="2d" file="{selected_bg1.name}" width="1024" height="1024"/>'
    table_texture_line = f'    <texture name="table_tex" type="2d" file="{selected_bg2.name}" width="1024" height="1024"/>'
    return bg_texture_line, table_texture_line, (selected_bg1, selected_bg2)


def _copy_textures_to_runtime(runtime_dir: Path, selected_pair: tuple[Path, Path] | None) -> None:
    if selected_pair is None:
        return
    bg1, bg2 = selected_pair
    if not bg1.exists() or not bg2.exists():
        return
    shutil.copy2(bg1, runtime_dir / bg1.name)
    shutil.copy2(bg2, runtime_dir / bg2.name)


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


def _discover_build_assets() -> dict[str, BuildAssetDef]:
    results: dict[str, BuildAssetDef] = {}
    candidates = []
    candidates.extend((MODEL_ROOT / "object").glob("*.xml"))
    candidates.extend((MODEL_ROOT / "instrument").glob("*.xml"))
    for path in sorted(candidates):
        try:
            root_body = _parse_root_body_name(path)
        except Exception:
            continue
        key = f"{path.parent.name}/{path.stem}"
        results[key] = BuildAssetDef(key=key, xml_path=path, root_body_name=root_body)
    return results


def _load_build_scene_v1(
    scene_json_path: Path,
    payload: dict[str, Any],
    randomize_texture: bool = True,
) -> tuple[LoadedScene, dict[str, Any]]:
    assets_map = _discover_build_assets()
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

    build_items: list[dict[str, Any]] = []
    payload_format = str(payload.get("format", "")).strip() or "evobody_build_scene_v1"

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

    runtime_dir = _create_timestamped_runtime_dir("evobody_eval_build_")
    runtime_suffix = runtime_dir.name[len("evobody_eval_build_"):]
    runtime_xml = runtime_dir / f"build_runtime_{runtime_suffix}.xml"

    base_pos_text = " ".join(f"{x:.6f}" for x in robot_base_pos)
    base_quat_text = " ".join(f"{x:.6f}" for x in robot_base_quat)
    cam_pos_text = " ".join(f"{x:.6f}" for x in camera_pos)
    cam_quat_text = " ".join(f"{x:.6f}" for x in camera_quat)

    bg_texture_line, table_texture_line, selected_texture_pair = _select_randomized_textures(randomize_texture)

    lines = [
        '<mujoco model="builder_eval">',
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
        model_file = asset_def.xml_path
        scale = item["scale"]
        if not np.allclose(scale, np.ones(3, dtype=float)):
            sx, sy, sz = [float(x) for x in scale.tolist()]
            scaled_name = f"{asset_def.xml_path.stem}__scaled__id{item['id']}__s_{sx:.4f}_{sy:.4f}_{sz:.4f}.xml"
            model_file = _write_scaled_asset_xml(asset_def.xml_path, scaled_asset_dir / scaled_name, scale)
        model_name = _sanitize_mj_name(f"{item['instance_name']}_model", fallback="asset_model")
        lines.append(f'    <model name="{model_name}" file="{model_file.as_posix()}" content_type="text/xml"/>')

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
                f'      <attach model="{model_name}" body="{asset_def.root_body_name}" prefix="{body_name}/"/>',
                '    </body>',
            ]
        )

    lines.extend(['  </worldbody>', '</mujoco>'])
    runtime_xml.write_text("\n".join(lines), encoding="utf-8")
    _copy_textures_to_runtime(runtime_dir, selected_texture_pair)

    model = mujoco.MjModel.from_xml_path(str(runtime_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    try:
        jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
        act_id = model.actuator("/ur:shoulder_pan").id
        grip_act_id = model.actuator("/ur:2f85:fingers_actuator").id
        data.qpos[jnt_adr : jnt_adr + 6] = robot_joint_targets
        data.ctrl[act_id : act_id + 6] = robot_joint_targets
        data.ctrl[grip_act_id] = float(np.clip(robot_gripper, 0.0, 255.0))
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    scene = LoadedScene(model=model, data=data, scene_json_path=scene_json_path, scene_xml_path=runtime_xml)
    meta = {
        "scene_json": str(scene_json_path),
        "scene_xml": str(runtime_xml),
        "payload_format": payload_format,
        "assets_declared": len(assets),
        "assets_loaded": len(build_items),
        "loaded_via": "native_build_scene_loader",
    }
    return scene, meta


def _load_scene_from_json(
    scene_json_path: Path,
    randomize_texture: bool = True,
) -> tuple[LoadedScene, dict[str, Any]]:
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene json does not exist: {scene_json_path}")

    payload = json.loads(scene_json_path.read_text(encoding="utf-8"))
    payload_format = str(payload.get("format", "")).strip()

    if payload_format in {"evobody_build_scene_v1", "evobody_build_scene_v2"}:
        return _load_build_scene_v1(scene_json_path, payload, randomize_texture=randomize_texture)

    if payload_format == "evobody_manual_scene_v1" or (
        not payload_format and isinstance(payload.get("assets"), list) and not payload.get("scene_xml")
    ):
        assets = payload.get("assets", []) if isinstance(payload.get("assets"), list) else []
        scene, base_meta = _load_build_scene_v1(
            scene_json_path,
            payload,
            randomize_texture=randomize_texture,
        )
        arm_applied = _apply_robot_arm_state(scene.model, scene.data, payload.get("robot_arm"))
        mujoco.mj_forward(scene.model, scene.data)
        meta = {
            "scene_json": str(scene_json_path),
            "scene_xml": str(scene.scene_xml_path),
            "payload_format": "evobody_manual_scene_v1" if payload_format == "evobody_manual_scene_v1" else "legacy_manual_scene",
            "assets_declared": len(assets),
            "assets_loaded": int(base_meta.get("assets_loaded", 0)),
            "arm_applied": bool(arm_applied),
            "loaded_via": "native_build_scene_loader",
        }
        return scene, meta

    scene_xml_text = str(payload.get("scene_xml", "")).strip()
    if not scene_xml_text:
        payload_format = str(payload.get("format", "")).strip() or "unknown"
        raise ValueError(
            "Scene json cannot be loaded. Unsupported format and missing 'scene_xml', "
            f"payload_format={payload_format!r}."
        )

    scene_xml = Path(scene_xml_text).expanduser()
    if not scene_xml.is_absolute():
        scene_xml = PROJECT_ROOT / scene_xml
    if not scene_xml.exists():
        raise FileNotFoundError(f"scene_xml does not exist: {scene_xml}")

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    applied_assets = 0
    if isinstance(payload.get("assets"), list):
        applied_assets = _apply_snapshot_assets(model, data, payload["assets"])

    arm_applied = _apply_robot_arm_state(model, data, payload.get("robot_arm"))
    mujoco.mj_forward(model, data)

    scene = LoadedScene(model=model, data=data, scene_json_path=scene_json_path, scene_xml_path=scene_xml)
    meta = {
        "scene_json": str(scene_json_path),
        "scene_xml": str(scene_xml),
        "assets_applied": applied_assets,
        "arm_applied": arm_applied,
        "loaded_via": "xml_snapshot",
    }
    return scene, meta


def _cleanup_generated_scene_cache(scene: LoadedScene, scene_meta: dict[str, Any] | None) -> None:
    if scene_meta is None:
        return

    loaded_via = str(scene_meta.get("loaded_via", "")).strip()
    if loaded_via != "native_build_scene_loader":
        return

    runtime_xml = scene.scene_xml_path
    runtime_dir = runtime_xml.parent
    try:
        runtime_dir_resolved = runtime_dir.resolve()
        log_root_resolved = LOG_ROOT.resolve()
    except Exception:
        runtime_dir_resolved = runtime_dir
        log_root_resolved = LOG_ROOT

    # Guard rails: only remove timestamped eval cache dirs under logs/.
    if runtime_dir_resolved.parent != log_root_resolved:
        return
    if not runtime_dir_resolved.name.startswith("evobody_eval_build_"):
        return
    if not runtime_xml.name.startswith("build_runtime_"):
        return

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _make_camera(model: mujoco.MjModel, camera_name: str) -> mujoco.MjvCamera | None:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = camera_id
    return cam


def _capture(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera | None):
    if camera is None:
        return None
    renderer.update_scene(data)
    mujoco.mjv_updateCamera(model, data, camera, renderer._scene)
    return renderer.render().astype(np.uint8)


def _create_renderer(model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer:
    try:
        return mujoco.Renderer(model, width, height)
    except Exception as err:
        backend = os.getenv("MUJOCO_GL", "")
        raise RuntimeError(
            "Failed to initialize MuJoCo renderer. "
            f"MUJOCO_GL={backend!r}. On headless servers, try `MUJOCO_GL=osmesa` or install EGL support and use `MUJOCO_GL=egl`."
        ) from err


def _apply_arm_perturbation(model: mujoco.MjModel, data: mujoco.MjData, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None

    lows = np.array([-0.1, 0.0, -0.2, -0.1, 0.0, -0.2], dtype=float)
    highs = np.array([0.1, 0.3, 0.2, 0.1, 0.3, 0.2], dtype=float)
    perturb = np.random.uniform(lows, highs)

    jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
    act_id = model.actuator("/ur:shoulder_pan").id
    data.qpos[jnt_adr : jnt_adr + 6] += perturb
    data.ctrl[act_id : act_id + 6] += perturb
    mujoco.mj_forward(model, data)
    return perturb


def _build_state_action_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
    act_id = model.actuator("/ur:shoulder_pan").id
    grip_qpos_adr = model.joint("/ur:2f85:right_driver_joint").qposadr.item()
    grip_act_id = model.actuator("/ur:2f85:fingers_actuator").id

    state_indices = np.asarray(list(range(jnt_adr, jnt_adr + 6)) + [grip_qpos_adr], dtype=int)
    action_indices = np.asarray(list(range(act_id, act_id + 6)) + [grip_act_id], dtype=int)
    return state_indices, action_indices


def _slug(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-zA-Z0-9._-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "task"


def _extract_task_description_from_code(code: str | None) -> str | None:
    if not code:
        return None
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        line = line.strip("[](){} ").strip()
        return line or None
    return None


def _run_rollout(
    scene: LoadedScene,
    policy: Any,
    task_prompt: str,
    time_limit_s: float,
    control_repeat: int,
    perturb_arm: bool,
    output_root: Path,
) -> dict[str, Any]:
    model = scene.model
    data = scene.data
    perturb = _apply_arm_perturbation(model, data, perturb_arm)

    state_indices, action_indices = _build_state_action_indices(model)
    site_id = model.site("/ur:2f85:pinch").id
    jnt_adr = model.joint("/ur:shoulder_pan").qposadr.item()
    dof_adr = model.joint("/ur:shoulder_pan").dofadr.item()
    act_id = model.actuator("/ur:shoulder_pan").id
    grip_act_id = model.actuator("/ur:2f85:fingers_actuator").id
    jnt_span = slice(jnt_adr, jnt_adr + 6)
    dof_span = slice(dof_adr, dof_adr + 6)
    act_span = slice(act_id, act_id + 6)
    

    # Include microseconds to avoid run-dir collisions across fast repeated evaluations.
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{_slug(task_prompt)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = run_dir / "rollout.mp4"
    current_image_path = run_dir / EVAL_CURRENT_IMAGE_NAME
    finished_image_path = run_dir / EVAL_FINISHED_IMAGE_NAME
    finished_wrist_image_path = run_dir / EVAL_FINISHED_WRIST_IMAGE_NAME
    pre_scene_path = run_dir / EVAL_PRE_SCENE_NAME
    finished_scene_path = run_dir / EVAL_FINISHED_SCENE_NAME

    renderer = _create_renderer(model, 256, 256)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    renderer._scene_option.sitegroup[:] = False

    cam_front = _make_camera(model, "table_cam_front")
    cam_wrist = _make_camera(model, "/ur:wrist_cam")
    if cam_front is None:
        raise RuntimeError("Camera 'table_cam_front' not found in scene")

    pre = _capture(renderer, model, data, cam_front)
    if pre is not None:
        imageio.imwrite(current_image_path, pre)
    _save_scene_snapshot(scene.scene_xml_path, model, data, pre_scene_path)

    frame_count = 0
    adapter_hist: list[int] = []
    sim_start_time = float(data.time)
    sim_dt = float(model.opt.timestep)
    video_fps = max(1.0, 1.0 / max(sim_dt, 1e-9))

    try:
        writer = imageio.get_writer(video_path, fps=video_fps)
    except Exception:
        # Some environments miss mp4 backends (ffmpeg/pyav). Fallback to GIF to keep rollout running.
        video_path = run_dir / "rollout.gif"
        writer = imageio.get_writer(video_path, fps=max(1, int(round(video_fps))))
    try:
        while (float(data.time) - sim_start_time) < float(time_limit_s):
            front_img = _capture(renderer, model, data, cam_front)
            wrist_img = _capture(renderer, model, data, cam_wrist)
            wrist_img_2 = np.zeros_like(front_img) if front_img is not None else None

            obs = {
                "observation/state": np.asarray(data.qpos[state_indices], dtype=np.float32),
                "observation/image": front_img,
                "observation/wrist_image": wrist_img if wrist_img is not None else front_img,
                "observation/wrist_image_2": wrist_img_2,
                "prompt": task_prompt,
            }
            out = policy.infer(obs)
            actions = np.asarray(out["actions"], dtype=float)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2:
                raise RuntimeError(f"Policy returned invalid action shape: {actions.shape}")

            adapter_id = out.get("adapter_id")
            if adapter_id is not None and int(adapter_id) >= 0:
                adapter_hist.append(int(adapter_id))

            if actions.shape[1] < 7:
                raise RuntimeError(f"Action dim mismatch: got={actions.shape[1]}, required>=7 (dx,dy,dz,droll,dpitch,dyaw,gripper)")
            if actions.shape[1] > 7:
                actions = actions[:, :7]

            for action in actions:
                if (float(data.time) - sim_start_time) >= float(time_limit_s):
                    break
                dpos = np.asarray(action[:3], dtype=float)
                droll, dpitch, dyaw = [float(v) for v in action[3:6]]
                grip = float(action[6])
                cur_pos, cur_quat = _ee_pose(model, data, site_id)
                target_pos = cur_pos + dpos
                dquat = _quat_from_euler(droll, dpitch, dyaw)
                target_quat = _quat_mul(dquat, cur_quat)
                target_quat /= max(1e-12, float(np.linalg.norm(target_quat)))
                for _ in range(max(1, int(control_repeat))):
                    if (float(data.time) - sim_start_time) >= float(time_limit_s):
                        break
                    _solve_ik_step(
                        model,
                        data,
                        site_id=site_id,
                        jnt_span=jnt_span,
                        dof_span=dof_span,
                        act_span=act_span,
                        target_pos=target_pos,
                        target_quat=target_quat,
                    )
                    data.ctrl[grip_act_id] = float(np.clip(grip, 0.0, 255.0))
                    mujoco.mj_step(model, data)
                    frame = _capture(renderer, model, data, cam_front)
                    if frame is None:
                        raise RuntimeError("Failed to capture frame")
                    writer.append_data(frame)
                    frame_count += 1

        post = _capture(renderer, model, data, cam_front)
        post_wrist = _capture(renderer, model, data, cam_wrist)
        if post is not None:
            imageio.imwrite(finished_image_path, post)
        if post_wrist is not None:
            imageio.imwrite(finished_wrist_image_path, post_wrist)
        _save_scene_snapshot(scene.scene_xml_path, model, data, finished_scene_path)
    finally:
        writer.close()
        renderer.close()

    return {
        "run_dir": run_dir,
        "video_path": video_path,
        "current_image": current_image_path,
        "finished_image": finished_image_path,
        "finished_wrist_image": finished_wrist_image_path,
        "pre_scene_json": pre_scene_path,
        "finished_scene_json": finished_scene_path,
        "frames": frame_count,
        "adapter_history": adapter_hist,
        "arm_perturbation": None if perturb is None else [float(x) for x in perturb.tolist()],
        "video_fps": float(video_fps),
        "sim_time_end": float(data.time),
    }


def _run_judge(
    task_prompt: str,
    finished_image: Path,
    finished_wrist_image: Path | None = None,
) -> tuple[bool, str]:
    from evoma import TaskSuccessJudgeAgent

    api_base_url = os.getenv("API_BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    if not api_base_url or not api_key or not model_name:
        raise RuntimeError("API_BASE_URL, API_KEY, MODEL_NAME are required in environment for judge")

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    judge = TaskSuccessJudgeAgent(client=client, model=model_name)
    wrist = str(finished_wrist_image) if finished_wrist_image is not None and finished_wrist_image.is_file() else None
    return judge.judge(task_prompt=task_prompt, image_path=str(finished_image), wrist_image_path=wrist)


def _serialize_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(rollout["run_dir"]),
        "video_path": str(rollout["video_path"]),
        "current_image": str(rollout["current_image"]),
        "finished_image": str(rollout["finished_image"]),
        "finished_wrist_image": str(rollout.get("finished_wrist_image", "")),
        "pre_scene_json": str(rollout["pre_scene_json"]),
        "finished_scene_json": str(rollout["finished_scene_json"]),
        "frames": int(rollout["frames"]),
        "video_fps": float(rollout["video_fps"]),
        "adapter_history": [int(x) for x in rollout["adapter_history"]],
        "arm_perturbation": rollout["arm_perturbation"],
        "sim_time_end": float(rollout["sim_time_end"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI0.5 policy on one scene using LeRobot")
    parser.add_argument("--scene-json", required=True, help="Scene snapshot json path")
    parser.add_argument("--task", required=True, help="Task description")

    parser.add_argument("--config-name", default="new_task_pi05", help="Reserved for compatibility")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH, help="LeRobot pretrained policy path or repo id")
    parser.add_argument("--bank-dir", default=str(DEFAULT_BANK_DIR), help="Reserved for compatibility")
    parser.add_argument("--checkpoint-assets-dir", default="third_party/lerobot/assets", help="Reserved for compatibility")
    parser.add_argument(
        "--use-norm-stats",
        "--eval-use-norm-stats",
        dest="use_norm_stats",
        action="store_true",
        default=True,
        help="Use norm_stats normalize/unnormalize transforms during inference (default: true)",
    )
    parser.add_argument(
        "--no-use-norm-stats",
        "--no-eval-use-norm-stats",
        dest="use_norm_stats",
        action="store_false",
        help="Disable norm_stats transforms during inference",
    )

    parser.add_argument("--num-steps", type=int, default=10, help="Action horizon sampled per policy call")
    parser.add_argument("--time-limit-s", type=float, default=1.0, help="Simulation time limit in seconds")
    parser.add_argument("--control-repeat", type=int, default=10, help="Mujoco steps per control action")
    parser.add_argument("--eval-loops", type=int, default=5, help="Number of full evaluate runs")
    parser.add_argument("--perturb-arm", action="store_true", help="Apply random UR5 perturbation before rollout")
    parser.add_argument(
        "--randomize-scene",
        "--eval-randomize-scene",
        dest="randomize_scene",
        action="store_true",
        default=True,
        help="Randomize background/table textures when loading build-scene JSON (default: true)",
    )
    parser.add_argument(
        "--no-randomize-scene",
        "--no-eval-randomize-scene",
        dest="randomize_scene",
        action="store_false",
        help="Disable scene texture randomization and use deterministic texture pair",
    )

    parser.add_argument("--default-prompt", default="", help="Default prompt injected by transforms if missing")
    parser.add_argument(
        "--inference-backend",
        choices=["pytorch"],
        default="pytorch",
        help="Policy inference backend",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_EVAL_ROOT), help="Output directory for evaluation artifacts")
    parser.add_argument("--result-json-out", default="", help="Optional path to also write result.json")
    parser.add_argument(
        "--judge",
        dest="enable_judge",
        action="store_true",
        default=True,
        help="Enable VLM judge for success evaluation (default: enabled)",
    )
    parser.add_argument(
        "--no-judge",
        dest="enable_judge",
        action="store_false",
        help="Disable VLM judge and treat rollout as success",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_json_path = _resolve_path(args.scene_json)
    base_model_path: Path | str
    raw_model_ref = str(args.base_model_path or "").strip()
    if _is_remote_uri(raw_model_ref):
        base_model_path = raw_model_ref
    else:
        resolved_model_path = _resolve_path(raw_model_ref)
        if resolved_model_path.exists():
            base_model_path = resolved_model_path
        else:
            # Keep non-existing refs as repo ids (e.g. "lerobot/pi05").
            base_model_path = raw_model_ref
    bank_dir = _resolve_path(args.bank_dir)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_assets_dir and args.checkpoint_assets_dir.strip():
        checkpoint_assets_dir = _resolve_path(args.checkpoint_assets_dir)
    else:
        if isinstance(base_model_path, Path):
            checkpoint_assets_dir = base_model_path.parent / "assets"
        else:
            checkpoint_assets_dir = str(base_model_path).rstrip("/").replace("/params", "/assets")

    _log(
        "Validation bootstrap load: "
        f"base={base_model_path}, "
        f"backend={args.inference_backend}, "
        f"judge_enabled={bool(args.enable_judge)}, "
        f"compat_bank_dir_ignored={bool(args.bank_dir.strip())}, "
        f"compat_checkpoint_assets_ignored={bool(args.checkpoint_assets_dir.strip())}"
    )
    _log("Loading scene json")
    _log(f"Loading continual policy (backend={args.inference_backend})")
    policy_cls = ContinualPolicyLeRobot
    policy = policy_cls(
        config_name=args.config_name,
        base_model_path=base_model_path,
        bank_dir=bank_dir,
        checkpoint_assets_dir=checkpoint_assets_dir,
        use_norm_stats=bool(args.use_norm_stats),
        num_steps=int(args.num_steps),
        default_prompt=args.default_prompt.strip() if args.default_prompt and args.default_prompt.strip() else None,
    )

    eval_loops = max(1, int(args.eval_loops))
    success_count = 0
    loop_results: list[dict[str, Any]] = []
    final_scene_meta: dict[str, Any] | None = None
    final_rollout: dict[str, Any] | None = None

    for eval_idx in range(1, eval_loops + 1):
        scene: LoadedScene | None = None
        scene_meta: dict[str, Any] | None = None
        try:
            _log(f"[{eval_idx}/{eval_loops}] Loading scene json")
            scene, scene_meta = _load_scene_from_json(
                scene_json_path,
                randomize_texture=bool(args.randomize_scene),
            )
            _log(f"[{eval_idx}/{eval_loops}] Scene loaded: {scene_meta}")

            _log(f"[{eval_idx}/{eval_loops}] Running rollout")
            rollout = _run_rollout(
                scene=scene,
                policy=policy,
                task_prompt=args.task.strip(),
                time_limit_s=float(args.time_limit_s),
                control_repeat=int(args.control_repeat),
                perturb_arm=bool(args.perturb_arm),
                output_root=output_root,
            )

            if bool(args.enable_judge):
                _log(f"[{eval_idx}/{eval_loops}] Running judge")
                one_success, judge_text = _run_judge(
                    task_prompt=args.task.strip(),
                    finished_image=rollout["finished_image"],
                    finished_wrist_image=rollout.get("finished_wrist_image"),
                )
            else:
                one_success, judge_text = True, "SKIPPED: judge disabled by --no-judge"
            if one_success:
                success_count += 1
            else:
                vp = rollout.get("video_path")
                src_mp4 = Path(vp) if vp is not None else Path()
                if src_mp4.is_file():
                    LOG_ROOT.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_mp4, LOG_ROOT / "fail.mp4")

            loop_result = {
                "index": eval_idx,
                "status": "SUCCESS" if one_success else "FAIL",
                "success": bool(one_success),
                "judge": judge_text,
                "scene": scene_meta,
                "rollout": _serialize_rollout(rollout),
            }
            loop_results.append(loop_result)
            final_scene_meta = scene_meta
            final_rollout = rollout
        finally:
            if scene is not None:
                _cleanup_generated_scene_cache(scene, scene_meta)

    success_rate = float(success_count) / float(eval_loops)
    success = success_rate > 0.5
    status = "SUCCESS" if success else "FAIL"

    if final_scene_meta is None or final_rollout is None:
        raise RuntimeError("No evaluation loop result was produced.")

    last_successful_finished_scene_json = ""
    for lr in reversed(loop_results):
        if not lr.get("success"):
            continue
        ro = lr.get("rollout")
        if isinstance(ro, dict):
            fs = str(ro.get("finished_scene_json", "")).strip()
            if fs:
                last_successful_finished_scene_json = fs
                break

    result = {
        "status": status,
        "success": bool(success),
        "success_count": int(success_count),
        "eval_loops": int(eval_loops),
        "success_rate": float(success_rate),
        "last_successful_finished_scene_json": last_successful_finished_scene_json,
        "judge": loop_results[-1]["judge"],
        "task": args.task.strip(),
        "scene": final_scene_meta,
        "config_name": args.config_name,
        "base_model_path": str(base_model_path),
        "bank_dir": str(bank_dir),
        "checkpoint_assets_dir": str(checkpoint_assets_dir),
        "use_norm_stats": bool(args.use_norm_stats),
        "num_steps": int(args.num_steps),
        "time_limit_s": float(args.time_limit_s),
        "control_repeat": int(args.control_repeat),
        "inference_backend": args.inference_backend,
        # Keep the top-level single rollout field for backward compatibility.
        "rollout": _serialize_rollout(final_rollout),
        "loop_results": loop_results,
        "loop_runs": loop_results,
        "enable_judge": bool(args.enable_judge),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    result_path = Path(rollout["run_dir"]) / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.result_json_out and args.result_json_out.strip():
        ext_result_path = _resolve_path(args.result_json_out)
        ext_result_path.parent.mkdir(parents=True, exist_ok=True)
        ext_result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        "Evaluation finished: "
        f"{status} (success_rate={success_rate:.3f}, success_count={success_count}/{eval_loops})"
    )
    _log(f"Video: {final_rollout['video_path']}")
    _log(f"Finished image: {final_rollout['finished_image']}")
    _log(f"Result json: {result_path}")


if __name__ == "__main__":
    main()
