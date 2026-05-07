from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ur7e_controller import ROBOT_IP, UR7eVectorController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_PATH = "checkpoints/pi05_base"
DEFAULT_BANK_DIR = PROJECT_ROOT / "checkpoints" / "pi05_adapter_bank_pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "vla_real_eval"
PALIGEMMA_REMOTE_ID = "google/paligemma-3b-pt-224"


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [evaluate_vla_real] {msg}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(str(path_text).strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _is_remote_uri(path_text: str) -> bool:
    text = (path_text or "").strip().lower()
    return text.startswith(("gs://", "s3://", "http://", "https://"))


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


def _resolve_model_arg(raw_model_ref: str) -> Path | str:
    if _is_remote_uri(raw_model_ref):
        return raw_model_ref
    candidate = _resolve_path(raw_model_ref)
    return candidate if candidate.exists() else raw_model_ref


def _resolve_local_paligemma_dir(checkpoint_root: Path) -> Path | None:
    for candidate in (
        checkpoint_root / "paligemma-3b-pt-224",
        checkpoint_root / "google" / "paligemma-3b-pt-224",
    ):
        if (candidate / "config.json").exists():
            return candidate.resolve()
    return None


def _rewrite_preprocessor_tokenizer_to_local(model_dir: Path, local_paligemma_dir: Path) -> bool:
    preprocessor_path = model_dir / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        return False
    raw = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    steps = raw.get("steps") if isinstance(raw, dict) else None
    if not isinstance(steps, list):
        return False

    changed = False
    local_name = local_paligemma_dir.as_posix()
    for step in steps:
        if not isinstance(step, dict) or step.get("registry_name") != "tokenizer_processor":
            continue
        cfg = step.get("config")
        if not isinstance(cfg, dict):
            continue
        if str(cfg.get("tokenizer_name", "")).strip() == PALIGEMMA_REMOTE_ID:
            cfg["tokenizer_name"] = local_name
            changed = True

    if changed:
        preprocessor_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


class RealSenseRGBPair:
    def __init__(
        self,
        serials: Sequence[str],
        width: int,
        height: int,
        fps: int,
        warmup_frames: int,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'pyrealsense2'. Install Intel RealSense SDK Python bindings first."
            ) from exc

        self._rs = rs
        self._serials = [s.strip() for s in serials if s.strip()]
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._warmup_frames = max(0, int(warmup_frames))
        self._pipelines: list[Any] = []

    def __enter__(self) -> "RealSenseRGBPair":
        serials = self._serials or self._discover_serials()
        if not serials:
            raise RuntimeError("No Intel RealSense devices were found.")
        if len(serials) == 1:
            _log("Only one RealSense RGB stream found; front image will also be used as wrist image.")

        for serial in serials[:2]:
            pipeline = self._rs.pipeline()
            cfg = self._rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                self._rs.stream.color,
                self._width,
                self._height,
                self._rs.format.rgb8,
                self._fps,
            )
            pipeline.start(cfg)
            self._pipelines.append(pipeline)

        for _ in range(self._warmup_frames):
            self.capture()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for pipeline in self._pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        self._pipelines.clear()

    def _discover_serials(self) -> list[str]:
        ctx = self._rs.context()
        out: list[str] = []
        for dev in ctx.query_devices():
            try:
                out.append(dev.get_info(self._rs.camera_info.serial_number))
            except Exception:
                continue
        return out

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._pipelines:
            raise RuntimeError("RealSense pipelines are not started.")

        images: list[np.ndarray] = []
        for pipeline in self._pipelines:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("Failed to read a RealSense RGB frame.")
            images.append(np.asarray(color.get_data(), dtype=np.uint8).copy())
        if len(images) == 1:
            images.append(images[0].copy())
        return images[0], images[1]


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
    ) -> None:
        del config_name, bank_dir, checkpoint_assets_dir, use_norm_stats
        self._num_steps = max(1, int(num_steps))
        self._default_prompt = default_prompt or ""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lerobot_src = PROJECT_ROOT / "third_party" / "lerobot" / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))

        try:
            from lerobot.configs.policies import PreTrainedConfig
        except ImportError:
            try:
                from lerobot.configs import PreTrainedConfig
            except ImportError:
                from lerobot.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.constants import OBS_STATE

        resolved_base_model_path = _resolve_base_model_ref(base_model_path)
        model_ref = str(resolved_base_model_path)
        cfg = PreTrainedConfig.from_pretrained(model_ref)
        cfg.device = str(self._device)
        cfg.n_action_steps = self._num_steps
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
        self._state_dim = self._infer_state_dim(cfg)

    def _infer_state_dim(self, cfg: Any) -> int:
        feature = getattr(cfg, "input_features", {}).get(self._obs_state_key)
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape:
            return int(shape[-1])
        return 7

    def _to_bchw_image(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={img.shape}")
        chw = np.transpose(img, (2, 0, 1))
        if self._policy_type == "smolvla":
            tensor = torch.as_tensor(chw[None, ...], dtype=torch.float32)
            if float(tensor.max().item()) > 1.0:
                tensor = tensor / 255.0
            return tensor
        return torch.as_tensor(chw[None, ...], dtype=torch.uint8)

    def _prepare_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        front = np.asarray(obs.get("observation/image"), dtype=np.uint8)
        wrist = np.asarray(obs.get("observation/wrist_image", front), dtype=np.uint8)
        state = np.asarray(obs.get("observation/state"), dtype=np.float32).reshape(-1)
        if state.size < self._state_dim:
            state = np.pad(state, (0, self._state_dim - state.size))
        elif state.size > self._state_dim:
            state = state[: self._state_dim]

        batch: dict[str, Any] = {
            self._obs_state_key: torch.as_tensor(state[None, :], dtype=torch.float32),
            "task": [str(obs.get("prompt") or self._default_prompt or "")],
        }
        if not self._visual_keys:
            batch["observation.image"] = self._to_bchw_image(front)
        else:
            for key in self._visual_keys:
                src = wrist if "wrist" in key or "camera2" in key or "left" in key else front
                batch[key] = self._to_bchw_image(src)
        return self._preprocessor(batch)

    def infer_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        proc = self._prepare_batch(obs)
        with torch.inference_mode():
            if hasattr(self._policy, "predict_action_chunk"):
                pred = self._policy.predict_action_chunk(proc)[:, : self._num_steps]
                try:
                    pred = self._postprocessor(pred)
                except Exception:
                    flat = pred.reshape(-1, pred.shape[-1])
                    flat = self._postprocessor(flat)
                    pred = flat.reshape(1, -1, flat.shape[-1])
            else:
                actions = [self._postprocessor(self._policy.select_action(proc)) for _ in range(self._num_steps)]
                pred = torch.stack(actions, dim=1)
        actions = np.asarray(pred.detach().cpu().numpy(), dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[1] < 7:
            raise RuntimeError(
                f"Action dim mismatch: got={actions.shape[1]}, required>=7 "
                "(dx,dy,dz,droll,dpitch,dyaw,gripper)"
            )
        return actions[:, :7]


@dataclass
class SafetyLimits:
    max_pos_delta_m: float
    max_rot_delta_rad: float
    gripper_min: float
    gripper_max: float


def _clip_action(action: Iterable[float], limits: SafetyLimits) -> np.ndarray:
    arr = np.asarray(list(action), dtype=np.float32).reshape(-1)
    if arr.size != 7:
        raise ValueError(f"Expected 7D action, got shape={arr.shape}")
    arr[:3] = np.clip(arr[:3], -limits.max_pos_delta_m, limits.max_pos_delta_m)
    arr[3:6] = np.clip(arr[3:6], -limits.max_rot_delta_rad, limits.max_rot_delta_rad)
    arr[6] = np.clip(arr[6], limits.gripper_min, limits.gripper_max)
    return arr


def _state_from_controller(controller: UR7eVectorController | None) -> np.ndarray:
    if controller is None:
        return np.zeros(7, dtype=np.float32)
    joints = controller.get_current_joints()
    g = controller.get_gripper_open_ratio()
    return np.asarray([*joints, g], dtype=np.float32)


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VLA checkpoint on a real UR7e + RealSense setup.")
    parser.add_argument("--task", required=True, help="Task description text for the VLA policy")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH, help="LeRobot checkpoint/pretrained_model path or repo id")
    parser.add_argument("--checkpoint-root", default="checkpoints", help="Checkpoint root used to resolve local Paligemma files")
    parser.add_argument("--config-name", default="new_task_pi05", help="Reserved for evaluate_cli compatibility")
    parser.add_argument("--bank-dir", default=str(DEFAULT_BANK_DIR), help="Reserved for evaluate_cli compatibility")
    parser.add_argument("--checkpoint-assets-dir", default="third_party/lerobot/assets", help="Reserved for evaluate_cli compatibility")
    parser.add_argument("--default-prompt", default="", help="Default prompt injected by transforms if missing")
    parser.add_argument("--num-steps", type=int, default=10, help="Action chunk horizon to execute per policy call")
    parser.add_argument("--eval-loops", type=int, default=1, help="Number of policy chunks to execute")

    parser.add_argument("--robot-ip", default=ROBOT_IP, help="UR7e controller IP")
    parser.add_argument("--robotiq-urscript-defs-path", default="", help="Optional Robotiq URScript definitions file for fallback gripper control")
    parser.add_argument("--strict-gripper-connection", action="store_true", help="Fail if Robotiq gripper connection is unavailable")
    parser.add_argument("--arm-acceleration", type=float, default=0.4, help="UR movel acceleration")
    parser.add_argument("--arm-velocity", type=float, default=0.10, help="UR movel velocity")
    parser.add_argument("--step-delay-s", type=float, default=0.15, help="Delay after each action")

    parser.add_argument("--camera-serials", default="", help="Comma-separated RealSense serials: front[,wrist]. Auto-discovers if omitted")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)

    parser.add_argument("--max-pos-delta-m", type=float, default=0.03, help="Per-step position delta safety clamp")
    parser.add_argument("--max-rot-delta-rad", type=float, default=0.20, help="Per-step orientation delta safety clamp")
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Run cameras and model, log actions, but do not connect to the robot")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for real-rollout logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_root = _resolve_path(args.checkpoint_root)
    base_model_path = _resolve_model_arg(str(args.base_model_path or "").strip())
    if isinstance(base_model_path, Path):
        local_paligemma = _resolve_local_paligemma_dir(checkpoint_root)
        if local_paligemma is not None:
            _rewrite_preprocessor_tokenizer_to_local(base_model_path, local_paligemma)

    output_root = _resolve_path(args.output_root)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / "rollout.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    serials = [s.strip() for s in str(args.camera_serials).split(",") if s.strip()]
    limits = SafetyLimits(
        max_pos_delta_m=float(args.max_pos_delta_m),
        max_rot_delta_rad=float(args.max_rot_delta_rad),
        gripper_min=float(args.gripper_min),
        gripper_max=float(args.gripper_max),
    )

    _log(f"Loading policy checkpoint: {base_model_path}")
    policy = ContinualPolicyLeRobot(
        config_name=args.config_name,
        base_model_path=base_model_path,
        bank_dir=_resolve_path(args.bank_dir),
        checkpoint_assets_dir=_resolve_path(args.checkpoint_assets_dir),
        use_norm_stats=True,
        num_steps=int(args.num_steps),
        default_prompt=args.default_prompt.strip() or args.task.strip(),
    )

    controller: UR7eVectorController | None = None
    if not args.dry_run:
        controller = UR7eVectorController(
            robot_ip=str(args.robot_ip),
            robotiq_urscript_defs_path=str(args.robotiq_urscript_defs_path).strip() or None,
            strict_gripper_connection=bool(args.strict_gripper_connection),
        )

    _log("Starting RealSense RGB capture")
    with RealSenseRGBPair(
        serials=serials,
        width=int(args.camera_width),
        height=int(args.camera_height),
        fps=int(args.camera_fps),
        warmup_frames=int(args.camera_warmup_frames),
    ) as cameras:
        try:
            if controller is not None:
                _log(f"Connecting UR7e controller at {args.robot_ip}")
                controller.connect()
                if controller.is_gripper_available():
                    _log(f"Gripper backend: {controller.get_gripper_backend()}")

            for loop_idx in range(max(1, int(args.eval_loops))):
                front_img, wrist_img = cameras.capture()
                state = _state_from_controller(controller)
                obs = {
                    "observation/state": state,
                    "observation/image": front_img,
                    "observation/wrist_image": wrist_img,
                    "prompt": args.task.strip(),
                }
                action_chunk = policy.infer_action_chunk(obs)
                _log(f"[{loop_idx + 1}/{args.eval_loops}] action_chunk shape={action_chunk.shape}")

                for step_idx, action in enumerate(action_chunk):
                    clipped = _clip_action(action, limits)
                    event = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "loop_idx": loop_idx,
                        "step_idx": step_idx,
                        "task": args.task.strip(),
                        "state": [float(x) for x in state.tolist()],
                        "raw_action": [float(x) for x in np.asarray(action).tolist()],
                        "clipped_action": [float(x) for x in clipped.tolist()],
                        "dry_run": bool(args.dry_run),
                    }
                    if controller is None:
                        _log(f"dry-run action[{loop_idx}:{step_idx}]={event['clipped_action']}")
                    else:
                        current_ee, target_ee = controller.send_ee_delta_vector(
                            clipped,
                            acceleration=float(args.arm_acceleration),
                            velocity=float(args.arm_velocity),
                            wait_after_arm_s=float(args.step_delay_s),
                        )
                        event["current_ee"] = [float(x) for x in current_ee]
                        event["target_ee"] = [float(x) for x in target_ee]
                    _write_jsonl(log_path, event)

        finally:
            if controller is not None:
                controller.close()

    _log(f"Done. Rollout log: {log_path}")


if __name__ == "__main__":
    main()
