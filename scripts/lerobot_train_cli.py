import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PALIGEMMA_REMOTE_ID = "google/paligemma-3b-pt-224"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _resolve_dataset_dir(dataset_root: Path, dataset_repo_id: str) -> Path:
    repo_id_norm = _normalize_model_ref(dataset_repo_id)
    repo_rel = Path(*[seg for seg in repo_id_norm.split("/") if seg.strip()])
    candidate_repo = dataset_root / repo_rel
    if (candidate_repo / "meta" / "info.json").exists():
        return candidate_repo
    if (dataset_root / "meta" / "info.json").exists():
        return dataset_root
    return candidate_repo


def _auto_rename_map_for_dataset(dataset_dir: Path, policy_type: str) -> dict[str, str]:
    policy_type_norm = str(policy_type or "").strip().lower()
    smolvla_expected = {
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    }
    generated_keys = {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }
    generated_to_smolvla = {
        "observation.images.base_0_rgb": "observation.images.camera1",
        "observation.images.left_wrist_0_rgb": "observation.images.camera2",
        "observation.images.right_wrist_0_rgb": "observation.images.camera3",
    }
    generated_to_pi05 = {
        "observation.images.base_0_rgb": "image",
        "observation.images.left_wrist_0_rgb": "wrist_image",
        "observation.images.right_wrist_0_rgb": "wrist_image",
    }

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        # Conservative fallback for legacy dataset fields.
        return generated_to_smolvla if policy_type_norm == "smolvla" else generated_to_pi05

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        feature_keys = set((info.get("features") or {}).keys())
    except Exception:
        feature_keys = set()

    if policy_type_norm == "smolvla":
        # SmolVLA configs in this project expect camera1/camera2/camera3 names.
        if smolvla_expected.issubset(feature_keys):
            return {}
        if generated_keys & feature_keys:
            return generated_to_smolvla
        return {}

    has_new_image_keys = any(k.startswith("observation.images.") for k in feature_keys)
    has_legacy_image_keys = any(k in feature_keys for k in {"image", "wrist_image"})

    if has_new_image_keys:
        # Dataset already matches PI0.5 expected names; do not rename.
        return {}
    if has_legacy_image_keys:
        # Legacy generated datasets still use flat image keys.
        return {
            "observation.images.base_0_rgb": "image",
            "observation.images.left_wrist_0_rgb": "wrist_image",
            "observation.images.right_wrist_0_rgb": "wrist_image",
        }
    return {}


def _resolve_rename_map(text: str, dataset_dir: Path, policy_type: str) -> dict[str, str]:
    raw = str(text or "").strip()
    if not raw or raw.lower() == "auto":
        return _auto_rename_map_for_dataset(dataset_dir, policy_type=policy_type)

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--rename-map-json must be a JSON object or 'auto'")
    return {str(k): str(v) for k, v in parsed.items()}


def _normalize_model_ref(text: str) -> str:
    """Normalize slashes for Hub repo ids and for CLI args on Windows.

    HuggingFace ``repo_id`` must use ``/`` (e.g. ``lerobot/pi05_base``). Raw
    Windows paths still work when passed to ``Path`` after this (``C:/a/b``).
    """
    return str(text).strip().replace("\\", "/")


def _resolve_base_model_ref(text: str) -> str:
    ref = _normalize_model_ref(text)
    if not ref:
        return ref

    candidate = Path(ref).expanduser()
    if candidate.exists():
        return candidate.resolve().as_posix()

    local_base = (PROJECT_ROOT / "checkpoints" / "pi05_base").resolve()
    if local_base.exists() and ref in {"lerobot/pi05_base", "pi05_base"}:
        return local_base.as_posix()

    if ref.startswith(("gs://", "s3://", "http://", "https://")):
        return ref

    if "/" in ref:
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(repo_id=ref, repo_type="model")).resolve().as_posix()
        except Exception as err:
            raise RuntimeError(
                f"Failed to resolve base model repo id {ref!r} to a local snapshot path."
            ) from err

    return ref


def _resolve_local_paligemma_dir(checkpoint_root: Path) -> Path:
    candidates = [
        checkpoint_root / "paligemma-3b-pt-224",
        checkpoint_root / "google" / "paligemma-3b-pt-224",
    ]
    for p in candidates:
        if (p / "config.json").exists():
            return p.resolve()
    raise FileNotFoundError(
        "Local Paligemma checkpoint not found. Expected one of: "
        f"{(checkpoint_root / 'paligemma-3b-pt-224')}, "
        f"{(checkpoint_root / 'google' / 'paligemma-3b-pt-224')}"
    )


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
        if not isinstance(step, dict):
            continue
        if step.get("registry_name") != "tokenizer_processor":
            continue
        cfg = step.get("config")
        if not isinstance(cfg, dict):
            continue
        tok_name = str(cfg.get("tokenizer_name", "")).strip()
        if tok_name == PALIGEMMA_REMOTE_ID:
            cfg["tokenizer_name"] = local_name
            changed = True

    if changed:
        preprocessor_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def _load_router_registry(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            out[name] = int(value)
        except Exception:
            continue
    return out


def _save_router_registry(path: Path, registry: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_router_indices(
    *,
    checkpoint_root: Path,
    expert_id: str,
    router_num_experts: int,
    router_expert_index: int,
) -> tuple[int, int, Path]:
    registry_path = checkpoint_root / "router" / "router_registry.json"
    num_experts = int(router_num_experts)
    expert_index = int(router_expert_index)

    if num_experts > 0:
        return num_experts, expert_index, registry_path

    expert = str(expert_id).strip()
    if not expert:
        return 0, -1, registry_path

    registry = _load_router_registry(registry_path)
    if expert not in registry:
        next_index = max(registry.values(), default=-1) + 1
        registry[expert] = next_index
        _save_router_registry(registry_path, registry)

    return max(1, len(registry)), int(registry[expert]), registry_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PI05/SmolVLA with LeRobot LoRA-MoE continual finetuning.")
    p.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id, e.g. pick_place/task_x")
    p.add_argument("--dataset-root", required=True, help="Local LeRobot root dir")
    p.add_argument("--output-dir", required=True, help="Output directory for this training round")
    p.add_argument("--policy-path", default="", help="Optional previous checkpoint dir for continual finetuning")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu", action="store_true", help="Force training on CPU (equivalent to --device cpu)")
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--policy-type", default="pi05", choices=["pi05", "smolvla"])
    p.add_argument("--base-model-path", default="checkpoints/pi05_base")
    p.add_argument("--finetune-mode", choices=["full", "lora_moe"], default="lora_moe")
    p.add_argument("--expert-id", default="", help="Expert id/name for LoRA-MoE bookkeeping")
    p.add_argument("--peft-target-modules", default="", help="Optional PEFT target override. Empty uses PI05 default (action head only)")
    p.add_argument("--peft-r", type=int, default=16)
    p.add_argument("--peft-init-type", default="")
    p.add_argument("--router-num-experts", type=int, default=0, help="Router output dimension. 0 means auto-resolve from router registry")
    p.add_argument("--router-expert-index", type=int, default=-1, help="Supervised label index for current expert. -1 means auto-resolve")
    p.add_argument("--router-hidden-dim", type=int, default=512)
    p.add_argument("--router-loss-weight", type=float, default=0.2)
    p.add_argument(
        "--rename-map-json",
        default="auto",
        help="JSON rename map object, or 'auto' to infer from dataset feature keys.",
    )
    p.add_argument("--result-json-out", default="")
    p.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot dataset video backend (e.g. pyav, torchcodec). Default uses pyav to avoid torchcodec ABI issues.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_repo_id = _normalize_model_ref(args.dataset_repo_id)
    dataset_root = _resolve_path(args.dataset_root)
    dataset_dir = _resolve_dataset_dir(dataset_root, dataset_repo_id)
    output_dir = _resolve_path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    policy_path = (
        _resolve_path(_normalize_model_ref(args.policy_path))
        if args.policy_path.strip()
        else None
    )
    base_model_path = _normalize_model_ref(args.base_model_path)
    if str(args.policy_type) == "smolvla" and base_model_path in {"checkpoints/pi05_base", "pi05_base"}:
        base_model_path = "checkpoints/smolvla_base"
    device = "cpu" if bool(args.cpu) else str(args.device).strip()
    if not device:
        device = "cuda"
    rename_map = _resolve_rename_map(args.rename_map_json, dataset_dir, policy_type=str(args.policy_type))
    rename_map_json = json.dumps(rename_map, ensure_ascii=False)
    expert_id = str(args.expert_id).strip()
    router_num_experts, router_expert_index, router_registry_path = _resolve_router_indices(
        checkpoint_root=checkpoint_root,
        expert_id=expert_id,
        router_num_experts=int(args.router_num_experts),
        router_expert_index=int(args.router_expert_index),
    )

    if str(args.policy_type) == "pi05":
        local_paligemma_dir = _resolve_local_paligemma_dir(checkpoint_root)
        if policy_path is not None:
            _rewrite_preprocessor_tokenizer_to_local(policy_path, local_paligemma_dir)
        else:
            resolved_base = Path(_resolve_base_model_ref(base_model_path))
            if resolved_base.exists():
                _rewrite_preprocessor_tokenizer_to_local(resolved_base, local_paligemma_dir)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_dir}",
        f"--output_dir={output_dir}",
        f"--batch_size={int(args.batch_size)}",
        f"--steps={int(args.steps)}",
        "--eval_freq=0",
        f"--save_freq={int(args.save_freq)}",
        f"--log_freq={int(args.log_freq)}",
        f"--seed={int(args.seed)}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--rename_map={rename_map_json}",
        f"--dataset.video_backend={str(args.video_backend).strip() or 'pyav'}",
        f"--policy.router_num_experts={int(router_num_experts)}",
        f"--policy.router_expert_index={int(router_expert_index)}",
        f"--policy.router_hidden_dim={int(args.router_hidden_dim)}",
        f"--policy.router_loss_weight={float(args.router_loss_weight)}",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
    ]
    if policy_path is not None:
        if not policy_path.exists():
            raise FileNotFoundError(
                f"--policy-path does not exist: {policy_path}. "
                "Pass a valid local checkpoint directory (e.g. .../checkpoints/003000/pretrained_model), "
                "or omit --policy-path to start from base pi05."
            )
        cmd.append(f"--policy.path={policy_path.resolve().as_posix()}")
    elif args.finetune_mode == "lora_moe":
        # PEFT training requires a pretrained checkpoint/source model.
        cmd.append(f"--policy.path={_resolve_base_model_ref(base_model_path)}")
    else:
        cmd.append(f"--policy.type={str(args.policy_type)}")
    if args.finetune_mode == "lora_moe":
        cmd.extend(
            [
                f"--peft.r={int(args.peft_r)}",
                "--peft.method_type=LORA",
            ]
        )
        if args.peft_target_modules.strip():
            cmd.append(f"--peft.target_modules={args.peft_target_modules}")
        if args.peft_init_type.strip():
            cmd.append(f"--peft.init_type={args.peft_init_type.strip()}")

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    result = {
        "status": "SUCCESS" if proc.returncode == 0 else "FAIL",
        "exit_code": int(proc.returncode),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": str(dataset_root),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "policy_path": str(policy_path) if policy_path is not None else "",
        "base_model_path": str(base_model_path),
        "policy_type": str(args.policy_type),
        "device": str(device),
        "video_backend": str(args.video_backend).strip() or "pyav",
        "rename_map_json": str(rename_map_json),
        "finetune_mode": args.finetune_mode,
        "expert_id": expert_id,
        "router_num_experts": int(router_num_experts),
        "router_expert_index": int(router_expert_index),
        "router_registry_path": str(router_registry_path),
        "peft_target_modules": args.peft_target_modules if args.finetune_mode == "lora_moe" else "",
        "peft_r": int(args.peft_r) if args.finetune_mode == "lora_moe" else 0,
        "created_at": _now(),
        "train_command": cmd,
    }

    out = args.result_json_out.strip()
    if out:
        out_path = _resolve_path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
