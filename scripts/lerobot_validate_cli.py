import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PALIGEMMA_REMOTE_ID = "google/paligemma-3b-pt-224"

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] [lerobot_validate] {msg}", flush=True)


def _resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _normalize_model_ref(text: str) -> str:
    return str(text or "").strip().replace("\\", "/")


def _primitive_from_repo_id(repo_id: str) -> str:
    text = str(repo_id or "").strip()
    if not text:
        return "unknown"
    if "/" in text:
        return text.split("/", 1)[0].strip() or "unknown"
    return text


def _resolve_base_model_dir(text: str) -> Path:
    ref = _normalize_model_ref(text)
    candidate = Path(ref).expanduser()
    if candidate.is_absolute():
        p = candidate
    else:
        p = PROJECT_ROOT / candidate
    if p.exists():
        return p.resolve()

    # Compatibility aliases for local base checkpoint.
    local_base = (PROJECT_ROOT / "checkpoints" / "pi05_base").resolve()
    if local_base.exists() and ref in {"pi05_base", "checkpoints/pi05_base", "lerobot/pi05_base"}:
        return local_base

    raise FileNotFoundError(f"Base model path does not exist: {p}")


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


def _resolve_dataset_dir(dataset_root: Path, dataset_repo_id: str) -> Path:
    repo_rel = Path(*[seg for seg in str(dataset_repo_id).split("/") if seg.strip()])
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
        return generated_to_smolvla if policy_type_norm == "smolvla" else generated_to_pi05

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        feature_keys = set((info.get("features") or {}).keys())
    except Exception:
        feature_keys = set()

    if policy_type_norm == "smolvla":
        if smolvla_expected.issubset(feature_keys):
            return {}
        if generated_keys & feature_keys:
            return generated_to_smolvla
        return {}

    has_new_image_keys = any(k.startswith("observation.images.") for k in feature_keys)
    has_legacy_image_keys = any(k in feature_keys for k in {"image", "wrist_image"})
    if has_new_image_keys:
        return {}
    if has_legacy_image_keys:
        return generated_to_pi05
    return {}


def _resolve_rename_map(text: str, dataset_dir: Path, policy_type: str) -> dict[str, str]:
    raw = str(text or "").strip()
    if not raw or raw.lower() == "auto":
        return _auto_rename_map_for_dataset(dataset_dir, policy_type=policy_type)

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--rename-map-json must be a JSON object or 'auto'")
    return {str(k): str(v) for k, v in parsed.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline validation for LeRobot continual LoRA-MoE checkpoints.")
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--policy-path", default="", help="Single checkpoint pretrained_model dir")
    p.add_argument(
        "--policy-paths-json",
        default="",
        help='Optional JSON file containing checkpoint list for MoE, e.g. ["ckpt_a","ckpt_b"]',
    )
    p.add_argument("--checkpoint-root", default="checkpoints", help="Checkpoint root for auto policy discovery")
    p.add_argument("--policy-type", default="pi05", choices=["pi05", "smolvla"])
    p.add_argument("--base-model-path", default="checkpoints/pi05_base", help="Base PI0.5 model directory")
    p.add_argument("--primitive", default="", help="Primitive name; defaults to dataset repo_id prefix")
    p.add_argument("--moe-router", choices=["router"], default="router")
    p.add_argument("--router-state-json", default="", help="Optional trained router state JSON")
    p.add_argument(
        "--router-top-k",
        type=int,
        default=1,
        help="Apply Top-K over router probabilities before one-hot routing",
    )
    p.add_argument(
        "--force-full-moe",
        action="store_true",
        help="Always load all latest experts from checkpoint-root for validation",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument(
        "--rename-map-json",
        default="auto",
        help="JSON rename map object, or 'auto' to infer from dataset feature keys.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--result-json-out", default="")
    return p.parse_args()


def _discover_latest_expert_policy_paths(checkpoint_root: Path, primitive: str) -> list[Path]:
    primitive_dir = checkpoint_root / "experts" / primitive
    if not primitive_dir.exists():
        return []
    candidates = [p for p in primitive_dir.glob("**/checkpoints/*/pretrained_model") if p.is_dir()]
    if not candidates:
        return []
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return [latest.resolve()]


def _discover_all_latest_expert_policy_paths(checkpoint_root: Path) -> list[Path]:
    experts_root = checkpoint_root / "experts"
    if not experts_root.exists():
        return []
    latest_paths: list[Path] = []
    for primitive_dir in experts_root.iterdir():
        if not primitive_dir.is_dir():
            continue
        candidates = [p for p in primitive_dir.glob("**/checkpoints/*/pretrained_model") if p.is_dir()]
        if not candidates:
            continue
        latest_paths.append(max(candidates, key=lambda p: p.stat().st_mtime).resolve())
    return sorted(latest_paths)


def _policy_path_to_primitive(policy_path: Path) -> str:
    parts = policy_path.resolve().parts
    if "experts" in parts:
        idx = parts.index("experts")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return policy_path.parent.name


def _extract_batch_task_text(batch: dict[str, Any], default_text: str) -> str:
    for key in ("task", "prompt", "instruction", "text"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
        else:
            text = str(value).strip()
            if text:
                return text
    return default_text


def _load_policy_paths(args: argparse.Namespace) -> tuple[list[Path], str, str]:
    checkpoint_root = _resolve_path(args.checkpoint_root)
    primitive = str(args.primitive).strip() or _primitive_from_repo_id(args.dataset_repo_id)

    if bool(args.force_full_moe):
        all_expert_paths = _discover_all_latest_expert_policy_paths(checkpoint_root)
        if all_expert_paths:
            return all_expert_paths, "auto_lora_moe_all_experts_forced", primitive
        base_path = _resolve_base_model_dir(args.base_model_path)
        return [base_path], "auto_base_only_forced", primitive

    paths: list[Path] = []
    if args.policy_paths_json.strip():
        jp = _resolve_path(args.policy_paths_json)
        raw = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("--policy-paths-json must be a non-empty JSON array")
        paths.extend(_resolve_path(str(x)) for x in raw if str(x).strip())
        return paths, "explicit_policy_paths_json", ""
    elif args.policy_path.strip():
        paths.append(_resolve_path(args.policy_path))
        return paths, "explicit_policy_path", ""

    all_expert_paths = _discover_all_latest_expert_policy_paths(checkpoint_root)
    if all_expert_paths:
        return all_expert_paths, "auto_lora_moe_all_experts", primitive

    base_path = _resolve_base_model_dir(args.base_model_path)
    return [base_path], "auto_base_only", primitive


def _build_policy(
    policy_path: Path,
    ds_meta,
    device: str,
    rename_map: dict[str, str],
    base_model_path: Path | None = None,
):
    try:
        # LeRobot 0.4.x
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError:
        try:
            # Older LeRobot versions
            from lerobot.configs import PreTrainedConfig
        except ImportError:
            # Compatibility fallback for alternative package layouts
            from lerobot.policies import PreTrainedConfig
    make_policy_fn = None
    get_policy_class_fn = None
    try:
        from lerobot.policies import make_policy, make_pre_post_processors

        make_policy_fn = make_policy
    except ImportError:
        try:
            # Some LeRobot releases expose policy builders under factory.
            from lerobot.policies.factory import make_policy, make_pre_post_processors

            make_policy_fn = make_policy
        except ImportError:
            # Last fallback: build policy class directly from config type.
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            get_policy_class_fn = get_policy_class

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    adapter_config_path = policy_path / "adapter_config.json"
    use_adapter = bool(adapter_config_path.exists())
    base_pretrained_path = str(base_model_path) if (use_adapter and base_model_path is not None) else str(policy_path)
    # When loading a PEFT adapter checkpoint, first load a plain base policy and then
    # attach the adapter from `policy_path` explicitly below.
    if use_adapter and hasattr(cfg, "use_peft"):
        cfg.use_peft = False
    cfg.pretrained_path = base_pretrained_path
    cfg.device = device
    if make_policy_fn is not None:
        try:
            policy = make_policy_fn(cfg=cfg, ds_meta=ds_meta, rename_map=rename_map)
        except Exception:
            raise
    else:
        if get_policy_class_fn is None:
            raise RuntimeError("Failed to import LeRobot policy constructors")
        policy_cls = get_policy_class_fn(cfg.type)
        policy = policy_cls.from_pretrained(pretrained_name_or_path=base_pretrained_path, config=cfg)
    if use_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(f"PEFT adapter found at {policy_path} but peft is unavailable") from exc
        policy = PeftModel.from_pretrained(policy, str(policy_path))

    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return policy, preprocessor, postprocessor


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


def _router_index_to_model_index(
    *,
    policy_paths: list[Path],
    checkpoint_root: Path,
    router_dim: int,
) -> dict[int, int]:
    primitive_to_model_index = {
        _policy_path_to_primitive(policy_path): idx for idx, policy_path in enumerate(policy_paths)
    }
    registry = _load_router_registry(checkpoint_root / "router" / "router_registry.json")
    index_to_model: dict[int, int] = {}
    for primitive, router_idx in registry.items():
        model_idx = primitive_to_model_index.get(primitive)
        if model_idx is None:
            continue
        index_to_model[int(router_idx)] = int(model_idx)
    if not index_to_model:
        # Conservative fallback if registry is missing: align by list order.
        for i in range(min(int(router_dim), len(policy_paths))):
            index_to_model[i] = i
    return index_to_model


def main() -> None:
    args = parse_args()
    if str(args.policy_type) == "smolvla" and str(args.base_model_path).strip() in {
        "checkpoints/pi05_base",
        "pi05_base",
    }:
        args.base_model_path = "checkpoints/smolvla_base"
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "lerobot" / "src"))
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    if str(args.policy_type) == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    else:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    dataset_root = _resolve_path(args.dataset_root)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    local_paligemma_dir = _resolve_local_paligemma_dir(checkpoint_root) if str(args.policy_type) == "pi05" else None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset_dir = _resolve_dataset_dir(dataset_root, args.dataset_repo_id)
    rename_map = _resolve_rename_map(args.rename_map_json, dataset_dir, policy_type=str(args.policy_type))
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=str(dataset_dir))
    dataset = LeRobotDataset(args.dataset_repo_id, root=str(dataset_dir))
    policy_paths, policy_source, resolved_primitive = _load_policy_paths(args)
    base_model_dir = _resolve_base_model_dir(args.base_model_path)
    _log(
        "Validation start load: "
        f"dataset={args.dataset_repo_id}, "
        f"base={base_model_dir}, "
        f"policy_source={policy_source}, "
        f"primitive={resolved_primitive}, "
        f"router={args.moe_router}, "
        f"router_state={args.router_state_json or '<auto>'}, "
        f"force_full_moe={bool(args.force_full_moe)}"
    )
    _log(
        "Validation adapters: "
        + (", ".join(str(p) for p in policy_paths) if policy_paths else "<none>")
    )
    if local_paligemma_dir is not None:
        _rewrite_preprocessor_tokenizer_to_local(base_model_dir, local_paligemma_dir)
        for pp in policy_paths:
            _rewrite_preprocessor_tokenizer_to_local(pp, local_paligemma_dir)
    backbone = base_model_dir if policy_source.startswith("auto_lora_moe") else None
    models = [
        _build_policy(pp, ds_meta, args.device, rename_map=rename_map, base_model_path=backbone)
        for pp in policy_paths
    ]
    dl = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), shuffle=True, num_workers=0)

    mse_sum = [0.0 for _ in models]
    routed_mse_sum = 0.0
    routed_count = 0
    n = 0
    routing_fallback_count = 0
    router_used = False
    router_top_k = max(1, int(args.router_top_k))
    router_policy, router_preprocessor, _ = models[0]

    def _unwrap_policy(policy_obj):
        seen: set[int] = set()
        stack = [policy_obj]
        while stack:
            obj = stack.pop()
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if hasattr(obj, "prepare_action") and (
                hasattr(obj, "_preprocess_images") or hasattr(obj, "prepare_images")
            ):
                return obj
            for attr in ("model", "base_model", "module"):
                nxt = getattr(obj, attr, None)
                if nxt is not None:
                    stack.append(nxt)
        return policy_obj

    def _unwrap_core_model(policy_obj):
        routed_policy = _unwrap_policy(policy_obj)
        seen: set[int] = set()
        stack = [getattr(routed_policy, "model", None), getattr(policy_obj, "model", None), policy_obj]
        while stack:
            obj = stack.pop()
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if hasattr(obj, "_compute_router_logits") and hasattr(obj, "embed_prefix"):
                return obj
            for attr in ("model", "base_model", "module"):
                nxt = getattr(obj, attr, None)
                if nxt is not None:
                    stack.append(nxt)
        return None

    with torch.inference_mode():
        for i, batch in enumerate(dl):
            if i >= max(1, int(args.max_batches)):
                break
            batch_mse: list[float] = []
            batch_preds: list[torch.Tensor] = []
            for mi, (policy, preprocessor, postprocessor) in enumerate(models):
                proc = preprocessor(batch)
                pred = policy.select_action(proc)
                pred = postprocessor(pred)
                target = batch["action"][:, 0].to(pred.device) if batch["action"].ndim == 3 else batch["action"].to(pred.device)
                mse = float(torch.mean((pred - target) ** 2).item())
                mse_sum[mi] += mse
                batch_mse.append(mse)
                batch_preds.append(pred)

            proc_router = router_preprocessor(batch)
            routed_policy = _unwrap_policy(router_policy)
            if not hasattr(routed_policy, "prepare_action"):
                raise RuntimeError("Unable to access policy action prep hooks for router inference.")
            core_model = _unwrap_core_model(router_policy)
            if core_model is None or not hasattr(core_model, "forward"):
                raise RuntimeError("Unable to access policy core model for router forward pass.")
            if hasattr(routed_policy, "_preprocess_images"):
                images, img_masks = routed_policy._preprocess_images(proc_router)
            else:
                images, img_masks = routed_policy.prepare_images(proc_router)
            tokens = proc_router["observation.language.tokens"]
            masks = proc_router["observation.language.attention_mask"]
            # Compute router logits from prefix features only (images + language [+state for smolvla]).
            if str(args.policy_type) == "smolvla":
                state = routed_policy.prepare_state(proc_router)
                prefix_embs, prefix_pad_masks, prefix_att_masks = core_model.embed_prefix(
                    images, img_masks, tokens, masks, state
                )
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                (prefix_out, _), _ = core_model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                    fill_kv_cache=True,
                )
            else:
                prefix_embs, prefix_pad_masks, prefix_att_masks = core_model.embed_prefix(images, img_masks, tokens, masks)
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = core_model._prepare_attention_masks_4d(prefix_att_2d_masks)
                (prefix_out, _), _ = core_model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )
            router_logits = core_model._compute_router_logits(prefix_out.to(dtype=torch.float32), prefix_pad_masks)
            if router_logits is None:
                raise RuntimeError(
                    "Router logits are unavailable from checkpoint model. "
                    "Ensure router_num_experts > 0 and router head is trained."
                )
            router_used = True
            probs = torch.softmax(router_logits.to(dtype=torch.float32), dim=-1)
            top_k = min(int(router_top_k), int(probs.shape[-1]))
            one_hot = torch.zeros_like(probs)
            if top_k < int(probs.shape[-1]):
                topk = torch.topk(probs, k=top_k, dim=-1)
                masked = torch.zeros_like(probs)
                masked.scatter_(1, topk.indices, topk.values)
                route_indices = torch.argmax(masked, dim=-1)
                one_hot.scatter_(1, topk.indices, 1.0)
            else:
                route_indices = torch.argmax(probs, dim=-1)
                one_hot.scatter_(1, route_indices.unsqueeze(1), 1.0)
            if probs.shape[0] > 0:
                softmax_vec = [round(float(v), 6) for v in probs[0].detach().cpu().tolist()]
                one_hot_vec = [int(v) for v in one_hot[0].detach().cpu().tolist()]
                _log(
                    f"Router sample0 softmax={softmax_vec} top_k={int(top_k)} one_hot={one_hot_vec}"
                )
            route_map = _router_index_to_model_index(
                policy_paths=policy_paths,
                checkpoint_root=checkpoint_root,
                router_dim=int(probs.shape[-1]),
            )
            for bi in range(route_indices.shape[0]):
                router_idx = int(route_indices[bi].item())
                model_idx = route_map.get(router_idx)
                if model_idx is None or model_idx >= len(batch_preds):
                    routing_fallback_count += 1
                    model_idx = min(router_idx, len(batch_preds) - 1)
                pred_b = batch_preds[model_idx][bi]
                target_b = target[bi].to(pred_b.device)
                routed_mse_sum += float(torch.mean((pred_b - target_b) ** 2).item())
                routed_count += 1
            n += 1

    expert_metrics = [
        {
            "policy_path": str(policy_paths[i]),
            "offline_action_mse": (mse_sum[i] / n) if n > 0 else float("inf"),
        }
        for i in range(len(policy_paths))
    ]
    result = {
        "status": "SUCCESS",
        "dataset_repo_id": args.dataset_repo_id,
        "primitive": resolved_primitive,
        "policy_source": policy_source,
        "base_model_path": str(base_model_dir),
        "policy_path": str(policy_paths[0]),
        "policy_paths": [str(p) for p in policy_paths],
        "moe_router": args.moe_router,
        "router_top_k": int(router_top_k),
        "router_used": bool(router_used),
        "routing_fallback_count": int(routing_fallback_count),
        "num_batches": n,
        "rename_map": rename_map,
        "offline_action_mse": (routed_mse_sum / routed_count) if routed_count > 0 else float("inf"),
        "experts": expert_metrics,
        "created_at": _now(),
    }

    out = args.result_json_out.strip()
    if out:
        out_path = _resolve_path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
