import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SMOLVLA_LIBERO_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
ALLOWED_PRIMITIVES = {"pick_place", "push", "pull", "press", "open", "close", "pour"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] [libero_long_pipeline] {msg}", flush=True)


def _resolve_path(text: str) -> Path:
    p = Path(str(text)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    rc = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def _normalize_model_ref(text: str) -> str:
    return str(text or "").strip().replace("\\", "/")


def _resolve_dataset_dir(dataset_root: Path, dataset_repo_id: str) -> Path:
    repo_id_norm = _normalize_model_ref(dataset_repo_id)
    repo_rel = Path(*[seg for seg in repo_id_norm.split("/") if seg.strip()])
    candidate_repo = dataset_root / repo_rel
    if (candidate_repo / "meta" / "info.json").exists():
        return candidate_repo
    if (dataset_root / "meta" / "info.json").exists():
        return dataset_root
    return candidate_repo


def _read_expert_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"experts": {}}
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return {"experts": {}}
    if not isinstance(raw.get("experts"), dict):
        raw["experts"] = {}
    return raw


def _write_expert_state(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _load_router_registry(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = _read_json(path)
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
    _write_json(path, registry)


def _resolve_router_indices(checkpoint_root: Path, expert_id: str) -> tuple[int, int]:
    registry_path = checkpoint_root / "router" / "router_registry.json"
    expert = str(expert_id).strip()
    if not expert:
        return 0, -1
    registry = _load_router_registry(registry_path)
    if expert not in registry:
        registry[expert] = max(registry.values(), default=-1) + 1
        _save_router_registry(registry_path, registry)
    return max(1, len(registry)), int(registry[expert])


def _discover_latest_expert_policy(checkpoint_root: Path, primitive: str) -> Path | None:
    primitive_dir = checkpoint_root / "experts" / primitive
    if not primitive_dir.exists():
        return None
    candidates = [p for p in primitive_dir.glob("**/checkpoints/*/pretrained_model") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_pretrained_model_dir(train_output_dir: Path) -> Path:
    candidates = sorted((train_output_dir / "checkpoints").glob("*/pretrained_model"))
    if not candidates:
        raise RuntimeError(f"No pretrained_model found under {train_output_dir}")
    return candidates[-1]


def _alloc_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    idx = 1
    while True:
        cand = base_dir.parent / f"{base_dir.name}_v{idx:03d}"
        if not cand.exists():
            return cand
        idx += 1


def _extract_eval_metric_files(eval_output_dir: Path) -> list[Path]:
    if not eval_output_dir.exists():
        return []
    out: list[Path] = []
    for p in eval_output_dir.glob("**/*.json"):
        name = p.name.lower()
        if "metrics" in name or "result" in name or "summary" in name:
            out.append(p)
    return sorted(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIBERO-Long continual training+evaluation pipeline for PI05 LoRA-MoE.")
    p.add_argument("--dataset-repo-id", default="HuggingFaceVLA/libero")
    p.add_argument("--dataset-root", default="~/.cache/huggingface/lerobot")
    p.add_argument("--policy-type", default="smolvla", choices=["pi05", "smolvla"])
    p.add_argument("--task-suite", default="libero_10", choices=["libero_10"])
    p.add_argument("--num-tasks", type=int, default=10)
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--train-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--replay-rate", type=float, default=0.2)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--eval-device", default="cuda")
    p.add_argument("--eval-control-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--checkpoint-root", default="checkpoints")
    p.add_argument("--loop-log-dir", default="logs/libero_long_pipeline")
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "train", "eval"],
        help="Run full pipeline, training only, or evaluation only.",
    )
    p.add_argument(
        "--primitive-map-json",
        default="",
        help="Optional JSON object mapping task_id(string/int) -> primitive_name",
    )
    p.add_argument(
        "--base-model-path",
        default="checkpoints/pi05_base",
        help="Base PI05 checkpoint path used when no previous expert exists",
    )
    p.add_argument(
        "--primitive-semantic-mapping",
        action="store_true",
        help="Infer primitive by task semantics via TaskSupervisorAgent (safe fallback to id-based naming).",
    )
    p.add_argument(
        "--primitive-agent-model",
        default="qwen/qwen3.5-397b-a17b",
        help="Model name for TaskSupervisorAgent when --primitive-semantic-mapping is enabled.",
    )
    p.add_argument(
        "--primitive-map-cache-json",
        default="",
        help="Path to task_id->primitive mapping JSON. If missing in train stage, it will be generated before training.",
    )
    p.add_argument(
        "--refresh-primitive-map",
        action="store_true",
        help="Regenerate task_id->primitive mapping JSON even if cache file exists.",
    )
    return p.parse_args()


def _resolve_task_primitive(task_id: int, mapping: dict[str, str]) -> str:
    key = str(int(task_id))
    name = str(mapping.get(key, "")).strip()
    if name:
        return name
    return f"libero_long_task_{int(task_id):02d}"


def _normalize_primitive_name(text: str) -> str:
    v = str(text or "").strip().lower()
    if not v:
        return ""
    aliases = {
        "pickplace": "pick_place",
        "pick-and-place": "pick_place",
        "pick_and_place": "pick_place",
    }
    v = aliases.get(v, v)
    return v if v in ALLOWED_PRIMITIVES else ""


def _heuristic_primitive_from_text(task_prompt: str) -> str:
    t = str(task_prompt or "").lower()
    if any(x in t for x in ["open", "unlatch", "unlock"]):
        return "open"
    if any(x in t for x in ["close", "shut"]):
        return "close"
    if any(x in t for x in ["press", "button", "switch"]):
        return "press"
    if "pour" in t:
        return "pour"
    if any(x in t for x in ["pull", "drag"]):
        return "pull"
    if any(x in t for x in ["push", "slide"]):
        return "push"
    if any(x in t for x in ["pick", "place", "put", "insert", "stack"]):
        return "pick_place"
    return ""


def _load_libero_task_prompts(task_suite: str, num_tasks: int) -> dict[int, str]:
    try:
        from libero.libero import benchmark
    except Exception:
        return {}
    try:
        bench = benchmark.get_benchmark_dict()
        if task_suite not in bench:
            return {}
        suite = bench[task_suite]()
        tasks = list(getattr(suite, "tasks", []) or [])
        out: dict[int, str] = {}
        for idx in range(min(int(num_tasks), len(tasks))):
            out[idx] = str(getattr(tasks[idx], "language", "")).strip()
        return out
    except Exception:
        return {}


class PrimitiveResolver:
    def __init__(self, *, mapping: dict[str, str], enable_semantic: bool, model: str):
        self.mapping = dict(mapping)
        self.enable_semantic = bool(enable_semantic)
        self.model = str(model).strip()
        self._supervisor = None

    def _ensure_supervisor(self):
        if self._supervisor is not None:
            return self._supervisor
        if not self.enable_semantic:
            return None
        try:
            from scripts.evoma import TaskSupervisorAgent, _get_client

            self._supervisor = TaskSupervisorAgent(_get_client(), model=self.model)
        except Exception:
            self._supervisor = None
        return self._supervisor

    def resolve(self, task_id: int, task_prompt: str) -> str:
        key = str(int(task_id))
        direct = _resolve_task_primitive(task_id, self.mapping)
        if key in self.mapping and str(self.mapping.get(key, "")).strip():
            return direct

        inferred = _heuristic_primitive_from_text(task_prompt)
        if inferred:
            return inferred

        supervisor = self._ensure_supervisor()
        if supervisor is not None and str(task_prompt).strip():
            try:
                plan = supervisor.analyze_task(
                    task_prompt=str(task_prompt).strip(),
                    scene_context={
                        "suite": "libero",
                        "task_prompt": str(task_prompt).strip(),
                        "note": "Infer one primitive skill type for expert routing.",
                    },
                    image_path=str(PROJECT_ROOT / "logs" / "current_view.png"),
                )
                atomic_tasks = list(plan.get("atomic_tasks") or [])
                if atomic_tasks:
                    candidate = _normalize_primitive_name(atomic_tasks[0].get("primitive", ""))
                    if candidate:
                        return candidate
            except Exception:
                pass
        return direct


def _extract_task_ids_from_dataset(dataset_dir: Path, num_tasks: int) -> list[int]:
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        try:
            info = _read_json(info_path)
            total = int(info.get("total_tasks", 0))
            if total > 0:
                return list(range(min(int(num_tasks), total)))
        except Exception:
            pass
    return list(range(int(num_tasks)))


def _build_task_primitive_mapping(
    *,
    task_ids: list[int],
    task_prompts: dict[int, str],
    primitive_map_override: dict[str, str],
    enable_semantic: bool,
    model: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    resolver = PrimitiveResolver(
        mapping=primitive_map_override,
        enable_semantic=enable_semantic,
        model=model,
    )
    mapping: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        key = str(int(task_id))
        prompt = str(task_prompts.get(int(task_id), "")).strip()
        primitive = resolver.resolve(int(task_id), prompt)
        if prompt:
            mapping[prompt] = primitive
        records.append(
            {
                "task_id": int(task_id),
                "task_prompt": prompt,
                "primitive": primitive,
            }
        )
    return mapping, records


def _primitive_from_prompt_mapping(
    *,
    task_id: int,
    task_prompt: str,
    prompt_to_primitive: dict[str, str],
    fallback_taskid_map: dict[str, str] | None = None,
) -> str:
    prompt = str(task_prompt or "").strip()
    if prompt:
        found = str(prompt_to_primitive.get(prompt, "")).strip()
        if found:
            return found
    fallback = fallback_taskid_map or {}
    return _resolve_task_primitive(task_id, fallback)


def _build_train_cmd(
    *,
    dataset_repo_id: str,
    dataset_dir: Path,
    output_dir: Path,
    policy_path: Path | None,
    checkpoint_root: Path,
    base_model_path: str,
    policy_type: str,
    train_steps: int,
    train_batch_size: int,
    train_device: str,
    expert_id: str,
    task_suite: str,
    task_id: int,
) -> list[str]:
    router_num_experts, router_expert_index = _resolve_router_indices(checkpoint_root, expert_id)
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_dir}",
        f"--output_dir={output_dir}",
        f"--batch_size={int(train_batch_size)}",
        f"--steps={int(train_steps)}",
        "--eval_freq=0",
        "--save_freq=1000",
        "--log_freq=50",
        "--seed=42",
        f"--policy.device={train_device}",
        "--policy.push_to_hub=false",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
        "--peft.method_type=LORA",
        "--peft.r=16",
        f"--policy.router_num_experts={int(router_num_experts)}",
        f"--policy.router_expert_index={int(router_expert_index)}",
        "--policy.router_hidden_dim=512",
        "--policy.router_loss_weight=0.2",
        "--env.type=libero",
        f"--env.task={task_suite}",
        f"--env.task_ids=[{int(task_id)}]",
    ]
    if str(policy_type).strip() == "smolvla" and str(task_suite).strip() == "libero_10":
        # LIBERO observations expose image/image2; SmolVLA base expects camera1/2/3.
        # Map two available camera keys and declare one empty camera slot.
        cmd.append(f"--rename_map={json.dumps(SMOLVLA_LIBERO_RENAME_MAP, ensure_ascii=False)}")
        cmd.append("--policy.empty_cameras=1")
    if policy_path is not None:
        cmd.append(f"--policy.path={policy_path.resolve().as_posix()}")
    else:
        cmd.append(f"--policy.path={_normalize_model_ref(base_model_path)}")
    return cmd


def _build_eval_cmd(
    *,
    policy_path: Path,
    output_dir: Path,
    task_suite: str,
    task_id: int,
    eval_episodes: int,
    eval_batch_size: int,
    eval_control_mode: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path.resolve().as_posix()}",
        "--env.type=libero",
        f"--env.task={task_suite}",
        f"--env.task_ids=[{int(task_id)}]",
        f"--eval.n_episodes={int(eval_episodes)}",
        f"--eval.batch_size={int(eval_batch_size)}",
        "--env.max_parallel_tasks=1",
        f"--env.control_mode={eval_control_mode}",
        f"--output_dir={output_dir}",
    ]


def main() -> None:
    args = parse_args()
    dataset_root = _resolve_path(args.dataset_root)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    loop_dir = _resolve_path(args.loop_log_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    loop_dir.mkdir(parents=True, exist_ok=True)
    expert_state_path = checkpoint_root / "expert_registry.json"
    expert_state = _read_expert_state(expert_state_path)
    replay_rate = float(max(0.0, min(1.0, float(args.replay_rate))))

    primitive_map: dict[str, str] = {}
    if str(args.primitive_map_json).strip():
        primitive_map_raw = _read_json(_resolve_path(args.primitive_map_json))
        if isinstance(primitive_map_raw, dict):
            primitive_map = {str(k): str(v) for k, v in primitive_map_raw.items() if str(v).strip()}

    dataset_dir = _resolve_dataset_dir(dataset_root, str(args.dataset_repo_id))
    primitive_map_cache_path = (
        _resolve_path(args.primitive_map_cache_json)
        if str(args.primitive_map_cache_json).strip()
        else (loop_dir / "task_primitive_mapping.json")
    )
    task_prompts = _load_libero_task_prompts(str(args.task_suite), int(args.num_tasks))
    if str(args.policy_type) == "smolvla" and str(args.base_model_path).strip() in {
        "checkpoints/pi05_base",
        "pi05_base",
    }:
        args.base_model_path = "checkpoints/smolvla_base"

    use_cached_primitive_map = primitive_map_cache_path.exists() and not bool(args.refresh_primitive_map)
    if str(args.stage) in {"all", "train"} and not use_cached_primitive_map:
        task_ids = _extract_task_ids_from_dataset(dataset_dir, int(args.num_tasks))
        generated_map, mapping_records = _build_task_primitive_mapping(
            task_ids=task_ids,
            task_prompts=task_prompts,
            primitive_map_override=primitive_map,
            enable_semantic=bool(args.primitive_semantic_mapping),
            model=str(args.primitive_agent_model),
        )
        mapping_payload = {
            "created_at": _now(),
            "dataset_repo_id": str(args.dataset_repo_id),
            "task_suite": str(args.task_suite),
            "num_tasks": int(args.num_tasks),
            "task_prompt_to_primitive": generated_map,
            "records": mapping_records,
        }
        _write_json(primitive_map_cache_path, mapping_payload)
        _log(f"Primitive mapping generated: {primitive_map_cache_path}")

    if not primitive_map_cache_path.exists():
        raise RuntimeError(
            f"Primitive mapping JSON not found: {primitive_map_cache_path}. "
            "Run train stage first or provide --primitive-map-cache-json."
        )
    primitive_mapping_payload = _read_json(primitive_map_cache_path)
    prompt_map_raw = primitive_mapping_payload.get("task_prompt_to_primitive", {})
    if not isinstance(prompt_map_raw, dict):
        prompt_map_raw = {}
    taskid_map_raw = primitive_mapping_payload.get("task_id_to_primitive", {})
    if not isinstance(taskid_map_raw, dict):
        taskid_map_raw = {}
    if not prompt_map_raw and not taskid_map_raw:
        raise RuntimeError(f"Invalid primitive mapping JSON: {primitive_map_cache_path}")

    effective_prompt_map = {
        str(k).strip(): str(v).strip()
        for k, v in prompt_map_raw.items()
        if str(k).strip() and str(v).strip()
    }
    effective_taskid_map = {
        str(k).strip(): str(v).strip()
        for k, v in taskid_map_raw.items()
        if str(k).strip() and str(v).strip()
    }

    summary: dict[str, Any] = {
        "status": "RUNNING",
        "created_at": _now(),
        "stage": str(args.stage),
        "policy_type": str(args.policy_type),
        "task_suite": str(args.task_suite),
        "num_tasks": int(args.num_tasks),
        "train_steps": int(args.train_steps),
        "replay_rate": replay_rate,
        "checkpoint_root": str(checkpoint_root),
        "dataset_repo_id": str(args.dataset_repo_id),
        "dataset_root": str(dataset_root),
        "dataset_dir": str(dataset_dir),
        "primitive_map_cache_json": str(primitive_map_cache_path),
        "train_rounds": [],
        "evaluation": [],
    }
    _write_json(loop_dir / "summary.json", summary)

    env = os.environ.copy()
    # Do not force offline mode here. If dataset isn't cached locally,
    # LeRobot should be able to fetch metadata/assets from HuggingFace.
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)
    env.setdefault("MUJOCO_GL", "egl")

    if str(args.stage) in {"all", "train"}:
        for task_id in range(int(args.num_tasks)):
            task_prompt = str(task_prompts.get(task_id, "")).strip()
            primitive = _primitive_from_prompt_mapping(
                task_id=task_id,
                task_prompt=task_prompt,
                prompt_to_primitive=effective_prompt_map,
                fallback_taskid_map=effective_taskid_map,
            )
            synthetic_repo_id = f"{args.dataset_repo_id}#{args.task_suite}:task_{task_id:02d}:all"
            experts = expert_state.setdefault("experts", {})
            primitive_state = experts.get(primitive, {})
            prev_policy_path = str(primitive_state.get("policy_path", "")).strip()
            if prev_policy_path and not Path(prev_policy_path).exists():
                prev_policy_path = ""
            if not prev_policy_path:
                discovered = _discover_latest_expert_policy(checkpoint_root, primitive)
                if discovered is not None:
                    prev_policy_path = str(discovered)
            has_existing_expert = bool(prev_policy_path)
            history_repo_ids = [str(x).strip() for x in primitive_state.get("history_repo_ids", []) if str(x).strip()]

            replay_repo_id = ""
            replay_steps = 0
            current_steps = int(args.train_steps)
            replay_candidates = [x for x in history_repo_ids if x and x != synthetic_repo_id]
            replay_candidates = list(dict.fromkeys(replay_candidates))
            if has_existing_expert and replay_candidates and replay_rate > 0.0:
                replay_steps = max(1, int(round(int(args.train_steps) * replay_rate)))
                current_steps = max(1, int(args.train_steps) - replay_steps)
                replay_repo_id = random.choice(replay_candidates)

            last_policy = Path(prev_policy_path) if prev_policy_path else None

            round_log: dict[str, Any] = {
                "task_id": int(task_id),
                "primitive": primitive,
                "task_prompt": task_prompt,
                "dataset_marker_repo_id": synthetic_repo_id,
                "has_existing_expert": has_existing_expert,
                "replay_repo_id": replay_repo_id,
                "replay_steps": int(replay_steps),
                "current_steps": int(current_steps),
            }

            if replay_repo_id and replay_steps > 0:
                replay_base = checkpoint_root / "experts" / primitive / f"task_{task_id:02d}_replay"
                replay_out = _alloc_unique_dir(replay_base)
                _log(
                    f"[task {task_id}] replay train start | primitive={primitive} "
                    f"adapter_init={last_policy or '<none>'} marker={replay_repo_id}"
                )
                replay_cmd = _build_train_cmd(
                    dataset_repo_id=str(args.dataset_repo_id),
                    dataset_dir=dataset_dir,
                    output_dir=replay_out,
                    policy_path=last_policy,
                    checkpoint_root=checkpoint_root,
                    base_model_path=str(args.base_model_path),
                    policy_type=str(args.policy_type),
                    train_steps=int(replay_steps),
                    train_batch_size=int(args.train_batch_size),
                    train_device=str(args.train_device),
                    expert_id=primitive,
                    task_suite=str(args.task_suite),
                    task_id=int(task_id),
                )
                _run(replay_cmd, env=env)
                last_policy = _latest_pretrained_model_dir(replay_out)
                round_log["replay_output_dir"] = str(replay_out)
                round_log["replay_policy_path"] = str(last_policy)

            current_base = checkpoint_root / "experts" / primitive / f"task_{task_id:02d}_current"
            current_out = _alloc_unique_dir(current_base)
            _log(
                f"[task {task_id}] current train start | primitive={primitive} "
                f"adapter_init={last_policy or '<none>'} episodes=<all>"
            )
            train_cmd = _build_train_cmd(
                dataset_repo_id=str(args.dataset_repo_id),
                dataset_dir=dataset_dir,
                output_dir=current_out,
                policy_path=last_policy,
                checkpoint_root=checkpoint_root,
                base_model_path=str(args.base_model_path),
                policy_type=str(args.policy_type),
                train_steps=int(current_steps),
                train_batch_size=int(args.train_batch_size),
                train_device=str(args.train_device),
                expert_id=primitive,
                task_suite=str(args.task_suite),
                task_id=int(task_id),
            )
            _run(train_cmd, env=env)
            latest_policy = _latest_pretrained_model_dir(current_out)

            experts[primitive] = {
                "policy_path": str(latest_policy),
                "history_repo_ids": [x for x in history_repo_ids if x != synthetic_repo_id] + [synthetic_repo_id],
                "updated_at": _now(),
                "last_task_id": int(task_id),
            }
            _write_expert_state(expert_state_path, expert_state)

            round_log["current_output_dir"] = str(current_out)
            round_log["latest_policy_path"] = str(latest_policy)
            summary["train_rounds"].append(round_log)
            _write_json(loop_dir / "summary.json", summary)

    if str(args.stage) == "all":
        _log("All LIBERO-Long tasks trained, start evaluation")
    elif str(args.stage) == "eval":
        _log("Eval-only mode: skip training, evaluate from existing expert registry")

    if str(args.stage) in {"all", "eval"}:
        # Reload in eval-only mode to ensure we read latest persisted checkpoints.
        expert_state = _read_expert_state(expert_state_path)
        eval_items: list[dict[str, Any]] = []
        for task_id in range(int(args.num_tasks)):
            task_prompt = str(task_prompts.get(task_id, "")).strip()
            primitive = _primitive_from_prompt_mapping(
                task_id=task_id,
                task_prompt=task_prompt,
                prompt_to_primitive=effective_prompt_map,
                fallback_taskid_map=effective_taskid_map,
            )
            primitive_state = expert_state.get("experts", {}).get(primitive, {})
            policy_path = Path(str(primitive_state.get("policy_path", "")).strip())
            if not str(policy_path).strip() or not policy_path.exists():
                raise RuntimeError(f"Missing trained policy for task_id={task_id}, primitive={primitive}")
            eval_out = loop_dir / "eval" / f"task_{task_id:02d}"
            eval_out.mkdir(parents=True, exist_ok=True)
            eval_cmd = _build_eval_cmd(
                policy_path=policy_path,
                output_dir=eval_out,
                task_suite=str(args.task_suite),
                task_id=int(task_id),
                eval_episodes=int(args.eval_episodes),
                eval_batch_size=int(args.eval_batch_size),
                eval_control_mode=str(args.eval_control_mode),
            )
            _log(f"[eval task {task_id}] start | primitive={primitive} policy={policy_path}")
            _run(eval_cmd, env=env)
            metric_files = _extract_eval_metric_files(eval_out)
            eval_items.append(
                {
                    "task_id": int(task_id),
                    "primitive": primitive,
                    "policy_path": str(policy_path),
                    "eval_output_dir": str(eval_out),
                    "metric_files": [str(x) for x in metric_files],
                }
            )
            summary["evaluation"] = eval_items
            _write_json(loop_dir / "summary.json", summary)

    summary["status"] = "SUCCESS"
    summary["finished_at"] = _now()
    summary["notes"] = [
        "Training uses LoRA-MoE continual finetuning with replay logic aligned to scripts/pipeline_lerobot.py.",
        "Before training, task_id->primitive mapping is generated and cached to JSON, then reused by train/eval.",
        "Each task training stage requests env.task_ids=[task_id] and uses all trajectories of that task.",
        "Evaluation runs lerobot_eval on libero_10 task-by-task and stores discovered metric JSON file paths.",
        "Use --stage=train or --stage=eval to split training/evaluation into standalone runs.",
    ]
    _write_json(loop_dir / "summary.json", summary)
    _log(f"Done. Summary: {loop_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
