import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] [pipeline_lerobot] {msg}", flush=True)


def _resolve_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    rc = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot-based closed loop for atomic task chain.")
    p.add_argument("--scene-json", required=True)
    p.add_argument("--task", default="")
    p.add_argument("--atomic-tasks-json", default="")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--eval-loops", type=int, default=3)
    p.add_argument("--eval-min-success", type=int, default=2)
    p.add_argument("--generate-success-target", type=int, default=10)
    p.add_argument("--generate-max-attempts", type=int, default=0)
    p.add_argument("--generate-lerobot-root", default="~/.cache/huggingface/lerobot")
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--train-device", default="cuda", choices=["cuda", "cpu"], help="Device for training steps")
    p.add_argument("--policy-type", default="smolvla", choices=["pi05", "smolvla"], help="Backbone policy architecture for LoRA-MoE experts")
    p.add_argument("--base-model-path", default="", help="Optional base model path; defaults to checkpoints/<policy>_base")
    p.add_argument("--replay-rate", type=float, default=0.2, help="Replay ratio for existing primitive expert")
    p.add_argument("--val-max-batches", type=int, default=20)
    p.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Root directory for expert checkpoints and registry state",
    )
    p.add_argument("--loop-log-dir", default="logs/closed_loop_lerobot")
    return p.parse_args()


def _load_atomic_tasks(task: str, atomic_tasks_json: str) -> list[str]:
    if atomic_tasks_json.strip():
        raw = json.loads(_resolve_path(atomic_tasks_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("--atomic-tasks-json must be non-empty array")
        return [str(x).strip() for x in raw if str(x).strip()]
    if not task.strip():
        raise ValueError("Provide --task or --atomic-tasks-json")
    return [x.strip() for x in task.split(",") if x.strip()]


def _primitive_from_repo_id(repo_id: str) -> str:
    text = str(repo_id or "").strip()
    if not text:
        return "unknown"
    if "/" in text:
        return text.split("/", 1)[0].strip() or "unknown"
    return text


def _latest_pretrained_model_dir(train_output_dir: Path) -> Path:
    ckpt = train_output_dir / "checkpoints"
    candidates = sorted(ckpt.glob("*/pretrained_model"))
    if not candidates:
        raise RuntimeError(f"No checkpoint found under: {ckpt}")
    return candidates[-1]


def _discover_latest_expert_policy(checkpoint_root: Path, primitive: str) -> Path | None:
    primitive_dir = checkpoint_root / "experts" / primitive
    if not primitive_dir.exists():
        return None
    candidates = [p for p in primitive_dir.glob("**/checkpoints/*/pretrained_model") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _alloc_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    idx = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_v{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def _read_expert_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"experts": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {"experts": {}}
    experts = raw.get("experts")
    if not isinstance(experts, dict):
        raw["experts"] = {}
    return raw


def _write_expert_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_pipeline_state(tasks: list[str], base_scene_json: Path) -> dict[str, Any]:
    return {
        "tasks": list(tasks),
        "base_scene_json": str(base_scene_json),
        "task_states": [
            {
                "status": "pending",
                "start_scene_json": "",
                "last_eval_scene_json": "",
                "repo_ids": [],
                "primitive": "",
                "updated_at": "",
            }
            for _ in tasks
        ],
        "updated_at": _now(),
    }


def _load_pipeline_state(path: Path, tasks: list[str], base_scene_json: Path) -> dict[str, Any]:
    expected_tasks = [str(x).strip() for x in tasks]
    if not path.exists():
        return _default_pipeline_state(expected_tasks, base_scene_json)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_pipeline_state(expected_tasks, base_scene_json)

    if not isinstance(raw, dict):
        return _default_pipeline_state(expected_tasks, base_scene_json)

    loaded_tasks = raw.get("tasks")
    if not isinstance(loaded_tasks, list):
        return _default_pipeline_state(expected_tasks, base_scene_json)
    loaded_tasks = [str(x).strip() for x in loaded_tasks]
    if loaded_tasks != expected_tasks:
        return _default_pipeline_state(expected_tasks, base_scene_json)

    task_states = raw.get("task_states")
    if not isinstance(task_states, list) or len(task_states) != len(expected_tasks):
        return _default_pipeline_state(expected_tasks, base_scene_json)

    normalized_states: list[dict[str, Any]] = []
    for st in task_states:
        if not isinstance(st, dict):
            st = {}
        repo_ids = st.get("repo_ids", [])
        if not isinstance(repo_ids, list):
            repo_ids = []
        normalized_states.append(
            {
                "status": str(st.get("status", "pending")).strip().lower() or "pending",
                "start_scene_json": str(st.get("start_scene_json", "")).strip(),
                "last_eval_scene_json": str(st.get("last_eval_scene_json", "")).strip(),
                "repo_ids": [str(x).strip() for x in repo_ids if str(x).strip()],
                "primitive": str(st.get("primitive", "")).strip(),
                "updated_at": str(st.get("updated_at", "")).strip(),
            }
        )

    base_scene = str(raw.get("base_scene_json", "")).strip() or str(base_scene_json)
    return {
        "tasks": expected_tasks,
        "base_scene_json": base_scene,
        "task_states": normalized_states,
        "updated_at": str(raw.get("updated_at", "")).strip(),
    }


def _write_pipeline_state(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now()
    _write_json(path, payload)


def _extract_eval_finished_scene_json(eval_result: dict[str, Any], prefer_success: bool) -> str:
    if prefer_success:
        fs = str(eval_result.get("last_successful_finished_scene_json", "")).strip()
        if fs:
            return fs
    rollout = eval_result.get("rollout")
    if isinstance(rollout, dict):
        return str(rollout.get("finished_scene_json", "")).strip()
    return ""


def _cache_atomic_scene_checkpoint(
    source_scene_json: str,
    *,
    loop_dir: Path,
    atomic_index: int,
    round_index: int,
    label: str,
) -> str:
    src = Path(str(source_scene_json).strip())
    if not str(source_scene_json).strip() or not src.exists() or not src.is_file():
        return ""

    checkpoint_dir = loop_dir / "scene_checkpoints" / f"atomic_{atomic_index:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / f"round_{round_index:03d}_{label}.json"
    latest = checkpoint_dir / "latest.json"
    shutil.copy2(src, dest)
    shutil.copy2(dest, latest)
    return str(dest)


def _build_train_cmd(
    *,
    dataset_repo_id: str,
    dataset_root: str,
    output_dir: Path,
    checkpoint_root: Path,
    policy_path: str,
    steps: int,
    batch_size: int,
    device: str,
    expert_id: str,
    result_json_out: Path,
    policy_type: str,
    base_model_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "lerobot_train_cli.py"),
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-root",
        dataset_root,
        "--output-dir",
        str(output_dir),
        "--checkpoint-root",
        str(checkpoint_root),
        "--policy-type",
        str(policy_type),
        "--base-model-path",
        str(base_model_path),
        "--steps",
        str(int(steps)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--finetune-mode",
        "lora_moe",
        "--expert-id",
        expert_id,
        "--result-json-out",
        str(result_json_out),
    ]
    if str(policy_path).strip():
        cmd.extend(["--policy-path", str(policy_path)])
    return cmd


def main() -> None:
    args = parse_args()
    scene_json = _resolve_path(args.scene_json)
    loop_dir = _resolve_path(args.loop_log_dir)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    loop_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks = _load_atomic_tasks(args.task, args.atomic_tasks_json)
    replay_rate = float(max(0.0, min(1.0, args.replay_rate)))
    default_base_model_path = checkpoint_root / ("smolvla_base" if str(args.policy_type) == "smolvla" else "pi05_base")
    resolved_base_model_path = _resolve_path(args.base_model_path) if str(args.base_model_path).strip() else default_base_model_path
    expert_state_path = checkpoint_root / "expert_registry.json"
    pipeline_state_path = checkpoint_root / "pipeline_state.json"
    router_state_path = checkpoint_root / "router" / "router_state.json"
    legacy_expert_state_path = loop_dir / "expert_registry.json"
    if not expert_state_path.exists() and legacy_expert_state_path.exists():
        expert_state = _read_expert_state(legacy_expert_state_path)
        _write_expert_state(expert_state_path, expert_state)
    else:
        expert_state = _read_expert_state(expert_state_path)

    pipeline_state = _load_pipeline_state(pipeline_state_path, tasks, scene_json)
    task_states: list[dict[str, Any]] = pipeline_state["task_states"]

    start_atomic_index = 0
    while start_atomic_index < len(tasks) and task_states[start_atomic_index].get("status") == "verified":
        start_atomic_index += 1

    if start_atomic_index <= 0:
        current_scene_json = scene_json
    else:
        prev_scene_text = str(task_states[start_atomic_index - 1].get("last_eval_scene_json", "")).strip()
        prev_scene = Path(prev_scene_text) if prev_scene_text else None
        current_scene_json = prev_scene if prev_scene is not None and prev_scene.exists() else scene_json

    _log(
        "Pipeline start: "
        f"resume_atomic_index={start_atomic_index}, "
        f"resume_scene={current_scene_json}, "
        f"base_model={resolved_base_model_path}, "
        f"policy_type={args.policy_type}"
    )

    summary: dict[str, Any] = {
        "status": "RUNNING",
        "tasks": tasks,
        "created_at": _now(),
        "checkpoint_root": str(checkpoint_root),
        "policy_type": str(args.policy_type),
        "base_model_path": str(resolved_base_model_path),
        "expert_state_path": str(expert_state_path),
        "pipeline_state_path": str(pipeline_state_path),
        "router_state_path": str(router_state_path),
        "router_samples": [],
        "resume_start_atomic_index": int(start_atomic_index),
        "resume_scene_json": str(current_scene_json),
        "rounds": [],
    }
    _write_json(loop_dir / "summary.json", summary)

    last_policy_path_for_summary = ""

    for ti in range(start_atomic_index, len(tasks)):
        task = tasks[ti]
        task_state = task_states[ti]
        task_state["start_scene_json"] = str(current_scene_json)
        if task_state.get("status") != "verified":
            task_state["status"] = "pending"
        task_state["updated_at"] = _now()
        _write_pipeline_state(pipeline_state_path, pipeline_state)

        verified = False
        for r in range(1, max(1, args.max_rounds) + 1):
            rd = loop_dir / f"atomic_{ti:03d}" / f"round_{r:03d}"
            rd.mkdir(parents=True, exist_ok=True)
            eval_json = rd / "evaluate_result.json"
            gen_json = rd / "generate_result.json"
            train_json = rd / "train_result.json"
            val_json = rd / "validate_result.json"

            eval_cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_cli.py"),
                "--scene-json",
                str(current_scene_json),
                "--task",
                task,
                "--base-model-path",
                str(resolved_base_model_path),
                "--eval-loops",
                str(int(args.eval_loops)),
                "--result-json-out",
                str(eval_json),
                "--inference-backend",
                "pytorch",
            ]
            _log(
                f"[atomic {ti} round {r}] Validation start | "
                f"load(base={resolved_base_model_path}, router=composer, backend=pytorch, force_full_moe=true)"
            )
            _run(eval_cmd)
            ev = _read_json(eval_json)
            success_count = int(ev.get("success_count", 0))
            eval_scene_src = _extract_eval_finished_scene_json(
                ev,
                prefer_success=success_count >= int(args.eval_min_success),
            )
            eval_scene_ckpt = _cache_atomic_scene_checkpoint(
                eval_scene_src,
                loop_dir=loop_dir,
                atomic_index=ti,
                round_index=r,
                label="pass" if success_count >= int(args.eval_min_success) else "fail",
            )
            if eval_scene_ckpt:
                task_state["last_eval_scene_json"] = eval_scene_ckpt
                task_state["updated_at"] = _now()
                _write_pipeline_state(pipeline_state_path, pipeline_state)

            if success_count >= int(args.eval_min_success):
                summary["rounds"].append(
                    {
                        "atomic_index": ti,
                        "round": r,
                        "task": task,
                        "status": "ATOMIC_VERIFIED",
                        "eval": ev,
                        "start_scene_json": str(current_scene_json),
                        "end_scene_json": eval_scene_ckpt,
                    }
                )
                task_state["status"] = "verified"
                task_state["updated_at"] = _now()
                _write_pipeline_state(pipeline_state_path, pipeline_state)
                if eval_scene_ckpt:
                    current_scene_json = Path(eval_scene_ckpt)
                verified = True
                break

            known_repo_ids = list(task_state.get("repo_ids", []))
            known_repo_ids = [str(x).strip() for x in known_repo_ids if str(x).strip()]
            reused_existing_dataset = bool(known_repo_ids)
            if reused_existing_dataset:
                repo_id = known_repo_ids[-1]
                gen = {
                    "status": "REUSED_EXISTING_DATASET",
                    "repo_id": repo_id,
                    "task": task,
                    "source": "pipeline_state",
                    "created_at": _now(),
                }
                _write_json(gen_json, gen)
            else:
                _run(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "generate_cli.py"),
                        "--scene-json",
                        str(current_scene_json),
                        "--task",
                        task,
                        "--success-target",
                        str(int(args.generate_success_target)),
                        "--max-attempts",
                        str(int(args.generate_max_attempts)),
                        "--lerobot-root",
                        str(args.generate_lerobot_root),
                        "--overwrite",
                        "--result-json-out",
                        str(gen_json),
                    ]
                )
                gen = _read_json(gen_json)
                repo_id = str(gen.get("repo_id", "")).strip()
                if not repo_id:
                    raise RuntimeError(f"generate_cli missing repo_id: {gen}")
                known_repo_ids = [x for x in known_repo_ids if x != repo_id] + [repo_id]
                task_state["repo_ids"] = known_repo_ids
                task_state["updated_at"] = _now()
                _write_pipeline_state(pipeline_state_path, pipeline_state)

            primitive = _primitive_from_repo_id(repo_id)
            if primitive and not str(task_state.get("primitive", "")).strip():
                task_state["primitive"] = primitive
                task_state["updated_at"] = _now()
                _write_pipeline_state(pipeline_state_path, pipeline_state)

            experts = expert_state.setdefault("experts", {})
            primitive_state = experts.get(primitive, {})
            prev_policy_path = str(primitive_state.get("policy_path", "")).strip()
            if prev_policy_path and not Path(prev_policy_path).exists():
                prev_policy_path = ""
            if not prev_policy_path:
                discovered_prev = _discover_latest_expert_policy(checkpoint_root, primitive)
                if discovered_prev is not None:
                    prev_policy_path = str(discovered_prev)
            history_repo_ids = list(primitive_state.get("history_repo_ids", []))
            history_repo_ids = [str(x).strip() for x in history_repo_ids if str(x).strip()]
            has_existing_expert = bool(prev_policy_path)
            same_task_alt_repo_ids = [x for x in known_repo_ids if x != repo_id]

            replay_repo_id = ""
            replay_steps = 0
            current_steps = int(args.train_steps)
            replay_candidates = [x for x in same_task_alt_repo_ids + history_repo_ids if x and x != repo_id]
            replay_candidates = list(dict.fromkeys(replay_candidates))
            if has_existing_expert and replay_candidates and replay_rate > 0:
                replay_steps = max(1, int(round(int(args.train_steps) * replay_rate)))
                current_steps = max(1, int(args.train_steps) - replay_steps)
                replay_repo_id = random.choice(replay_candidates)

            last_policy_path = prev_policy_path
            if replay_repo_id and replay_steps > 0:
                replay_train_json = rd / "train_replay_result.json"
                replay_base_dir = checkpoint_root / "experts" / primitive / f"atomic_{ti:03d}_round_{r:03d}_replay"
                replay_out_dir = _alloc_unique_dir(replay_base_dir)
                _log(
                    f"[atomic {ti} round {r}] Train(replay) start | "
                    f"load(base={resolved_base_model_path}, adapter_init={last_policy_path or 'none'}, mode=lora_moe, expert={primitive}, dataset={replay_repo_id})"
                )
                _run(
                    _build_train_cmd(
                        dataset_repo_id=replay_repo_id,
                        dataset_root=str(args.generate_lerobot_root),
                        output_dir=replay_out_dir,
                        checkpoint_root=checkpoint_root,
                        policy_path=last_policy_path,
                        steps=int(replay_steps),
                        batch_size=int(args.train_batch_size),
                        device=str(args.train_device),
                        expert_id=primitive,
                        result_json_out=replay_train_json,
                        policy_type=str(args.policy_type),
                        base_model_path=str(resolved_base_model_path),
                    )
                )
                last_policy_path = str(_latest_pretrained_model_dir(replay_out_dir))

            current_base_dir = checkpoint_root / "experts" / primitive / f"atomic_{ti:03d}_round_{r:03d}_current"
            current_out_dir = _alloc_unique_dir(current_base_dir)
            _log(
                f"[atomic {ti} round {r}] Train(current) start | "
                f"load(base={resolved_base_model_path}, adapter_init={last_policy_path or 'none'}, mode=lora_moe, expert={primitive}, dataset={repo_id})"
            )
            _run(
                _build_train_cmd(
                    dataset_repo_id=repo_id,
                    dataset_root=str(args.generate_lerobot_root),
                    output_dir=current_out_dir,
                    checkpoint_root=checkpoint_root,
                    policy_path=last_policy_path,
                    steps=int(current_steps),
                    batch_size=int(args.train_batch_size),
                    device=str(args.train_device),
                    expert_id=primitive,
                    result_json_out=train_json,
                    policy_type=str(args.policy_type),
                    base_model_path=str(resolved_base_model_path),
                )
            )
            train_result = _read_json(train_json)
            policy_path = str(_latest_pretrained_model_dir(current_out_dir))
            last_policy_path_for_summary = policy_path

            updated_history = [x for x in history_repo_ids if x != repo_id] + [repo_id]
            experts[primitive] = {
                "policy_path": policy_path,
                "history_repo_ids": updated_history,
                "updated_at": _now(),
            }
            _write_expert_state(expert_state_path, expert_state)

            # Router supervision is trained jointly in lerobot_train_cli (PI05 forward loss),
            # using --policy.router_expert_index derived from the current primitive.
            summary["router_state_path"] = ""
            _write_json(loop_dir / "summary.json", summary)

            _log(
                f"[atomic {ti} round {r}] Validate(expert) start | "
                f"load(base={resolved_base_model_path}, adapters=latest_experts_under_{checkpoint_root / 'experts'}, router=router, router_state={router_state_path}, force_full_moe=true, dataset={repo_id})"
            )
            _run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "lerobot_validate_cli.py"),
                    "--dataset-repo-id",
                    repo_id,
                    "--dataset-root",
                    str(args.generate_lerobot_root),
                    "--checkpoint-root",
                    str(checkpoint_root),
                    "--base-model-path",
                    str(resolved_base_model_path),
                    "--policy-type",
                    str(args.policy_type),
                    "--force-full-moe",
                    "--batch-size",
                    str(int(args.train_batch_size)),
                    "--max-batches",
                    str(int(args.val_max_batches)),
                    "--result-json-out",
                    str(val_json),
                ]
            )
            summary["rounds"].append(
                {
                    "atomic_index": ti,
                    "round": r,
                    "task": task,
                    "status": "CONTINUE",
                    "repo_id": repo_id,
                    "reused_existing_dataset": reused_existing_dataset,
                    "primitive": primitive,
                    "has_existing_expert": has_existing_expert,
                    "replay_repo_id": replay_repo_id,
                    "replay_rate": replay_rate,
                    "replay_steps": replay_steps,
                    "current_steps": current_steps,
                    "policy_path": policy_path,
                    "start_scene_json": str(current_scene_json),
                    "end_scene_json": eval_scene_ckpt,
                    "train": train_result,
                    "validate": _read_json(val_json),
                }
            )
            _write_json(loop_dir / "summary.json", summary)

        if not verified:
            summary["status"] = "FAILED"
            summary["failed_atomic_index"] = ti
            summary["failed_task_start_scene_json"] = str(current_scene_json)
            summary["failed_task_end_scene_json"] = str(task_state.get("last_eval_scene_json", ""))
            task_state["status"] = "failed"
            task_state["updated_at"] = _now()
            _write_pipeline_state(pipeline_state_path, pipeline_state)
            _write_json(loop_dir / "summary.json", summary)
            raise RuntimeError(f"Atomic task failed after max rounds: {task}")

    summary["status"] = "SUCCESS"
    if last_policy_path_for_summary:
        summary["final_policy_path"] = last_policy_path_for_summary
    summary["finished_at"] = _now()
    _write_json(loop_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
