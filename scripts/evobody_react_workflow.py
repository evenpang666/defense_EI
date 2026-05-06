"""Chain evobody supervisor/coder/judger via DefenseAgent ReActAgent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure local vendored DefenseAgent package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFENSE_AGENT_ROOT = PROJECT_ROOT / "third_party" / "DefenseAgent"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(DEFENSE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFENSE_AGENT_ROOT))

from DefenseAgent import AgentConfig, ReActAgent  # noqa: E402
from DefenseAgent.config import AgentProfile  # noqa: E402
from DefenseAgent.memory import MemoryBackendConfig  # noqa: E402

from evobody_profiles.coder.python_tools.scene_runtime_tools import (  # noqa: E402
    execute_generated_code_in_loaded_scene,
    load_mujoco_scene_from_json,
)
from evobody_profiles.supervisor.python_tools.scene_context_tools import (  # noqa: E402
    load_scene_context_from_snapshot,
)
from evobody_profiles.supervisor.python_tools.scene_loader_tool import (  # noqa: E402
    scene_loader_script,
)


DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "scripts" / "evobody_profiles"
DEFAULT_SUPERVISOR_PROFILE = DEFAULT_PROFILE_ROOT / "supervisor" / "profile.yaml"
DEFAULT_CODER_PROFILE = DEFAULT_PROFILE_ROOT / "coder" / "profile.yaml"
DEFAULT_JUDGER_PROFILE = DEFAULT_PROFILE_ROOT / "judger" / "profile.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_SCENE_SNAPSHOT = PROJECT_ROOT / "chemistry.json"
DEFAULT_FRONT_IMAGE = DEFAULT_LOG_DIR / "current_view.png"
DEFAULT_FINAL_FRONT_IMAGE = DEFAULT_LOG_DIR / "finished_view.png"
DEFAULT_FINAL_WRIST_IMAGE = DEFAULT_LOG_DIR / "finished_view_wrist.png"


@dataclass
class PipelineOutputs:
    plan: dict[str, Any]
    code: str
    judge: dict[str, Any]
    raw_supervisor_text: str
    raw_coder_text: str
    raw_judger_text: str


def _resolve_memory_backend_from_profile(
    profile_path: Path,
    *,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    embedding_dims: int,
) -> MemoryBackendConfig:
    profile = AgentProfile.from_yaml(profile_path)
    llm_provider = (profile.llm.provider or "").strip().lower()
    llm_model = (profile.llm.model or "").strip()
    llm_api_key = (profile.llm.api_key or "").strip()
    llm_base_url = (profile.llm.base_url or "").strip()

    if not llm_provider or not llm_model or not llm_api_key:
        raise ValueError(
            f"coder profile llm config incomplete in {profile_path}; "
            "need provider/model/api_key for mem0 backend."
        )

    resolved_embedding_api_key = (embedding_api_key or "").strip() or llm_api_key
    resolved_embedding_base_url = (embedding_base_url or "").strip() or llm_base_url
    resolved_embedding_model = embedding_model.strip()
    if not resolved_embedding_model:
        raise ValueError("embedding model is required for coder memory backend")

    return MemoryBackendConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        embedding_provider="openai",
        embedding_api_key=resolved_embedding_api_key,
        embedding_model=resolved_embedding_model,
        embedding_base_url=resolved_embedding_base_url,
        embedding_dims=embedding_dims,
    )


def _build_react_agent(
    profile_path: Path,
    *,
    enable_memory: bool,
    memory_backend: MemoryBackendConfig | None = None,
) -> ReActAgent:
    return ReActAgent(
        AgentConfig(
            profile=profile_path,
            use_memory=enable_memory,
            use_reflection=enable_memory,
            use_compressor=enable_memory,
            memory_backend=memory_backend,
        )
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty model response")

    fenced = re.search(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("json payload is not an object")
        return payload

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate = text[left : right + 1]
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("json payload is not an object")
        return payload

    raise ValueError("no json object found in model response")


def _extract_python_code(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty model response")

    fenced = re.search(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    generic = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if generic:
        return generic.group(1).strip()

    return text


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _bootstrap_current_view_if_missing(scene_snapshot: Path, front_image: Path) -> None:
    if front_image.exists():
        return
    raw = scene_loader_script(
        scene_json_path=str(scene_snapshot),
        current_view_path=str(front_image),
        randomize_texture=False,
    )
    payload = _extract_json_object(raw)
    if not bool(payload.get("ok")) or not front_image.exists():
        raise FileNotFoundError(
            f"failed to bootstrap current view via scene_loader_script; payload={payload}"
        )


def _ensure_final_image_exists(
    *,
    code: str,
    scene_snapshot: Path,
    final_front_image: Path,
    log_dir: Path,
) -> None:
    """Fallback executor: if the coder agent did not actually run the generated
    code (no `finished_view.png` on disk), execute it here so the judger always
    has a final-state image to inspect.
    """
    if final_front_image.exists():
        return

    load_raw = load_mujoco_scene_from_json(str(scene_snapshot), randomize_texture=False)
    load_payload = _extract_json_object(load_raw)
    if not bool(load_payload.get("ok")):
        raise RuntimeError(
            f"fallback scene load failed before code execution; payload={load_payload}"
        )

    error_log_path = log_dir / "last_execution_error.txt"
    exec_raw = execute_generated_code_in_loaded_scene(
        code=code,
        final_image_path=str(final_front_image),
        error_output_path=str(error_log_path),
    )
    exec_payload = _extract_json_object(exec_raw)
    if not final_front_image.exists():
        raise FileNotFoundError(
            "fallback execution did not produce final image: "
            f"{final_front_image}; exec_payload={exec_payload}"
        )


async def run_evobody_react_workflow(
    *,
    task: str,
    scene_snapshot: Path,
    front_image: Path,
    final_front_image: Path,
    final_wrist_image: Path | None,
    supervisor_profile: Path,
    coder_profile: Path,
    judger_profile: Path,
    log_dir: Path,
    max_steps: int | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
    embedding_dims: int = 1536,
) -> PipelineOutputs:
    _bootstrap_current_view_if_missing(scene_snapshot, front_image)
    scene_context = load_scene_context_from_snapshot(scene_snapshot, front_image)
    _save_json(log_dir / "evobody_scene_context.json", scene_context)

    supervisor = _build_react_agent(supervisor_profile, enable_memory=False)
    coder = _build_react_agent(coder_profile, enable_memory=False)
    judger = _build_react_agent(judger_profile, enable_memory=False)

    supervisor_task = (
        "Task:\n"
        f"{task}\n\n"
        "Reconstructed scene context JSON:\n"
        f"{json.dumps(scene_context, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only."
    )
    try:
        supervisor_result = await supervisor.run(
            supervisor_task,
            max_steps=max_steps,
            images=[front_image],
        )
    except FileNotFoundError:
        # Retry path requested by user: if current_view is missing, bootstrap via tool, then rerun supervisor.
        _bootstrap_current_view_if_missing(scene_snapshot, front_image)
        supervisor_result = await supervisor.run(
            supervisor_task,
            max_steps=max_steps,
            images=[front_image],
        )
    raw_supervisor_text = supervisor_result.final_answer
    plan = _extract_json_object(raw_supervisor_text)
    _save_json(log_dir / "evobody_plan.json", plan)
    _save_text(log_dir / "evobody_plan_raw.txt", raw_supervisor_text)

    coder_task = (
        "Task:\n"
        f"{task}\n\n"
        "Atomic task JSON generated by supervisor:\n"
        f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        "Reconstructed scene context JSON:\n"
        f"{json.dumps(scene_context, ensure_ascii=False, indent=2)}\n\n"
        "Return runnable python code only."
    )
    coder_result = await coder.run(
        coder_task,
        max_steps=max_steps,
        images=[front_image],
    )
    raw_coder_text = coder_result.final_answer
    code = _extract_python_code(raw_coder_text)
    _save_text(log_dir / "atomic_actions.py", code)
    _save_text(log_dir / "atomic_actions_raw.txt", raw_coder_text)

    _ensure_final_image_exists(
        code=code,
        scene_snapshot=scene_snapshot,
        final_front_image=final_front_image,
        log_dir=log_dir,
    )

    judge_prompt = (
        "Task description:\n"
        f"{task}\n\n"
        "Generated atomic action code:\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        "Return JSON only with keys task_result and analysis."
    )
    judge_images: list[Path] = []
    if final_front_image.exists():
        judge_images.append(final_front_image)
    else:
        raise FileNotFoundError(
            f"final front image missing before judging: {final_front_image}"
        )
    if final_wrist_image and final_wrist_image.exists():
        judge_images.append(final_wrist_image)
    judger_result = await judger.run(
        judge_prompt,
        max_steps=max_steps,
        images=judge_images,
    )
    raw_judger_text = judger_result.final_answer
    judge = _extract_json_object(raw_judger_text)
    _save_json(log_dir / "finished_view_judge.json", judge)
    _save_text(log_dir / "finished_view_judge_raw.txt", raw_judger_text)

    return PipelineOutputs(
        plan=plan,
        code=code,
        judge=judge,
        raw_supervisor_text=raw_supervisor_text,
        raw_coder_text=raw_coder_text,
        raw_judger_text=raw_judger_text,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evobody 3-agent ReAct workflow.")
    parser.add_argument("--task", required=True, help="High-level user task text.")
    parser.add_argument("--scene-snapshot", type=Path, default=DEFAULT_SCENE_SNAPSHOT)
    parser.add_argument("--front-image", type=Path, default=DEFAULT_FRONT_IMAGE)
    parser.add_argument("--final-front-image", type=Path, default=DEFAULT_FINAL_FRONT_IMAGE)
    parser.add_argument("--final-wrist-image", type=Path, default=DEFAULT_FINAL_WRIST_IMAGE)
    parser.add_argument("--supervisor-profile", type=Path, default=DEFAULT_SUPERVISOR_PROFILE)
    parser.add_argument("--coder-profile", type=Path, default=DEFAULT_CODER_PROFILE)
    parser.add_argument("--judger-profile", type=Path, default=DEFAULT_JUDGER_PROFILE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model for coder memory (mem0).",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.environ.get("EMBEDDING_API_KEY"),
        help="Embedding API key for coder memory; defaults to EMBEDDING_API_KEY, then coder llm api key.",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=os.environ.get("EMBEDDING_BASE_URL"),
        help="Embedding base URL for coder memory; defaults to EMBEDDING_BASE_URL, then coder llm base_url.",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        default=int(os.environ.get("EMBEDDING_DIMS", "1536")),
        help="Embedding dims for coder memory vector store.",
    )
    parser.add_argument(
        "--no-wrist-image",
        action="store_true",
        help="Do not pass wrist image to judger.",
    )
    return parser.parse_args()


async def _main_async() -> int:
    args = _parse_args()
    wrist_image = None if args.no_wrist_image else args.final_wrist_image

    outputs = await run_evobody_react_workflow(
        task=args.task,
        scene_snapshot=args.scene_snapshot,
        front_image=args.front_image,
        final_front_image=args.final_front_image,
        final_wrist_image=wrist_image,
        supervisor_profile=args.supervisor_profile,
        coder_profile=args.coder_profile,
        judger_profile=args.judger_profile,
        log_dir=args.log_dir,
        max_steps=args.max_steps,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_dims=args.embedding_dims,
    )

    print("[workflow] supervisor status:", outputs.plan.get("status", "unknown"))
    print("[workflow] generated code lines:", len(outputs.code.splitlines()))
    print("[workflow] judge task_result:", outputs.judge.get("task_result", "unknown"))
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
