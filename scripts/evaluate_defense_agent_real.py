"""Run one-pass DefenseAgent Evobody validation on a real UR7e setup."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio


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

from evaluate_vla_real import RealSenseRGBPair  # noqa: E402
from ur7e_controller import ROBOT_IP, UR7eVectorController, make_real_runtime_api  # noqa: E402


DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "scripts" / "evobody_profiles"
DEFAULT_SUPERVISOR_PROFILE = DEFAULT_PROFILE_ROOT / "supervisor" / "profile.yaml"
DEFAULT_CODER_PROFILE = DEFAULT_PROFILE_ROOT / "coder" / "profile.yaml"
DEFAULT_JUDGER_PROFILE = DEFAULT_PROFILE_ROOT / "judger" / "profile.yaml"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "defense_agent_real"


@dataclass
class RealDefenseOutputs:
    plan: dict[str, Any]
    code: str
    execution: dict[str, Any]
    judge: dict[str, Any]
    raw_supervisor_text: str
    raw_coder_text: str
    raw_judger_text: str


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] [evaluate_defense_agent_real] {msg}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty model response")

    fenced = re.search(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        payload = json.loads(fenced.group(1).strip())
        if isinstance(payload, dict):
            return payload
        raise ValueError("json payload is not an object")

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        payload = json.loads(text[left : right + 1])
        if isinstance(payload, dict):
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

    return MemoryBackendConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        embedding_provider="openai",
        embedding_api_key=(embedding_api_key or "").strip() or llm_api_key,
        embedding_model=embedding_model.strip(),
        embedding_base_url=(embedding_base_url or "").strip() or llm_base_url,
        embedding_dims=int(embedding_dims),
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


def _check_generated_code(code: str) -> None:
    forbidden = ["def ", "class ", "import ", "from ", "exec(", "eval(", "subprocess", "os.", "sys."]
    hits = [tok for tok in forbidden if tok in code]
    if hits:
        raise ValueError(f"generated code contains forbidden tokens: {hits}")
    compile(code, "<real_robot_generated_code>", "exec")


def _execute_real_code(
    *,
    code: str,
    controller: UR7eVectorController | None,
    dry_run: bool,
) -> dict[str, Any]:
    _check_generated_code(code)
    if dry_run:
        return {"ok": True, "dry_run": True, "signal": "REAL_EXECUTION_SKIPPED"}
    if controller is None:
        raise RuntimeError("controller is required when dry_run is false")

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
        "__name__": "__real_atomic__",
    }
    exec_globals: dict[str, Any] = {"__builtins__": safe_builtins}
    exec_globals.update(make_real_runtime_api(controller))

    started = time.time()
    try:
        exec(compile(code, "<real_robot_generated_code>", "exec"), exec_globals)
        return {
            "ok": True,
            "dry_run": False,
            "signal": "REAL_EXECUTION_OK",
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception:
        return {
            "ok": False,
            "dry_run": False,
            "signal": "REAL_EXECUTION_FAILED",
            "elapsed_s": round(time.time() - started, 3),
            "error": traceback.format_exc(),
        }


async def run_defense_agent_real_once(
    *,
    task: str,
    current_front_image: Path,
    current_wrist_image: Path,
    final_front_image: Path,
    final_wrist_image: Path,
    supervisor_profile: Path,
    coder_profile: Path,
    judger_profile: Path,
    log_dir: Path,
    controller: UR7eVectorController | None,
    dry_run: bool,
    max_steps: int | None,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    embedding_dims: int,
    capture_final_images_fn: Callable[[], None] | None = None,
) -> RealDefenseOutputs:
    del embedding_model, embedding_api_key, embedding_base_url, embedding_dims

    supervisor = _build_react_agent(supervisor_profile, enable_memory=False)
    coder = _build_react_agent(coder_profile, enable_memory=False)
    judger = _build_react_agent(judger_profile, enable_memory=False)

    observation = {
        "environment": "real_robot",
        "robot": "UR7e",
        "cameras": {
            "front_rgb": str(current_front_image),
            "wrist_rgb": str(current_wrist_image),
        },
        "runtime_note": (
            "This is a real-robot one-pass validation. The only visual state inputs "
            "are the two Intel RealSense D435i RGB images. Do not assume a MuJoCo "
            "scene or reconstructed object pose JSON exists."
        ),
    }

    supervisor_task = (
        "REAL_ROBOT_VALIDATION\n"
        "Task:\n"
        f"{task}\n\n"
        "Real observation JSON:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}\n\n"
        "Produce exactly one atomic task for this one-pass real-robot run. Return JSON only."
    )
    supervisor_result = await supervisor.run(
        supervisor_task,
        max_steps=max_steps,
        images=[current_front_image, current_wrist_image],
    )
    raw_supervisor_text = supervisor_result.final_answer
    plan = _extract_json_object(raw_supervisor_text)
    _save_json(log_dir / "real_evobody_plan.json", plan)
    _save_text(log_dir / "real_evobody_plan_raw.txt", raw_supervisor_text)

    coder_task = (
        "REAL_ROBOT_VALIDATION\n"
        "Use the real-robot-code-contract skill. Do not call MuJoCo tools.\n\n"
        "Task:\n"
        f"{task}\n\n"
        "Atomic task JSON generated by supervisor:\n"
        f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        "Real observation JSON:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}\n\n"
        "Return runnable Python code only."
    )
    coder_result = await coder.run(
        coder_task,
        max_steps=max_steps,
        images=[current_front_image, current_wrist_image],
    )
    raw_coder_text = coder_result.final_answer
    code = _extract_python_code(raw_coder_text)
    _save_text(log_dir / "real_atomic_actions.py", code)
    _save_text(log_dir / "real_atomic_actions_raw.txt", raw_coder_text)

    execution = _execute_real_code(code=code, controller=controller, dry_run=dry_run)
    _save_json(log_dir / "real_execution.json", execution)
    if not bool(execution.get("ok")):
        _save_text(log_dir / "real_execution_error.txt", str(execution.get("error", "")))
    if capture_final_images_fn is not None:
        capture_final_images_fn()
    if not final_front_image.exists() or not final_wrist_image.exists():
        raise FileNotFoundError("final RealSense images are required before judging")

    judge_prompt = (
        "REAL_ROBOT_VALIDATION\n"
        "Task description:\n"
        f"{task}\n\n"
        "Generated and executed atomic action code:\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        "Real execution report:\n"
        f"{json.dumps(execution, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with keys task_result and analysis."
    )
    judger_result = await judger.run(
        judge_prompt,
        max_steps=max_steps,
        images=[final_front_image, final_wrist_image],
    )
    raw_judger_text = judger_result.final_answer
    judge = _extract_json_object(raw_judger_text)
    _save_json(log_dir / "real_finished_view_judge.json", judge)
    _save_text(log_dir / "real_finished_view_judge_raw.txt", raw_judger_text)

    return RealDefenseOutputs(
        plan=plan,
        code=code,
        execution=execution,
        judge=judge,
        raw_supervisor_text=raw_supervisor_text,
        raw_coder_text=raw_coder_text,
        raw_judger_text=raw_judger_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-pass DefenseAgent real-robot evaluation.")
    parser.add_argument("--task", required=True, help="High-level task description.")
    parser.add_argument("--supervisor-profile", type=Path, default=DEFAULT_SUPERVISOR_PROFILE)
    parser.add_argument("--coder-profile", type=Path, default=DEFAULT_CODER_PROFILE)
    parser.add_argument("--judger-profile", type=Path, default=DEFAULT_JUDGER_PROFILE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--robotiq-urscript-defs-path", default="")
    parser.add_argument("--strict-gripper-connection", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate and judge flow without connecting to UR7e.")

    parser.add_argument("--camera-serials", default="", help="Comma-separated RealSense serials: front,wrist. Auto-discovers if omitted.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)

    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_BASE_URL"))
    parser.add_argument("--embedding-dims", type=int, default=int(os.environ.get("EMBEDDING_DIMS", "1536")))
    return parser.parse_args()


async def _main_async() -> int:
    args = parse_args()
    run_dir = _resolve_path(args.log_root) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    current_front = run_dir / "current_front_rgb.png"
    current_wrist = run_dir / "current_wrist_rgb.png"
    final_front = run_dir / "finished_front_rgb.png"
    final_wrist = run_dir / "finished_wrist_rgb.png"
    serials = [s.strip() for s in str(args.camera_serials).split(",") if s.strip()]

    controller: UR7eVectorController | None = None
    if not args.dry_run:
        controller = UR7eVectorController(
            robot_ip=str(args.robot_ip),
            robotiq_urscript_defs_path=str(args.robotiq_urscript_defs_path).strip() or None,
            strict_gripper_connection=bool(args.strict_gripper_connection),
        )

    _log("Starting RealSense capture")
    with RealSenseRGBPair(
        serials=serials,
        width=int(args.camera_width),
        height=int(args.camera_height),
        fps=int(args.camera_fps),
        warmup_frames=int(args.camera_warmup_frames),
    ) as cameras:
        front_img, wrist_img = cameras.capture()
        imageio.imwrite(current_front, front_img)
        imageio.imwrite(current_wrist, wrist_img)

        try:
            if controller is not None:
                _log(f"Connecting UR7e controller at {args.robot_ip}")
                controller.connect()
                if controller.is_gripper_available():
                    _log(f"Gripper backend: {controller.get_gripper_backend()}")

            def capture_final_images() -> None:
                front_done, wrist_done = cameras.capture()
                imageio.imwrite(final_front, front_done)
                imageio.imwrite(final_wrist, wrist_done)

            outputs = await run_defense_agent_real_once(
                task=str(args.task).strip(),
                current_front_image=current_front,
                current_wrist_image=current_wrist,
                final_front_image=final_front,
                final_wrist_image=final_wrist,
                supervisor_profile=_resolve_path(args.supervisor_profile),
                coder_profile=_resolve_path(args.coder_profile),
                judger_profile=_resolve_path(args.judger_profile),
                log_dir=run_dir,
                controller=controller,
                dry_run=bool(args.dry_run),
                max_steps=args.max_steps,
                embedding_model=str(args.embedding_model),
                embedding_api_key=args.embedding_api_key,
                embedding_base_url=args.embedding_base_url,
                embedding_dims=int(args.embedding_dims),
                capture_final_images_fn=capture_final_images,
            )

        finally:
            if controller is not None:
                controller.close()

    summary = {
        "run_dir": str(run_dir),
        "plan_status": outputs.plan.get("status"),
        "execution_ok": bool(outputs.execution.get("ok")),
        "judge_result": outputs.judge.get("task_result"),
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if bool(outputs.execution.get("ok")) else 2


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
