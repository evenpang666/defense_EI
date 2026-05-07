"""Run the defense_ei_agents real-world DefenseAgent workflow on a UR7e setup."""

from __future__ import annotations

import argparse
import ast
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
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFENSE_AGENT_ROOT = PROJECT_ROOT / "third_party" / "DefenseAgent"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(DEFENSE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFENSE_AGENT_ROOT))

from evaluate_vla_real import RealSenseRGBPair  # noqa: E402
from ur7e_controller import ROBOT_IP, UR7eVectorController, make_real_runtime_api  # noqa: E402


DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "scripts" / "defense_ei_agents"
DEFAULT_PLANNER_PROFILE = DEFAULT_PROFILE_ROOT / "planner" / "profile.yaml"
DEFAULT_SUPERVISOR_PROFILE = DEFAULT_PROFILE_ROOT / "supervisor" / "profile.yaml"
DEFAULT_CODER_PROFILE = DEFAULT_PROFILE_ROOT / "coder" / "profile.yaml"
DEFAULT_JUDGER_PROFILE = DEFAULT_PROFILE_ROOT / "judger" / "profile.yaml"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "defense_agent_real"
ALLOWED_RUNTIME_CALLS = {
    "ee_pose",
    "move_ee",
    "gripper_control",
    "move_x",
    "move_y",
    "move_z",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "sleep",
    "print",
}
ALLOWED_BUILTIN_CALLS = {
    "range",
    "len",
    "min",
    "max",
    "abs",
    "float",
    "int",
    "str",
    "list",
    "dict",
    "tuple",
    "bool",
    "enumerate",
    "zip",
    "round",
    "sum",
    "isinstance",
}
FORBIDDEN_RUNTIME_CALLS = {
    "pick_and_place",
    "pick_place",
    "push",
    "pull",
    "press",
    "open",
    "close",
    "pour",
    "move_to",
}


@dataclass
class RealDefenseOutputs:
    plan: dict[str, Any]
    atomic_task_info: dict[str, Any]
    attempts: list[dict[str, Any]]
    completed: bool
    code: str
    execution: dict[str, Any]
    judge: dict[str, Any]
    raw_planner_text: str
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


def _prepare_real_profile_copy(profile_path: Path, log_dir: Path, role: str) -> Path:
    """Write a temporary profile with paths resolved inside the run directory."""
    profile_path = _resolve_path(profile_path)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("agent"), dict):
        raise ValueError(f"Invalid DefenseAgent profile: {profile_path}")

    agent = raw["agent"]
    tools = agent.get("tools")
    if not isinstance(tools, dict):
        return profile_path

    source_dir = profile_path.parent

    def abs_ref(ref: str) -> str:
        p = Path(str(ref))
        return str(p if p.is_absolute() else (source_dir / p).resolve())

    tools["skills"] = [abs_ref(ref) for ref in tools.get("skills", [])]
    tools["python"] = [
        abs_ref(ref.rsplit(":", 1)[0]) + ":" + ref.rsplit(":", 1)[1]
        for ref in tools.get("python", [])
        if ":" in ref
    ]
    tools["mcp"] = []
    tools["allow_skill_execution"] = False

    prompt = agent.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("path"), str):
        prompt_path = Path(prompt["path"])
        if not prompt_path.is_absolute():
            prompt["path"] = str((source_dir / prompt_path).resolve())

    out_path = log_dir / f"real_{role}_profile.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path


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
    from DefenseAgent.config import AgentProfile
    from DefenseAgent.memory import MemoryBackendConfig

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
    from DefenseAgent import AgentConfig, ReActAgent

    return ReActAgent(
        AgentConfig(
            profile=profile_path,
            use_memory=enable_memory,
            use_reflection=enable_memory,
            use_compressor=enable_memory,
            memory_backend=memory_backend,
        )
    )


def _validate_generated_code_runtime_calls(tree: ast.AST) -> dict[str, Any]:
    called_names: list[str] = []
    forbidden_hits: list[str] = []
    unknown_calls: list[str] = []
    allowed = ALLOWED_RUNTIME_CALLS | ALLOWED_BUILTIN_CALLS

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
            called_names.append(name)
            if name in FORBIDDEN_RUNTIME_CALLS:
                forbidden_hits.append(name)
            elif name not in allowed:
                unknown_calls.append(name)
        elif isinstance(node.func, ast.Attribute):
            unknown_calls.append(f"attribute_call:{node.func.attr}")

    return {
        "ok": not forbidden_hits and not unknown_calls,
        "called_names": sorted(set(called_names)),
        "forbidden_runtime_calls": sorted(set(forbidden_hits)),
        "unknown_calls": sorted(set(unknown_calls)),
    }


def _check_generated_code(code: str) -> dict[str, Any]:
    lowered = (code or "").lower()
    forbidden_text = [
        tok
        for tok in (
            "quat",
            "quaternion",
            "wxyz",
            "xyzw",
            "move_to",
            "pick_and_place",
            "pick_place",
            "push(",
            "pull(",
            "press(",
            "open(",
            "close(",
            "pour(",
        )
        if tok in lowered
    ]
    if forbidden_text:
        raise ValueError(f"generated code contains forbidden tokens: {forbidden_text}")

    tree = ast.parse(code, mode="exec")
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            hits.append("def")
        elif isinstance(node, ast.ClassDef):
            hits.append("class")
        elif isinstance(node, ast.Import):
            hits.append("import")
        elif isinstance(node, ast.ImportFrom):
            hits.append("from")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            hits.append(node.func.id)
        elif isinstance(node, ast.Name) and node.id in {"subprocess", "os", "sys"}:
            hits.append(node.id)
    if hits:
        raise ValueError(f"generated code contains forbidden syntax: {sorted(set(hits))}")
    runtime_report = _validate_generated_code_runtime_calls(tree)
    if not runtime_report["ok"]:
        raise ValueError(f"generated code calls unsupported runtime tools: {runtime_report}")
    compile(tree, "<real_robot_generated_code>", "exec")
    return runtime_report


def _execute_real_code(
    *,
    code: str,
    controller: UR7eVectorController | None,
) -> dict[str, Any]:
    _check_generated_code(code)
    if controller is None:
        raise RuntimeError("controller is required for real execution")

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
            "signal": "REAL_EXECUTION_OK",
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception:
        return {
            "ok": False,
            "signal": "REAL_EXECUTION_FAILED",
            "elapsed_s": round(time.time() - started, 3),
            "error": traceback.format_exc(),
        }


def _snapshot_robot_initial_state(
    controller: UR7eVectorController,
) -> dict[str, Any]:
    return {
        "joints": controller.get_current_joints(),
        "gripper_open_ratio": controller.get_gripper_open_ratio(),
    }


def _restore_robot_initial_state(
    controller: UR7eVectorController | None,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    if controller is None:
        return {"ok": False, "signal": "REAL_RESTORE_NO_CONTROLLER"}
    try:
        joints = snapshot.get("joints")
        if isinstance(joints, list) and len(joints) == 6:
            controller.move_joints(joints, acceleration=0.8, velocity=0.25)
        gripper_ratio = snapshot.get("gripper_open_ratio")
        if isinstance(gripper_ratio, (int, float)) and controller.is_gripper_available():
            controller.set_gripper(float(gripper_ratio))
        return {"ok": True, "signal": "REAL_RESTORE_OK"}
    except Exception:
        return {
            "ok": False,
            "signal": "REAL_RESTORE_FAILED",
            "error": traceback.format_exc(),
        }


def _capture_or_copy_images(
    *,
    capture_images_fn: Callable[[Path, Path], None] | None,
    target_front: Path,
    target_wrist: Path,
) -> None:
    target_front.parent.mkdir(parents=True, exist_ok=True)
    if capture_images_fn is not None:
        capture_images_fn(target_front, target_wrist)
        return
    raise RuntimeError("camera capture callback is required for real execution")


def _atomic_infos_from_supervisor(payload: dict[str, Any]) -> list[dict[str, Any]]:
    infos = payload.get("atomic_task_info", payload.get("atomic_tasks", []))
    if not isinstance(infos, list):
        raise ValueError("supervisor output must include atomic_task_info list")
    return [info for info in infos if isinstance(info, dict)]


def _is_success_judge(judge: dict[str, Any]) -> bool:
    return str(judge.get("task_result", "")).strip().upper() == "SUCCESS"


async def run_defense_agent_real_once(
    *,
    task: str,
    current_front_image: Path,
    current_wrist_image: Path,
    planner_profile: Path,
    supervisor_profile: Path,
    coder_profile: Path,
    judger_profile: Path,
    log_dir: Path,
    controller: UR7eVectorController,
    max_steps: int | None,
    max_attempts_per_atomic: int,
    embedding_model: str,
    embedding_api_key: str | None,
    embedding_base_url: str | None,
    embedding_dims: int,
    capture_images_fn: Callable[[Path, Path], None] | None = None,
) -> RealDefenseOutputs:
    del embedding_model, embedding_api_key, embedding_base_url, embedding_dims

    planner = _build_react_agent(planner_profile, enable_memory=False)
    supervisor = _build_react_agent(supervisor_profile, enable_memory=False)
    coder = _build_react_agent(coder_profile, enable_memory=False)
    judger = _build_react_agent(judger_profile, enable_memory=False)

    initial_robot_state = _snapshot_robot_initial_state(controller)
    _save_json(log_dir / "initial_robot_state.json", initial_robot_state)

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

    planner_task = (
        "DEFENSE_EI_AGENTS_REAL_PLANNING\n"
        "Task:\n"
        f"{task}\n\n"
        "Real observation JSON:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}\n\n"
        "Decompose the task into primitive_skill-level atomic tasks. Return JSON only."
    )
    planner_result = await planner.run(
        planner_task,
        max_steps=max_steps,
        images=[current_front_image, current_wrist_image],
    )
    raw_planner_text = planner_result.final_answer
    plan = _extract_json_object(raw_planner_text)
    _save_json(log_dir / "defense_ei_plan.json", plan)
    _save_text(log_dir / "defense_ei_plan_raw.txt", raw_planner_text)

    if str(plan.get("status", "")).strip().lower() == "completed":
        return RealDefenseOutputs(
            plan=plan,
            atomic_task_info={"status": "completed", "summary": plan.get("summary", ""), "atomic_task_info": []},
            attempts=[],
            completed=True,
            code="",
            execution={"ok": True, "signal": "PLAN_ALREADY_COMPLETED"},
            judge={"task_result": "SUCCESS", "analysis": ""},
            raw_planner_text=raw_planner_text,
            raw_supervisor_text="",
            raw_coder_text="",
            raw_judger_text="",
        )

    supervisor_task = (
        "DEFENSE_EI_AGENTS_REAL_SUPERVISION\n"
        "Task:\n"
        f"{task}\n\n"
        "Planner JSON:\n"
        f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        "Real observation JSON:\n"
        f"{json.dumps(observation, ensure_ascii=False, indent=2)}\n\n"
        "Produce atomic_task_info for every planner atomic task. Return JSON only."
    )
    supervisor_result = await supervisor.run(
        supervisor_task,
        max_steps=max_steps,
        images=[current_front_image, current_wrist_image],
    )
    raw_supervisor_text = supervisor_result.final_answer
    atomic_task_info = _extract_json_object(raw_supervisor_text)
    _save_json(log_dir / "defense_ei_atomic_task_info.json", atomic_task_info)
    _save_text(log_dir / "defense_ei_atomic_task_info_raw.txt", raw_supervisor_text)

    atomic_infos = _atomic_infos_from_supervisor(atomic_task_info)
    if not atomic_infos:
        return RealDefenseOutputs(
            plan=plan,
            atomic_task_info=atomic_task_info,
            attempts=[],
            completed=True,
            code="",
            execution={"ok": True, "signal": "NO_ATOMIC_TASKS"},
            judge={"task_result": "SUCCESS", "analysis": ""},
            raw_planner_text=raw_planner_text,
            raw_supervisor_text=raw_supervisor_text,
            raw_coder_text="",
            raw_judger_text="",
        )

    attempts: list[dict[str, Any]] = []
    latest_front = current_front_image
    latest_wrist = current_wrist_image
    last_code = ""
    last_execution: dict[str, Any] = {}
    last_judge: dict[str, Any] = {}
    raw_coder_text = ""
    raw_judger_text = ""
    completed = True

    for atomic_index, atomic_info in enumerate(atomic_infos, start=1):
        feedback: dict[str, Any] | None = None
        atomic_done = False
        for attempt_index in range(1, max(1, int(max_attempts_per_atomic)) + 1):
            attempt_dir = log_dir / f"atomic_{atomic_index:02d}" / f"attempt_{attempt_index:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _save_json(attempt_dir / "atomic_task_info.json", atomic_info)

            coder_task = (
                "DEFENSE_EI_AGENTS_REAL_CODING\n"
                "Use the real-robot-code-contract and primitive-skill-contract skills. "
                "Do not call simulation tools.\n\n"
                "High-level task:\n"
                f"{task}\n\n"
                "Current atomic task info JSON:\n"
                f"{json.dumps(atomic_info, ensure_ascii=False, indent=2)}\n\n"
                "Current observation JSON:\n"
                f"{json.dumps({'front_rgb': str(latest_front), 'wrist_rgb': str(latest_wrist)}, ensure_ascii=False, indent=2)}\n\n"
                "Prior judger feedback JSON for this same atomic task, or null:\n"
                f"{json.dumps(feedback, ensure_ascii=False, indent=2)}\n\n"
                "Return runnable Python code only."
            )
            coder_result = await coder.run(
                coder_task,
                max_steps=max_steps,
                images=[latest_front, latest_wrist],
            )
            raw_coder_text = coder_result.final_answer
            last_code = _extract_python_code(raw_coder_text)
            _save_text(attempt_dir / "real_atomic_actions.py", last_code)
            _save_text(attempt_dir / "real_atomic_actions_raw.txt", raw_coder_text)
            _save_text(log_dir / "real_atomic_actions.py", last_code)

            try:
                runtime_report = _check_generated_code(last_code)
                syntax_report = {
                    "ok": True,
                    "signal": "SYNTAX_AND_RUNTIME_TOOL_CHECK_OK",
                    "runtime_tool_check": runtime_report,
                }
            except Exception:
                syntax_report = {
                    "ok": False,
                    "signal": "SYNTAX_OR_RUNTIME_TOOL_CHECK_FAILED",
                    "error": traceback.format_exc(),
                }
            _save_json(attempt_dir / "syntax_check.json", syntax_report)

            if not bool(syntax_report.get("ok")):
                last_execution = {"ok": False, "signal": "SYNTAX_OR_RUNTIME_TOOL_CHECK_FAILED"}
            else:
                last_execution = _execute_real_code(code=last_code, controller=controller)
            _save_json(attempt_dir / "real_execution.json", last_execution)
            if not bool(last_execution.get("ok")):
                _save_text(attempt_dir / "real_execution_error.txt", str(last_execution.get("error", "")))

            after_front = attempt_dir / "after_front_rgb.png"
            after_wrist = attempt_dir / "after_wrist_rgb.png"
            _capture_or_copy_images(
                capture_images_fn=capture_images_fn,
                target_front=after_front,
                target_wrist=after_wrist,
            )

            judge_prompt = (
                "DEFENSE_EI_AGENTS_REAL_JUDGING\n"
                "High-level task:\n"
                f"{task}\n\n"
                "Atomic task info JSON:\n"
                f"{json.dumps(atomic_info, ensure_ascii=False, indent=2)}\n\n"
                "Generated and executed atomic action code:\n"
                "```python\n"
                f"{last_code}\n"
                "```\n\n"
                "Real execution report:\n"
                f"{json.dumps(last_execution, ensure_ascii=False, indent=2)}\n\n"
                "Return JSON only with keys task_result and analysis."
            )
            judger_result = await judger.run(
                judge_prompt,
                max_steps=max_steps,
                images=[after_front, after_wrist],
            )
            raw_judger_text = judger_result.final_answer
            last_judge = _extract_json_object(raw_judger_text)
            _save_json(attempt_dir / "real_atomic_judge.json", last_judge)
            _save_text(attempt_dir / "real_atomic_judge_raw.txt", raw_judger_text)

            attempts.append(
                {
                    "atomic_id": atomic_info.get("id", atomic_index),
                    "attempt": attempt_index,
                    "syntax": syntax_report,
                    "execution": last_execution,
                    "judge": last_judge,
                    "after_front_rgb": str(after_front),
                    "after_wrist_rgb": str(after_wrist),
                }
            )

            if _is_success_judge(last_judge):
                latest_front = after_front
                latest_wrist = after_wrist
                atomic_done = True
                break

            feedback = {
                "task_result": "FAIL",
                "analysis": str(last_judge.get("analysis", "")).strip(),
                "execution": last_execution,
            }
            restore_report = _restore_robot_initial_state(controller, initial_robot_state)
            _save_json(attempt_dir / "restore_initial_robot_state.json", restore_report)
            restored_front = attempt_dir / "restored_front_rgb.png"
            restored_wrist = attempt_dir / "restored_wrist_rgb.png"
            _capture_or_copy_images(
                capture_images_fn=capture_images_fn,
                target_front=restored_front,
                target_wrist=restored_wrist,
            )
            latest_front = restored_front
            latest_wrist = restored_wrist

        if not atomic_done:
            completed = False
            break

    _save_json(log_dir / "defense_ei_attempts.json", {"attempts": attempts, "completed": completed})
    _save_json(log_dir / "real_execution.json", last_execution)
    _save_json(log_dir / "real_finished_view_judge.json", last_judge)
    _save_text(log_dir / "real_finished_view_judge_raw.txt", raw_judger_text)

    return RealDefenseOutputs(
        plan=plan,
        atomic_task_info=atomic_task_info,
        attempts=attempts,
        completed=completed,
        code=last_code,
        execution=last_execution,
        judge=last_judge,
        raw_planner_text=raw_planner_text,
        raw_supervisor_text=raw_supervisor_text,
        raw_coder_text=raw_coder_text,
        raw_judger_text=raw_judger_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DefenseAgent real-robot defense_ei_agents evaluation.")
    parser.add_argument("--task", required=True, help="High-level task description.")
    parser.add_argument("--planner-profile", type=Path, default=DEFAULT_PLANNER_PROFILE)
    parser.add_argument("--supervisor-profile", type=Path, default=DEFAULT_SUPERVISOR_PROFILE)
    parser.add_argument("--coder-profile", type=Path, default=DEFAULT_CODER_PROFILE)
    parser.add_argument("--judger-profile", type=Path, default=DEFAULT_JUDGER_PROFILE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-attempts-per-atomic", type=int, default=3)

    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--robotiq-urscript-defs-path", default="")
    parser.add_argument("--strict-gripper-connection", action="store_true")
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Only capture current front/wrist RGB images, then exit before agent generation or robot connection.",
    )

    parser.add_argument("--camera-serials", default="", help="Comma-separated RealSense serials: front,wrist. Auto-discovers if omitted.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=15)
    parser.add_argument("--current-front-image", type=Path, default=None, help="Use an existing front RGB image instead of capturing.")
    parser.add_argument("--current-wrist-image", type=Path, default=None, help="Use an existing wrist RGB image instead of capturing.")

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
    serials = [s.strip() for s in str(args.camera_serials).split(",") if s.strip()]

    controller = UR7eVectorController(
        robot_ip=str(args.robot_ip),
        robotiq_urscript_defs_path=str(args.robotiq_urscript_defs_path).strip() or None,
        strict_gripper_connection=bool(args.strict_gripper_connection),
    )

    source_front = _resolve_path(args.current_front_image) if args.current_front_image else None
    source_wrist = _resolve_path(args.current_wrist_image) if args.current_wrist_image else None
    planner_profile = _prepare_real_profile_copy(_resolve_path(args.planner_profile), run_dir, "planner")
    supervisor_profile = _prepare_real_profile_copy(_resolve_path(args.supervisor_profile), run_dir, "supervisor")
    coder_profile = _prepare_real_profile_copy(_resolve_path(args.coder_profile), run_dir, "coder")
    judger_profile = _prepare_real_profile_copy(_resolve_path(args.judger_profile), run_dir, "judger")
    if source_front or source_wrist:
        if not source_front or not source_wrist:
            raise ValueError("--current-front-image and --current-wrist-image must be provided together.")
        if not source_front.exists() or not source_wrist.exists():
            raise FileNotFoundError(f"current image(s) not found: {source_front}, {source_wrist}")
        current_front = source_front
        current_wrist = source_wrist

        _log("Starting RealSense capture for post-execution feedback")
        with RealSenseRGBPair(
            serials=serials,
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            warmup_frames=int(args.camera_warmup_frames),
        ) as cameras:
            try:
                _log(f"Connecting UR7e controller at {args.robot_ip}")
                controller.connect()
                if controller.is_gripper_available():
                    _log(f"Gripper backend: {controller.get_gripper_backend()}")

                def capture_images(front_path: Path, wrist_path: Path) -> None:
                    front_done, wrist_done = cameras.capture()
                    imageio.imwrite(front_path, front_done)
                    imageio.imwrite(wrist_path, wrist_done)

                outputs = await run_defense_agent_real_once(
                    task=str(args.task).strip(),
                    current_front_image=current_front,
                    current_wrist_image=current_wrist,
                    planner_profile=planner_profile,
                    supervisor_profile=supervisor_profile,
                    coder_profile=coder_profile,
                    judger_profile=judger_profile,
                    log_dir=run_dir,
                    controller=controller,
                    max_steps=args.max_steps,
                    max_attempts_per_atomic=int(args.max_attempts_per_atomic),
                    embedding_model=str(args.embedding_model),
                    embedding_api_key=args.embedding_api_key,
                    embedding_base_url=args.embedding_base_url,
                    embedding_dims=int(args.embedding_dims),
                    capture_images_fn=capture_images,
                )
            finally:
                controller.close()

        summary = {
            "run_dir": str(run_dir),
            "current_front_image": str(current_front),
            "current_wrist_image": str(current_wrist),
            "plan_status": outputs.plan.get("status"),
            "atomic_task_count": len(_atomic_infos_from_supervisor(outputs.atomic_task_info)),
            "completed": outputs.completed,
            "attempt_count": len(outputs.attempts),
            "execution_ok": bool(outputs.execution.get("ok")),
            "execution_signal": outputs.execution.get("signal"),
            "judge_result": outputs.judge.get("task_result"),
        }
        _save_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if outputs.completed else 2

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

        if args.capture_only:
            summary = {
                "run_dir": str(run_dir),
                "capture_only": True,
                "front_image": str(current_front),
                "wrist_image": str(current_wrist),
                "front_shape": list(front_img.shape),
                "wrist_shape": list(wrist_img.shape),
                "front_exists": current_front.exists(),
                "wrist_exists": current_wrist.exists(),
            }
            _save_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        try:
            _log(f"Connecting UR7e controller at {args.robot_ip}")
            controller.connect()
            if controller.is_gripper_available():
                _log(f"Gripper backend: {controller.get_gripper_backend()}")

            def capture_images(front_path: Path, wrist_path: Path) -> None:
                front_done, wrist_done = cameras.capture()
                imageio.imwrite(front_path, front_done)
                imageio.imwrite(wrist_path, wrist_done)

            outputs = await run_defense_agent_real_once(
                task=str(args.task).strip(),
                current_front_image=current_front,
                current_wrist_image=current_wrist,
                planner_profile=planner_profile,
                supervisor_profile=supervisor_profile,
                coder_profile=coder_profile,
                judger_profile=judger_profile,
                log_dir=run_dir,
                controller=controller,
                max_steps=args.max_steps,
                max_attempts_per_atomic=int(args.max_attempts_per_atomic),
                embedding_model=str(args.embedding_model),
                embedding_api_key=args.embedding_api_key,
                embedding_base_url=args.embedding_base_url,
                embedding_dims=int(args.embedding_dims),
                capture_images_fn=capture_images,
            )

        finally:
            controller.close()

    summary = {
        "run_dir": str(run_dir),
        "plan_status": outputs.plan.get("status"),
        "atomic_task_count": len(_atomic_infos_from_supervisor(outputs.atomic_task_info)),
        "completed": outputs.completed,
        "attempt_count": len(outputs.attempts),
        "execution_ok": bool(outputs.execution.get("ok")),
        "execution_signal": outputs.execution.get("signal"),
        "judge_result": outputs.judge.get("task_result"),
    }
    _save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if outputs.completed else 2


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
