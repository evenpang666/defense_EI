"""Planner JSON contract helpers for defense_ei_agents."""

from __future__ import annotations

import json
from typing import Any


PRIMITIVE_SKILLS = {"pick_place", "push", "pull", "press", "open", "close", "pour"}


def extract_planner_json(payload_text: str) -> str:
    """Extract a JSON object from raw planner output."""
    text = (payload_text or "").strip()
    if not text:
        raise ValueError("empty payload")

    lower = text.lower()
    if "```json" in lower:
        start = lower.find("```json")
        body_start = text.find("\n", start)
        if body_start != -1:
            end = text.find("```", body_start + 1)
            if end != -1:
                return text[body_start + 1 : end].strip()

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        return text[left : right + 1].strip()
    raise ValueError("no json object found")


def planner_output_schema() -> dict[str, Any]:
    """Return the strict planner output schema."""
    return {
        "status": "in_progress | completed",
        "summary": "short text",
        "atomic_tasks": [
            {
                "id": "int",
                "description": "str",
                "primitive_skill": "pick_place|push|pull|press|open|close|pour",
                "source_object": "str|null",
                "target_object": "str|null",
                "reason": "str",
            }
        ],
    }


def validate_planner_json(payload_text: str) -> str:
    """Validate planner JSON and return a compact report."""
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("root must be an object")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"in_progress", "completed"}:
        raise ValueError("status must be in_progress or completed")
    tasks = payload.get("atomic_tasks")
    if not isinstance(tasks, list):
        raise ValueError("atomic_tasks must be a list")
    if status == "completed" and tasks:
        raise ValueError("completed plans must not include atomic_tasks")
    if status == "in_progress" and not tasks:
        raise ValueError("in_progress plans must include atomic_tasks")
    for expected_id, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError("each atomic task must be an object")
        if int(task.get("id", -1)) != expected_id:
            raise ValueError("atomic task ids must start at 1 and increment by 1")
        primitive = str(task.get("primitive_skill", task.get("primitive", ""))).strip()
        if primitive not in PRIMITIVE_SKILLS:
            raise ValueError(f"unsupported primitive_skill: {primitive}")
    return json.dumps({"ok": True, "status": status, "atomic_task_count": len(tasks)}, ensure_ascii=False)


def extract_and_validate_planner_json(raw_text: str) -> str:
    """Extract and validate planner JSON."""
    extracted = extract_planner_json(raw_text)
    payload = json.loads(extracted)
    report = json.loads(validate_planner_json(extracted))
    report["extracted_json"] = payload
    return json.dumps(report, ensure_ascii=False)
