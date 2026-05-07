"""Supervisor JSON contract helpers for defense_ei_agents."""

from __future__ import annotations

import json
from typing import Any


PRIMITIVE_SKILLS = {"pick_place", "push", "pull", "press", "open", "close", "pour"}


def extract_supervisor_json(payload_text: str) -> str:
    """Extract a JSON object from raw supervisor output."""
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


def supervisor_output_schema() -> dict[str, Any]:
    """Return the strict supervisor output schema."""
    return {
        "status": "in_progress | completed",
        "summary": "short text",
        "atomic_task_info": [
            {
                "id": "int",
                "description": "str",
                "primitive_skill": "pick_place|push|pull|press|open|close|pour",
                "operated_object": "str|null",
                "source_object": "str|null",
                "target_object": "str|null",
                "target_state": "str",
                "motion_type": "translation|rotation|hybrid",
                "constraints": ["str"],
                "done_criteria": "str",
                "failure_risks": ["str"],
            }
        ],
    }


def validate_supervisor_json(payload_text: str) -> str:
    """Validate supervisor JSON and return a compact report."""
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("root must be an object")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"in_progress", "completed"}:
        raise ValueError("status must be in_progress or completed")
    infos = payload.get("atomic_task_info")
    if not isinstance(infos, list):
        raise ValueError("atomic_task_info must be a list")
    if status == "completed" and infos:
        raise ValueError("completed supervisor output must not include atomic_task_info")
    for expected_id, info in enumerate(infos, start=1):
        if not isinstance(info, dict):
            raise ValueError("each atomic_task_info item must be an object")
        if int(info.get("id", -1)) != expected_id:
            raise ValueError("atomic task ids must start at 1 and increment by 1")
        primitive = str(info.get("primitive_skill", "")).strip()
        if primitive not in PRIMITIVE_SKILLS:
            raise ValueError(f"unsupported primitive_skill: {primitive}")
        if not str(info.get("target_state", "")).strip():
            raise ValueError("target_state is required")
        if not str(info.get("done_criteria", "")).strip():
            raise ValueError("done_criteria is required")
    return json.dumps({"ok": True, "status": status, "atomic_task_count": len(infos)}, ensure_ascii=False)


def extract_and_validate_supervisor_json(raw_text: str) -> str:
    """Extract and validate supervisor JSON."""
    extracted = extract_supervisor_json(raw_text)
    payload = json.loads(extracted)
    report = json.loads(validate_supervisor_json(extracted))
    report["extracted_json"] = payload
    return json.dumps(report, ensure_ascii=False)
