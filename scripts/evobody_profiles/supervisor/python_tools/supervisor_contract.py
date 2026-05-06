"""Supervisor output contract helpers for DefenseAgent workflow."""

from __future__ import annotations

import json
from typing import Any


def extract_supervisor_json(payload_text: str) -> str:
    """Extract JSON object text from raw model output."""
    text = (payload_text or "").strip()
    if not text:
        raise ValueError("empty payload")

    lower = text.lower()
    fence = "```json"
    if fence in lower:
        start = lower.find(fence)
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
    """Return the strict JSON schema expected from task supervisor output."""
    return {
        "status": "in_progress | completed",
        "summary": "short text",
        "atomic_tasks": [
            {
                "id": "int",
                "description": "str",
                "primitive": "pick_place|push|pull|press|open|close|pour",
                "source_object": "str|null",
                "target_object": "str|null",
                "constraints": ["str"],
                "done_criteria": "str",
            }
        ],
    }


def validate_supervisor_json(payload_text: str) -> str:
    """Validate top-level supervisor JSON shape and return a compact report."""
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("root must be an object")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"in_progress", "completed"}:
        raise ValueError("status must be in_progress or completed")
    atomic_tasks = payload.get("atomic_tasks", [])
    if not isinstance(atomic_tasks, list):
        raise ValueError("atomic_tasks must be a list")
    return json.dumps(
        {
            "ok": True,
            "status": status,
            "atomic_task_count": len(atomic_tasks),
        },
        ensure_ascii=False,
    )


def extract_and_validate_supervisor_json(raw_text: str) -> str:
    """Extract json from raw text, then validate supervisor output."""
    extracted = extract_supervisor_json(raw_text)
    payload = json.loads(extracted)
    report_raw = validate_supervisor_json(extracted)
    report = json.loads(report_raw)
    report["extracted_json"] = payload
    return json.dumps(report, ensure_ascii=False)
