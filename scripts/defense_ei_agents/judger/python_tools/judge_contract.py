"""Atomic task judge JSON contract helpers for defense_ei_agents."""

from __future__ import annotations

import json


def extract_task_judge_json(payload_text: str) -> str:
    """Extract JSON object text from raw judge output."""
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


def task_judge_schema() -> dict[str, str]:
    """Return strict output schema for an atomic task judge."""
    return {
        "task_result": "SUCCESS | FAIL",
        "analysis": "If FAIL, explain concrete failure factors for coder retry.",
    }


def normalize_task_judge_json(payload_text: str) -> str:
    """Normalize judge output to canonical JSON keys."""
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("root must be an object")
    task_result = str(payload.get("task_result", payload.get("verdict", ""))).strip().upper()
    if task_result not in {"SUCCESS", "FAIL"}:
        raise ValueError("task_result must be SUCCESS or FAIL")
    analysis = str(payload.get("analysis", "")).strip()
    return json.dumps({"task_result": task_result, "analysis": analysis}, ensure_ascii=False)


def extract_and_normalize_task_judge_json(raw_text: str) -> str:
    """Extract json from raw text, then normalize judge output."""
    extracted = extract_task_judge_json(raw_text)
    return normalize_task_judge_json(extracted)
