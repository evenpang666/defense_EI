"""Feedback and phase parsing helpers for iterative code regeneration."""

from __future__ import annotations

import json
import re

EVO_PHASE_LINE_RE = re.compile(
    r"^#\s*===\s*EVO_PHASE:\s*(?P<slug>[^\|]+?)\s*\|\s*(?P<goal>.+?)\s*===\s*$"
)


def parse_evo_phase_segments(code: str) -> tuple[str, list[dict[str, str]]]:
    lines = code.splitlines()
    prologue: list[str] = []
    segments: list[dict[str, str]] = []
    cur_slug: str | None = None
    cur_goal: str | None = None
    cur_body: list[str] = []

    def _flush() -> None:
        nonlocal cur_slug, cur_goal, cur_body
        if cur_slug is not None:
            segments.append({"slug": cur_slug, "goal": cur_goal or "", "body": "\n".join(cur_body).strip()})
        cur_slug = None
        cur_goal = None
        cur_body = []

    for line in lines:
        m = EVO_PHASE_LINE_RE.match(line)
        if m:
            _flush()
            cur_slug = m.group("slug").strip()
            cur_goal = (m.group("goal") or "").strip()
            cur_body = []
            continue
        if cur_slug is None:
            prologue.append(line)
        else:
            cur_body.append(line)
    _flush()
    return "\n".join(prologue).strip(), segments


def format_judge_feedback_for_atomic_skill(judge_text: str) -> str:
    raw = (judge_text or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "### prior_simulation_judge_feedback\n" + raw[:12000]
    if not isinstance(data, dict):
        return "### prior_simulation_judge_feedback\n" + raw[:12000]
    task_result = str(data.get("task_result", data.get("verdict", ""))).strip().upper()
    analysis = str(data.get("analysis", "")).strip()
    return (
        "### prior_simulation_judge_feedback\n"
        f"- task_result: {task_result or 'UNKNOWN'}\n\n"
        f"analysis:\n{analysis}"
    ).strip()


def format_segment_failure_for_atomic_skill(
    *,
    full_code: str,
    failed_index: int,
    segments: list[dict[str, str]],
    judge_text: str,
) -> str:
    judge_fmt = format_judge_feedback_for_atomic_skill(judge_text)
    fi = failed_index if 0 <= failed_index < len(segments) else 0
    failed = segments[fi] if segments else {}
    return (
        "### prior_segment_judge_feedback\n"
        f"failed_segment_index: {fi}\n"
        f"failed_slug: {failed.get('slug', '')}\n"
        f"failed_goal: {failed.get('goal', '')}\n\n"
        f"{judge_fmt}\n\n"
        f"### full_code_before_revision\n```python\n{(full_code or '').strip()}\n```"
    )
