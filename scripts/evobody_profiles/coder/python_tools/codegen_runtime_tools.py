"""Runtime helpers for code generator validation and API signature extraction."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

EVO_PHASE_LINE_RE = re.compile(
    r"^#\s*===\s*EVO_PHASE:\s*(?P<slug>[^\|]+?)\s*\|\s*(?P<goal>.+?)\s*===\s*$"
)


def extract_code_block(content: str) -> str:
    match = re.search(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.replace("```python", "").replace("```", "").strip()


def count_evo_phase_headers(code: str) -> int:
    return sum(1 for line in code.splitlines() if EVO_PHASE_LINE_RE.match(line))


def extract_atomic_api_signatures(project_root: Path) -> str:
    atomic_ops_path = project_root / "scripts" / "evoma_atomic_ops.py"
    if not atomic_ops_path.exists():
        return "# evoma_atomic_ops.py not found"
    text = atomic_ops_path.read_text(encoding="utf-8")
    signatures: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            signatures.append(stripped[:-1])
    return "\n".join(signatures) if signatures else "# no public signatures found"


def syntax_check_generated_code(
    code: str,
    *,
    forbidden_checker: Any,
) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except Exception as exc:
        return False, f"syntax error: {exc}"

    try:
        report_raw = forbidden_checker(code)
        report = json.loads(report_raw)
        if not bool(report.get("ok", False)):
            return False, f"forbidden tokens: {report.get('forbidden_hits', [])}"
    except Exception as exc:
        return False, f"python_tool check failed: {exc}"

    has_action_api = any(
        fn in code
        for fn in [
            "move_to(",
            "move_ee(",
            "gripper_control(",
            "pick_and_place(",
            "push(",
            "pull(",
            "press(",
            "open(",
            "close(",
            "pour(",
        ]
    )
    if not has_action_api:
        return False, "missing action API call"
    if count_evo_phase_headers(code) < 2:
        return False, "missing EVO_PHASE markers"
    return True, "ok"
