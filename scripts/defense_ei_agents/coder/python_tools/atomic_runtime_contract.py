"""Atomic code generation contract helpers for defense_ei_agents."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


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


def runtime_api_catalog(environment: str = "real") -> dict[str, Any]:
    """Return allowed runtime APIs for generated robotics code."""
    env = str(environment or "real").strip().lower()
    if env not in {"real", "real_robot", "real-robot", "ur7e", "hardware"}:
        raise ValueError("defense_ei_agents supports only the real UR7e runtime")
    return {
        "environment": "real",
        "primitive": [
            "move_ee(dx, dy, dz, droll, dpitch, dyaw, velocity=0.04, acceleration=0.18, wait_after_arm_s=0.2)",
            "gripper_control(value, delay)",
            "ee_pose()",
        ],
        "convenience": [
            "move_x(...)",
            "move_y(...)",
            "move_z(...)",
            "rotate_x(...)",
            "rotate_y(...)",
            "rotate_z(...)",
            "sleep(seconds)",
        ],
        "notes": [
            "UR7e real runtime; no simulation tools.",
            "Absolute TCP pose vectors are [x,y,z,rx,ry,rz]; xyz is meters and rx/ry/rz is UR rotation-vector radians.",
            "move_ee translation increments dx/dy/dz are millimeters in the gripper/wrist-image frame.",
            "move_ee rotation increments droll/dpitch/dyaw are radians.",
            "Use move_ee for all incremental motion; move_to is not exposed to generated code.",
            "Wrist image right is gripper +X, wrist image down is gripper +Y, and wrist image forward is gripper +Z.",
            "Use object-safe slow motion: small deltas, low velocity/acceleration, and pauses near contact.",
            "Gripper value is 0..255, where 0=open and 255=closed.",
            "Do not generate quaternion code or four-value rotation literals.",
            "Primitive names such as pick_place, push, pull, press, open, close, and pour are labels only; expand them into move_ee/gripper_control phases instead of calling them.",
        ],
    }


def extract_python_code(raw_text: str) -> str:
    """Extract the Python payload from a model response."""
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


def check_forbidden_tokens(code: str) -> str:
    """Return a JSON report indicating whether forbidden tokens are present."""
    forbidden = [
        "def ",
        "class ",
        "import ",
        "from ",
        "exec(",
        "eval(",
        "subprocess",
        "os.",
        "sys.",
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
    ]
    lowered = (code or "").lower()
    hit = [tok for tok in forbidden if tok in lowered]
    return json.dumps({"ok": len(hit) == 0, "forbidden_hits": hit}, ensure_ascii=False)


def syntax_check_code_via_ast_tree(code: str) -> str:
    """Validate syntax and forbidden AST nodes for generated code."""
    tree = ast.parse(code or "", mode="exec")
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
        return json.dumps({"ok": False, "forbidden_hits": sorted(set(hits))}, ensure_ascii=False)
    compile(tree, "<defense_ei_generated_code>", "exec")
    return json.dumps({"ok": True, "forbidden_hits": []}, ensure_ascii=False)


def validate_runtime_tool_calls(code: str) -> str:
    """Validate that generated code calls only the real runtime API surface."""
    tree = ast.parse(code or "", mode="exec")
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

    ok = not forbidden_hits and not unknown_calls
    return json.dumps(
        {
            "ok": ok,
            "called_names": sorted(set(called_names)),
            "forbidden_runtime_calls": sorted(set(forbidden_hits)),
            "unknown_calls": sorted(set(unknown_calls)),
        },
        ensure_ascii=False,
    )
