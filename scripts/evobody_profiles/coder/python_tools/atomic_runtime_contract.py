"""Atomic code generation contract helpers for DefenseAgent workflow."""

from __future__ import annotations

import json
from typing import Any


def runtime_api_catalog() -> dict[str, Any]:
    """Return allowed runtime APIs for generated robotics code."""
    return {
        "primitive": [
            "move_to(pos, quat, num_steps)",
            "move_ee(dx, dy, dz, droll, dpitch, dyaw, steps)",
            "gripper_control(value, delay)",
            "ee_pose()",
        ],
        "composite": [
            "pick_and_place(...)",
            "push(...)",
            "pull(...)",
            "press(...)",
            "open(...)",
            "close(...)",
            "pour(...)",
            "move_x(...)",
            "move_y(...)",
            "move_z(...)",
            "rotate_x(...)",
            "rotate_y(...)",
            "rotate_z(...)",
        ],
        "helpers": [
            "get_object_abs_pose(object_poses, object_name)",
            "recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg)",
        ],
    }


def check_forbidden_tokens(code: str) -> str:
    """Return a JSON report indicating whether forbidden tokens are present."""
    forbidden = ["def ", "class ", "import ", "from ", "open(", "exec(", "eval(", "subprocess", "os.", "sys."]
    hit = [tok for tok in forbidden if tok in code]
    return json.dumps({"ok": len(hit) == 0, "forbidden_hits": hit}, ensure_ascii=False)
