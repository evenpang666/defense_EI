import ast
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


REPORT_DONE_TOKEN = "Task Completed"
MODEL_NAME = "qwen/qwen3.5-397b-a17b"
PRE_SCENE_PATH = Path("logs/pre_scene.json")
PLAN_PATH = Path("logs/evoma_plan.json")
SCENE_CONTEXT_PATH = Path("logs/evoma_scene_context.json")
DEFAULT_JUDGE_JSON_PATH = Path("logs/judge_result.json")
RAW_CODE_PATH = Path("logs/evoma_atomic_actions_raw.py")
ATOMIC_CODE_PATH = Path("logs/atomic_actions.py")
EXTRA_ATOMIC_API_PATH = Path("scripts/evoma_atomic_ops.py")
GRASP_OFFSETS_PATH = Path("grasp_offsets.json")
EVOMA_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLETOP_CLEARANCE_MARGIN_M = 0.08
# Upper bound for planner-injected safe carry height (world z, meters).
TABLETOP_SAFE_CARRY_Z_MAX_M = 1.4

# Machine-readable phased execution markers (see AtomicActionSkill system prompt).
EVO_PHASE_LINE_RE = re.compile(
    r"^#\s*===\s*EVO_PHASE:\s*(?P<slug>[^\|]+?)\s*\|\s*(?P<goal>.+?)\s*===\s*$"
)


def parse_evo_phase_segments(code: str) -> tuple[str, list[dict[str, str]]]:
    """
    Split generated atomic code into a prologue and ordered phase bodies.

    Phase headers are single-line comments:
        # === EVO_PHASE: <slug> | <goal> ===

    Returns:
        (prologue, [{"slug", "goal", "body"}, ...])
    If no phase headers exist, returns ("", []) so callers can fall back to monolithic execution.
    """
    lines = code.splitlines()
    prologue: list[str] = []
    segments: list[dict[str, str]] = []
    cur_slug: str | None = None
    cur_goal: str | None = None
    cur_body: list[str] = []

    def _flush() -> None:
        nonlocal cur_slug, cur_goal, cur_body
        if cur_slug is not None:
            segments.append(
                {
                    "slug": cur_slug,
                    "goal": cur_goal or "",
                    "body": "\n".join(cur_body).strip(),
                }
            )
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
        else:
            if cur_slug is None:
                prologue.append(line)
            else:
                cur_body.append(line)
    _flush()
    return "\n".join(prologue).strip(), segments


def count_evo_phase_headers(code: str) -> int:
    return sum(1 for line in code.splitlines() if EVO_PHASE_LINE_RE.match(line))


def format_segment_failure_for_atomic_skill(
    *,
    full_code: str,
    failed_index: int,
    segments: list[dict[str, str]],
    judge_text: str,
) -> str:
    """
    Build extra_user_context for AtomicActionSkill after a **segment** judge failure.
    """
    judge_fmt = format_judge_feedback_for_atomic_skill(judge_text)
    seg_lines = []
    fi = failed_index if 0 <= failed_index < len(segments) else 0
    for i, seg in enumerate(segments):
        marker = f"# === EVO_PHASE: {seg.get('slug', '')} | {seg.get('goal', '')} ==="
        body = (seg.get("body") or "").strip()
        seg_lines.append(f"### Segment {i} ({'FAILED' if i == fi else 'ok'})\n{marker}\n```python\n{body}\n```")
    segments_doc = "\n\n".join(seg_lines)
    failed = segments[fi] if segments else {}
    return (
        "### prior_segment_judge_feedback (stage judge feedback)\n"
        "A mid-rollout **segment** judge rejected one motion stage. You must return the **full** "
        "runnable script again (all `EVO_PHASE` markers preserved, same rules as initial generation), "
        f"revising mainly the failed stage **#{fi}** `{failed.get('slug', '')}` "
        f"— goal: {failed.get('goal', '')!r}.\n\n"
        f"{judge_fmt}\n\n"
        "### segmented_code_snapshot\n"
        f"{segments_doc}\n\n"
        "### full_code_before_revision\n```python\n{(full_code or '').strip()}\n```\n"
    )


def validate_code(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "The code syntax is correct."
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"
    except IndentationError as exc:
        return False, f"Indentation error: {exc}"
    except Exception as exc:
        return False, f"Code validation failed: {exc}"


def file_to_data_url(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path.resolve()}")

    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _get_client() -> OpenAI:
    base_url = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_code_block(content: str) -> str:
    match = re.search(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()

    return content.replace("```python", "").replace("```", "").strip()


def _extract_json_block(content: str) -> str:
    fenced = re.search(r"```json(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    plain = re.search(r"\{.*\}", content, re.DOTALL)
    if plain:
        return plain.group(0).strip()
    return content.strip()


def _parse_judge_output(model_text: str, task_prompt: str, image_path: str) -> dict[str, Any]:
    """Parse evaluator LLM output into a dict suitable for JSON export."""
    raw = (model_text or "").strip()
    base: dict[str, Any] = {
        "task_prompt": task_prompt,
        "image_path": str(Path(image_path).resolve()),
    }
    try:
        blob = _extract_json_block(raw)
        data = json.loads(blob)
        if not isinstance(data, dict):
            raise ValueError("root must be a JSON object")
        task_result = str(data.get("task_result", data.get("verdict", ""))).strip().upper()
        if task_result not in ("SUCCESS", "FAIL"):
            raise ValueError("task_result must be SUCCESS or FAIL")
        analysis = str(data.get("analysis", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if not reason and analysis:
            reason = analysis.splitlines()[0].strip()[:160]
        out = {
            **base,
            "task_result": task_result,
            "analysis": analysis,
            # Backward-compatible aliases for legacy downstream consumers.
            "verdict": task_result,
            "reason": reason,
        }
        return out
    except Exception as exc:
        first = raw.splitlines()[0].strip().upper() if raw else ""
        legacy_ok = first == "SUCCESS"
        return {
            **base,
            "task_result": "SUCCESS" if legacy_ok else "FAIL",
            "analysis": raw if raw else "",
            # Backward-compatible aliases for legacy downstream consumers.
            "verdict": "SUCCESS" if legacy_ok else "FAIL",
            "reason": "Parsed via legacy first-line fallback" if legacy_ok else f"JSON parse failed: {exc}",
            "parse_error": True,
            "raw_model_text": raw[:12000],
        }


def format_judge_feedback_for_atomic_skill(judge_text: str) -> str:
    """
    Turn judge() JSON (or legacy plain text) into explicit **task_result** / **analysis**
    context for ``AtomicActionSkill.generate_atomic_actions(..., extra_user_context=...)``.
    """
    raw = (judge_text or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return (
            "### prior_simulation_judge_feedback (non-JSON)\n"
            "Use this evaluator text when revising the motion plan.\n\n"
            f"{raw[:12000]}"
        )
    if not isinstance(data, dict):
        return f"### prior_simulation_judge_feedback\n{raw[:12000]}"
    task_result = str(data.get("task_result", data.get("verdict", ""))).strip().upper()
    analysis = str(data.get("analysis", "")).strip()
    lines = [
        "### prior_simulation_judge_feedback",
        "The last rollout was judged from the final camera view(s) (table and/or wrist). You MUST treat **task_result** and **analysis** below as constraints when revising grasp, approach, lift/carry height, target choice, collision avoidance, and motion ordering.",
        f"- **task_result**: {task_result or 'UNKNOWN'}",
    ]
    if analysis:
        lines.extend(
            [
                "",
                "**analysis** (detailed — follow these observations):",
                analysis,
            ]
        )
    return "\n".join(lines).strip()


def _normalize_quat(quat: list[float] | tuple[float, ...]) -> list[float]:
    if len(quat) != 4:
        return [1.0, 0.0, 0.0, 0.0]
    norm = sum(float(v) * float(v) for v in quat) ** 0.5
    if norm < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in quat]


def _quat_wxyz_to_rotmat(quat: list[float] | tuple[float, ...]) -> list[list[float]]:
    w, x, y, z = _normalize_quat(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _rotate_vec(rot: list[list[float]], vec: list[float]) -> list[float]:
    return [
        rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
        rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
        rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
    ]


def _euler_rpy_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    roll = float(roll_deg) * 3.141592653589793 / 180.0
    pitch = float(pitch_deg) * 3.141592653589793 / 180.0
    yaw = float(yaw_deg) * 3.141592653589793 / 180.0
    cr = (roll * 0.5)
    cp = (pitch * 0.5)
    cy = (yaw * 0.5)
    sr = (roll * 0.5)
    sp = (pitch * 0.5)
    sy = (yaw * 0.5)
    # Avoid importing math; use simple trig via exponent-less approximations is unsafe.
    # Use numpy-like free formulation by delegating to built-in complex trig is not available.
    # We instead compute with oscilation-safe constants through python's math module-free fallback:
    # inline Taylor not robust, so use __import__ is forbidden in generated code only; safe here.
    import math  # local import keeps module surface minimal
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return _normalize_quat([w, x, y, z])


def _quat_mul_wxyz(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = _normalize_quat(q1)
    w2, x2, y2, z2 = _normalize_quat(q2)
    return _normalize_quat(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def recover_grasp_pose_from_offset(
    object_pos_xyz: list[float],
    object_quat_wxyz: list[float],
    offset_pos_xyz: list[float],
    offset_rpy_deg: list[float],
) -> dict[str, list[float]]:
    """
    Recover world grasp pose from relative offset representation.
    offset_pos_xyz: (dx, dy, dz), offset_rpy_deg: (droll, dpitch, dyaw), both relative to object.
    """
    if len(object_pos_xyz) != 3 or len(object_quat_wxyz) != 4:
        return {
            "grasp_pos_world_xyz": [0.0, 0.0, 0.0],
            "grasp_quat_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "grasp_rpy_offset_deg": [0.0, 0.0, 0.0],
        }
    if len(offset_pos_xyz) != 3:
        offset_pos_xyz = [0.0, 0.0, 0.0]
    if len(offset_rpy_deg) != 3:
        offset_rpy_deg = [0.0, 0.0, 0.0]

    obj_pos = [float(v) for v in object_pos_xyz]
    obj_quat = _normalize_quat([float(v) for v in object_quat_wxyz])
    dpos = [float(v) for v in offset_pos_xyz]
    drpy = [float(v) for v in offset_rpy_deg]

    rot_obj = _quat_wxyz_to_rotmat(obj_quat)
    dpos_world = _rotate_vec(rot_obj, dpos)
    grasp_pos_world = [
        round(obj_pos[0] + dpos_world[0], 6),
        round(obj_pos[1] + dpos_world[1], 6),
        round(obj_pos[2] + dpos_world[2], 6),
    ]

    dq = _euler_rpy_deg_to_quat_wxyz(drpy[0], drpy[1], drpy[2])
    grasp_quat_world = [round(v, 6) for v in _quat_mul_wxyz(obj_quat, dq)]
    return {
        "grasp_pos_world_xyz": grasp_pos_world,
        "grasp_quat_world_wxyz": grasp_quat_world,
        "grasp_rpy_offset_deg": [round(v, 6) for v in drpy],
    }


def _is_task_completed(report_text: str) -> bool:
    normalized = " ".join((report_text or "").strip().split()).lower()
    return normalized in {"task completed", "task completeed"}


def _ensure_logs_dir() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)


def _load_extra_atomic_api_text() -> str:
    """Load optional extra atomic API template text for prompt injection."""
    if not EXTRA_ATOMIC_API_PATH.exists():
        return "# extra atomic api file not found: scripts/evoma_atomic_ops.py"
    try:
        return EXTRA_ATOMIC_API_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        return f"# failed to load extra atomic api: {exc}"


def _extract_public_api_signatures(source: str) -> str:
    """Extract only public top-level function signatures + docstrings from atomic ops.

    Private helpers (names starting with '_') are skipped so the prompt does
    NOT encourage the LLM to redefine low-level math/utility code. Only
    pre-registered runtime builtins are surfaced.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        return source

    blocks: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue

        args_src = ast.unparse(node.args) if hasattr(ast, "unparse") else ""
        # Drop private `_xxx=...` keyword defaults so the signature we show
        # matches the runtime call surface (those are internal binding slots).
        public_args = ", ".join(
            part.strip()
            for part in args_src.split(",")
            if part.strip() and not part.strip().startswith("_")
        )

        doc = ast.get_docstring(node) or ""
        if doc:
            doc = doc.strip().splitlines()[0]
        sig = f"def {node.name}({public_args}):"
        if doc:
            blocks.append(f"{sig}\n    \"\"\"{doc}\"\"\"\n    ...")
        else:
            blocks.append(f"{sig}\n    ...")
    return "\n\n".join(blocks) if blocks else "# no public atomic ops exposed"


def _object_template_xml_path(source_key: str) -> Path | None:
    name = (source_key or "").strip().split("/")[-1]
    if not name:
        return None
    candidate = EVOMA_PROJECT_ROOT / "model" / "object" / f"{name}.xml"
    return candidate if candidate.exists() else None


def _geom_points_template_world(model: Any, data: Any, gid: int) -> np.ndarray:
    """Sample points on geom surface in standalone asset XML world frame (MuJoCo)."""
    import mujoco

    gtype = int(model.geom_type[gid])
    pos = np.asarray(data.geom_xpos[gid], dtype=float)
    mat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=float)
    pts: list[np.ndarray] = []

    def add_local(local: np.ndarray) -> None:
        pts.append(pos + mat @ local.reshape(3))

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(size[0])
        for ax in range(3):
            e = np.zeros(3)
            e[ax] = r
            add_local(e)
            add_local(-e)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy, hz = float(size[0]), float(size[1]), float(size[2])
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    add_local(np.array([sx * hx, sy * hy, sz * hz]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hh = float(size[0]), float(size[1])
        for ang in np.linspace(0.0, 2.0 * np.pi, num=12, endpoint=False):
            for z in (-hh, hh):
                add_local(np.array([r * np.cos(ang), r * np.sin(ang), z]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, hh = float(size[0]), float(size[1])
        for ang in np.linspace(0.0, 2.0 * np.pi, num=10, endpoint=False):
            for z in (-hh - r, hh + r):
                add_local(np.array([r * np.cos(ang), r * np.sin(ang), z]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        did = int(model.geom_dataid[gid])
        if did < 0:
            return np.zeros((0, 3))
        adr = int(model.mesh_vertadr[did])
        num = int(model.mesh_vertnum[did])
        verts = np.asarray(model.mesh_vert[adr : adr + num], dtype=float).reshape(-1, 3)
        stride = max(1, int(len(verts) // 4000))
        for v in verts[::stride]:
            add_local(v)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        rx, ry, rz = float(size[0]), float(size[1]), float(size[2])
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    add_local(np.array([sx * rx, sy * ry, sz * rz]))
    else:
        rb = float(model.geom_rbound[gid])
        if rb > 0.0:
            for ax in range(3):
                e = np.zeros(3)
                e[ax] = rb
                add_local(e)
                add_local(-e)

    if not pts:
        return np.zeros((0, 3))
    return np.stack(pts, axis=0)


def _world_points_after_scene_transform(
    template_points: np.ndarray,
    scale_xyz: list[float],
    quat_wxyz: list[float],
    pos_xyz: list[float],
) -> np.ndarray:
    """Apply scene JSON scale (template axes), then world rotation + translation."""
    if template_points.size == 0:
        return template_points.reshape(0, 3)
    s = np.asarray(scale_xyz, dtype=float).reshape(3)
    scaled = template_points * s.reshape(1, 3)
    R = np.asarray(_quat_wxyz_to_rotmat(quat_wxyz), dtype=float)
    p = np.asarray(pos_xyz, dtype=float).reshape(3)
    return (R @ scaled.T).T + p.reshape(1, 3)


def _estimate_asset_vertical_extent_world(
    xml_path: Path,
    scale_xyz: list[float],
    pos_xyz: list[float],
    quat_wxyz: list[float],
) -> tuple[float, float] | None:
    """Return (bottom_z_world, top_z_world) for one placed asset, or None if MuJoCo fails."""
    try:
        import mujoco
    except Exception:
        return None

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception:
        return None

    all_pts: list[np.ndarray] = []
    for gid in range(int(model.ngeom)):
        all_pts.append(_geom_points_template_world(model, data, gid))
    if not all_pts:
        return None
    template_pts = np.vstack(all_pts)
    world_pts = _world_points_after_scene_transform(template_pts, scale_xyz, quat_wxyz, pos_xyz)
    z = world_pts[:, 2]
    return float(z.min()), float(z.max())


def _build_tabletop_clearance(assets: list[dict[str, Any]]) -> dict[str, Any]:
    """World-space vertical extent from model/object/<Name>.xml geoms + scene pose/scale."""
    per_object: list[dict[str, Any]] = []
    tops: list[float] = []

    for obj in assets:
        name = str(obj.get("name", ""))
        key = str(obj.get("source_key", "")).strip()
        xml_path = _object_template_xml_path(key or name)
        scale = obj.get("scale", [1.0, 1.0, 1.0])
        if not isinstance(scale, list) or len(scale) != 3:
            scale = [1.0, 1.0, 1.0]
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        quat = obj.get("quat", [1.0, 0.0, 0.0, 0.0])
        ref_z = float(pos[2]) if isinstance(pos, list) and len(pos) == 3 else 0.0

        if xml_path is None:
            per_object.append(
                {
                    "name": name,
                    "source_key": key,
                    "geom_top_z_world": None,
                    "geom_bottom_z_world": None,
                    "placement_reference_z": round(ref_z, 6),
                    "geometry_status": "xml_not_found",
                }
            )
            continue

        extent = _estimate_asset_vertical_extent_world(
            xml_path,
            [float(scale[0]), float(scale[1]), float(scale[2])],
            list(pos),
            list(quat),
        )
        if extent is None:
            per_object.append(
                {
                    "name": name,
                    "source_key": key,
                    "geom_top_z_world": None,
                    "geom_bottom_z_world": None,
                    "placement_reference_z": round(ref_z, 6),
                    "geometry_status": "mujoco_load_failed",
                }
            )
            continue

        z_lo, z_hi = extent
        tops.append(z_hi)
        obj["geom_bottom_z_world"] = round(z_lo, 4)
        obj["geom_top_z_world"] = round(z_hi, 4)
        obj["placement_reference_z"] = round(ref_z, 6)
        obj["placement_z_note"] = (
            "JSON pos[2] is the placed body reference (often near the bottom), not the physical top. "
            "Use geom_top_z_world (from MuJoCo geoms in model/object/<Name>.xml) for vertical clearance."
        )
        per_object.append(
            {
                "name": name,
                "source_key": key,
                "geom_bottom_z_world": round(z_lo, 4),
                "geom_top_z_world": round(z_hi, 4),
                "placement_reference_z": round(ref_z, 6),
                "geometry_status": "ok",
            }
        )

    safe_z: float | None = None
    safe_uncapped: float | None = None
    if tops:
        safe_uncapped = max(tops) + float(TABLETOP_CLEARANCE_MARGIN_M)
        safe_z = min(float(safe_uncapped), float(TABLETOP_SAFE_CARRY_Z_MAX_M))

    was_capped = bool(
        safe_uncapped is not None and safe_uncapped > float(TABLETOP_SAFE_CARRY_Z_MAX_M) + 1e-9
    )

    return {
        "safe_carry_end_effector_z_m": None if safe_z is None else round(safe_z, 4),
        "safe_carry_uncapped_z_m": None if safe_uncapped is None else round(float(safe_uncapped), 4),
        "safe_carry_z_max_m": float(TABLETOP_SAFE_CARRY_Z_MAX_M),
        "safe_carry_was_capped": was_capped,
        "vertical_margin_m": float(TABLETOP_CLEARANCE_MARGIN_M),
        "per_object": per_object,
        "placement_z_is_body_reference": True,
        "derivation": (
            "MuJoCo loads model/object/<AssetName>.xml; samples geom corners / mesh vertices in template "
            "frame; applies scene JSON scale, quat (wxyz), pos; geom_top_z_world = max world Z over that "
            "asset's geoms (boxes, meshes, etc.). safe_carry_end_effector_z_m is min(uncapped, safe_carry_z_max_m)."
        ),
    }


class ScenePoseReconstructor:
    def __init__(self, scene_path: Path = PRE_SCENE_PATH, grasp_offsets_path: Path = GRASP_OFFSETS_PATH):
        self.scene_path = scene_path
        self.grasp_offsets_path = grasp_offsets_path
        self._grasp_offsets = self._load_grasp_offsets()

    def _load_grasp_offsets(self) -> dict[str, list[dict[str, Any]]]:
        if not self.grasp_offsets_path.exists():
            return {}
        try:
            payload = json.loads(self.grasp_offsets_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        # New format:
        # {
        #   "object/Beaker" or "Beaker": {
        #      "pose_name": {"pos":[dx,dy,dz], "3d_rotation":[droll,dpitch,dyaw]}
        #   }
        # }
        offsets: dict[str, list[dict[str, Any]]] = {}
        if not isinstance(payload, dict):
            return offsets

        for object_key, poses in payload.items():
            if not isinstance(poses, dict):
                continue
            key = str(object_key).strip()
            if not key:
                continue
            entries: list[dict[str, Any]] = []
            for pose_name, pose_cfg in poses.items():
                if not isinstance(pose_cfg, dict):
                    continue
                dpos = pose_cfg.get("pos", [])
                drpy = pose_cfg.get("3d_rotation", [])
                if not isinstance(dpos, list) or len(dpos) != 3:
                    continue
                if not isinstance(drpy, list) or len(drpy) != 3:
                    drpy = [0.0, 0.0, 0.0]
                try:
                    dpos_f = [float(dpos[0]), float(dpos[1]), float(dpos[2])]
                    drpy_f = [float(drpy[0]), float(drpy[1]), float(drpy[2])]
                except (TypeError, ValueError):
                    continue
                entries.append(
                    {
                        "pose_name": str(pose_name).strip() or "grasp_pose",
                        "offset_pos_xyz": dpos_f,
                        "offset_rpy_deg": drpy_f,
                    }
                )
            if not entries:
                continue
            offsets[key] = list(entries)
            offsets[key.split("/")[-1]] = list(entries)
        return offsets

    @staticmethod
    def _object_name_from_entry(entry: dict[str, Any]) -> str:
        key = str(entry.get("key", "")).strip()
        name = str(entry.get("name", "")).strip()
        if name:
            return name
        if key:
            return key.split("/")[-1]
        return "unknown_object"

    @staticmethod
    def _extract_assets(scene: dict[str, Any]) -> list[dict[str, Any]]:
        assets_raw = scene.get("assets", [])
        if not isinstance(assets_raw, list):
            return []

        assets: list[dict[str, Any]] = []
        for idx, item in enumerate(assets_raw, start=1):
            if not isinstance(item, dict):
                continue
            name = ScenePoseReconstructor._object_name_from_entry(item)
            pos = item.get("pos", [0.0, 0.0, 0.0])
            quat = item.get("quat", [1.0, 0.0, 0.0, 0.0])
            if not isinstance(pos, list) or len(pos) != 3:
                pos = [0.0, 0.0, 0.0]
            if not isinstance(quat, list) or len(quat) != 4:
                quat = [1.0, 0.0, 0.0, 0.0]
            scale_raw = item.get("scale", [1.0, 1.0, 1.0])
            if isinstance(scale_raw, list) and len(scale_raw) == 3:
                try:
                    scale_xyz = [float(scale_raw[0]), float(scale_raw[1]), float(scale_raw[2])]
                except (TypeError, ValueError):
                    scale_xyz = [1.0, 1.0, 1.0]
            else:
                scale_xyz = [1.0, 1.0, 1.0]

            assets.append(
                {
                    "id": idx,
                    "name": name,
                    "source_key": str(item.get("key", item.get("name", ""))),
                    "pos": [float(v) for v in pos],
                    "quat": _normalize_quat([float(v) for v in quat]),
                    "scale": scale_xyz,
                }
            )
        return assets

    @staticmethod
    def _extract_robot_pose(scene: dict[str, Any]) -> dict[str, Any]:
        robot_arm = scene.get("robot_arm")
        if isinstance(robot_arm, dict):
            ee_pos = robot_arm.get("ee_pos", [0.0, 0.0, 0.0])
            ee_quat = robot_arm.get("ee_quat", [1.0, 0.0, 0.0, 0.0])
            if isinstance(ee_pos, list) and len(ee_pos) == 3 and isinstance(ee_quat, list) and len(ee_quat) == 4:
                return {
                    "ee_pos": [float(v) for v in ee_pos],
                    "ee_quat": _normalize_quat([float(v) for v in ee_quat]),
                    "source": "robot_arm",
                }

        robot = scene.get("robot")
        if isinstance(robot, dict):
            base_pos = robot.get("base_pos", [0.0, 0.0, 0.824])
            base_quat = robot.get("base_quat", [1.0, 0.0, 0.0, 0.0])
            if isinstance(base_pos, list) and len(base_pos) == 3 and isinstance(base_quat, list) and len(base_quat) == 4:
                return {
                    "ee_pos": [float(v) for v in base_pos],
                    "ee_quat": _normalize_quat([float(v) for v in base_quat]),
                    "source": "robot_base_fallback",
                }

        return {
            "ee_pos": [0.0, 0.0, 1.0],
            "ee_quat": [1.0, 0.0, 0.0, 0.0],
            "source": "default",
        }

    @staticmethod
    def _nearest_neighbors(assets: list[dict[str, Any]], topk: int = 2) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        if len(assets) <= 1:
            return relations

        for src in assets:
            distances: list[tuple[float, dict[str, Any]]] = []
            sx, sy, sz = src["pos"]
            for dst in assets:
                if src["id"] == dst["id"]:
                    continue
                dx = sx - dst["pos"][0]
                dy = sy - dst["pos"][1]
                dz = sz - dst["pos"][2]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                distances.append((dist, dst))
            distances.sort(key=lambda x: x[0])
            for dist, dst in distances[:topk]:
                relations.append(
                    {
                        "from": src["name"],
                        "to": dst["name"],
                        "distance_m": round(float(dist), 4),
                    }
                )
        return relations

    @staticmethod
    def _candidate_poses(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Use a conservative top-down heuristic pose for pick/place seeds.
        candidates: list[dict[str, Any]] = []
        top_down_quat = [0.0, 1.0, 0.0, 0.0]
        for obj in assets:
            x, y, z = obj["pos"]
            candidates.append(
                {
                    "object": obj["name"],
                    "pre_grasp_pos": [round(x, 4), round(y, 4), round(z + 0.12, 4)],
                    "grasp_pos": [round(x, 4), round(y, 4), round(z + 0.04, 4)],
                    "lift_pos": [round(x, 4), round(y, 4), round(z + 0.16, 4)],
                    "suggested_quat_wxyz": top_down_quat,
                }
            )
        return candidates

    def _attach_grasp_references(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for obj in assets:
            source_key = str(obj.get("source_key", "")).strip()
            name = str(obj.get("name", "")).strip()
            all_offsets = self._grasp_offsets.get(source_key) or self._grasp_offsets.get(name)
            if all_offsets is None:
                continue

            for offset_item in all_offsets:
                recovered = recover_grasp_pose_from_offset(
                    object_pos_xyz=obj["pos"],
                    object_quat_wxyz=obj["quat"],
                    offset_pos_xyz=offset_item["offset_pos_xyz"],
                    offset_rpy_deg=offset_item["offset_rpy_deg"],
                )
                references.append(
                    {
                        "object": name,
                        "source_key": source_key,
                        "pose_name": offset_item["pose_name"],
                        "offset_pos_xyz": [round(v, 6) for v in offset_item["offset_pos_xyz"]],
                        "offset_rpy_deg": [round(v, 6) for v in offset_item["offset_rpy_deg"]],
                        "grasp_point_world_xyz": recovered["grasp_pos_world_xyz"],
                        "grasp_quat_world_wxyz": recovered["grasp_quat_world_wxyz"],
                        "object_pos_xyz": [round(v, 6) for v in obj["pos"]],
                        "object_quat_wxyz": [round(v, 6) for v in obj["quat"]],
                    }
                )
        return references

    def _attach_grasp_offset_candidates(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for obj in assets:
            source_key = str(obj.get("source_key", "")).strip()
            name = str(obj.get("name", "")).strip()
            all_offsets = self._grasp_offsets.get(source_key) or self._grasp_offsets.get(name)
            if not all_offsets:
                continue
            for offset_item in all_offsets:
                candidates.append(
                    {
                        "object": name,
                        "source_key": source_key,
                        "pose_name": offset_item["pose_name"],
                        "offset_pos_xyz": [round(v, 6) for v in offset_item["offset_pos_xyz"]],
                        "offset_rpy_deg": [round(v, 6) for v in offset_item["offset_rpy_deg"]],
                        "object_pos_xyz": [round(v, 6) for v in obj["pos"]],
                        "object_quat_wxyz": [round(v, 6) for v in obj["quat"]],
                    }
                )
        return candidates

    def build_context(self, task_prompt: str | None = None) -> dict[str, Any]:
        scene = {}
        if self.scene_path.exists():
            scene = json.loads(self.scene_path.read_text(encoding="utf-8"))

        assets = self._extract_assets(scene)
        tabletop_clearance = _build_tabletop_clearance(assets)
        robot_pose = self._extract_robot_pose(scene)
        relations = self._nearest_neighbors(assets)
        candidates = self._candidate_poses(assets)
        grasp_offset_candidates = self._attach_grasp_offset_candidates(assets)

        keyword_hits: list[str] = []
        text = (task_prompt or "").lower()
        for obj in assets:
            name = obj["name"].lower()
            if name and name in text:
                keyword_hits.append(obj["name"])

        return {
            "scene_path": str(self.scene_path),
            "scene_format": str(scene.get("format", "unknown")),
            "task_prompt": task_prompt or "",
            "robot": robot_pose,
            "objects": assets,
            "tabletop_clearance": tabletop_clearance,
            "relations": relations,
            "candidate_poses": candidates,
            "grasp_offset_candidates": grasp_offset_candidates,
            "keyword_hits": keyword_hits,
        }


class TaskSupervisorAgent:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def analyze_task(
        self,
        task_prompt: str,
        scene_context: dict[str, Any],
        image_path: str = "logs/current_view.png",
    ) -> dict[str, Any]:
        image_data_url = file_to_data_url(image_path)
        system_prompt = """
You are a robotics atomic task supervisor.
Given task text, a front camera image, and reconstructed object poses, produce robust atomic task.
Each atomic task should be an executable block such as: approach -> interaction -> leave-safe.
For instance, pick_place/open/close/push/pull/pour/press is one atomic task.
Within one atomic task precedure, the object pose must be changed by the robotic's actions.

Atomic task definitions:
- pick_place: pick and place an object (composite skill)
- push: push an object (straight skill)
- press: press an object (vertical skill)
- open: open an object (rotation skill)
- close: close an object (rotation skill)
- pour: pour a liquid into a container (rotation skill)
- pull: pull an object (straight skill)

Output must be JSON only using this schema:
{
  "status": "in_progress" | "completed",
  "summary": "short text",
  "atomic_tasks": [
    {
      "id": 1,
      "description": "...",
      "primitive": "pick_place|push|pull|press|open|close|pour",
      "source_object": "name or null",
      "target_object": "name or null",
      "constraints": ["..."],
      "done_criteria": "..."
    }
  ]
}
"""
        user_prompt = f"""
Task:
{task_prompt}

Reconstructed scene context JSON:
{json.dumps(scene_context, ensure_ascii=False, indent=2)}

Requirements:
1. Use object names from scene context whenever possible.
2. If task already completed, return status="completed" and atomic_tasks=[].
3. Keep constraints concrete (collision, clearance, top-down approach, etc.).
"""
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
        )

        content = (resp.output_text or "").strip()
        parsed = self._parse_plan(content)
        if parsed["status"] == "completed":
            return parsed
        if not parsed.get("atomic_tasks"):
            parsed["atomic_tasks"] = [
                {
                    "id": 1,
                    "description": task_prompt,
                    "primitive": "other",
                    "source_object": None,
                    "target_object": None,
                    "constraints": ["avoid collision", "maintain grasp stability"],
                    "done_criteria": "task condition satisfied",
                }
            ]
        return parsed

    @staticmethod
    def _atomic_tasks_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize supervisor JSON: atomic_tasks list, or legacy subtasks; single object becomes one element."""
        raw = payload.get("atomic_tasks")
        if raw is None:
            raw = payload.get("subtasks")
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw_list: list[Any] = [raw]
        elif isinstance(raw, list):
            raw_list = raw
        else:
            return []
        out: list[dict[str, Any]] = []
        for item in raw_list:
            if isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _parse_plan(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(_extract_json_block(text))
            if isinstance(payload, dict):
                status = str(payload.get("status", "in_progress")).strip().lower()
                if status not in {"in_progress", "completed"}:
                    status = "in_progress"
                return {
                    "status": status,
                    "summary": str(payload.get("summary", "")),
                    "atomic_tasks": TaskSupervisorAgent._atomic_tasks_from_payload(payload),
                }
        except Exception:
            pass

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and _is_task_completed(lines[0]):
            return {"status": "completed", "summary": REPORT_DONE_TOKEN, "atomic_tasks": []}

        atomic_tasks: list[dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            if line.lower().startswith("sub-task") or line.lower().startswith("subtask"):
                desc = line.split(":", 1)[-1].strip() if ":" in line else line
                atomic_tasks.append(
                    {
                        "id": idx,
                        "description": desc,
                        "primitive": "other",
                        "source_object": None,
                        "target_object": None,
                        "constraints": ["avoid collision"],
                        "done_criteria": "subtask done",
                    }
                )

        return {"status": "in_progress", "summary": "fallback parsed plan", "atomic_tasks": atomic_tasks}


class CodeVerifierAgent:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    @staticmethod
    def _has_forbidden_ops(code: str) -> tuple[bool, str]:
        # Line-level keyword checks (must be at start of a logical line)
        line_prefix_forbidden = ("def ", "class ", "import ", "from ")
        for raw_line in code.splitlines():
            stripped = raw_line.strip()
            for token in line_prefix_forbidden:
                if stripped.startswith(token):
                    return True, f"forbidden definition detected: '{token.strip()}' (all helpers are pre-registered builtins; call them directly)"

        forbidden = [
            "open(",
            "exec(",
            "eval(",
            "__import__",
            "subprocess",
            "os.",
            "sys.",
            "scene_context",
            "lambda ",
            "lambda:",
        ]
        lowered = code.lower()
        for token in forbidden:
            if token.lower() in lowered:
                return True, f"forbidden token detected: {token.strip()}"
        return False, ""

    def verify_and_repair(
        self,
        code: str,
        subtask: dict[str, Any],
        scene_context: dict[str, Any],
    ) -> str:
        is_valid, msg = validate_code(code)
        has_forbidden, forbidden_msg = self._has_forbidden_ops(code)
        _action_fns = [
            "move_to(", "move_ee(", "gripper_control(",
            "pick_and_place(", "push(", "pull(", "press(", "open(", "close(", "pour(",
            "move_x(", "move_y(", "move_z(",
            "rotate_x(", "rotate_y(", "rotate_z(",
        ]
        has_action_api = any(fn in code for fn in _action_fns)
        missing_phases = count_evo_phase_headers(code) < 2

        if is_valid and not has_forbidden and has_action_api and not missing_phases:
            return code

        repair_prompt = f"""
Repair the robotics action code.
Requirements:
1. Output python code block only.
2. ONLY call pre-registered runtime builtins. Do NOT use `def`, `class`,
   `import`, `from`, `lambda`. Do NOT redefine helper functions.
3. Pre-registered primitive APIs: move_to, move_ee, gripper_control, ee_pose, np.
4. Pre-registered composite skills (call directly): pick_and_place, push, pull,
   press, open, close, pour, move_x, move_y, move_z, rotate_x, rotate_y,
   rotate_z, get_object_abs_pose, recover_grasp_pose_from_offset.
5. Also forbidden: open(, exec(, eval(, subprocess, os., sys..
6. Quaternion order is (w, x, y, z).
7. Keep behavior aligned with the atomic task.
8. Do not reference scene_context or the atomic-task JSON at runtime; inline concrete constants directly in code.
9. Horizontal clearance uses **geometry tops**, not `objects[*].pos[2]` (that
   value is a body placement reference / point, often near the bottom, not the
   mesh top). Use `scene_context.tabletop_clearance.safe_carry_end_effector_z_m`
   (already **capped** at `safe_carry_z_max_m`, default 1.4 m) and each
   `objects[*].geom_top_z_world` (from MuJoCo `model/object/<Name>.xml` geoms).
   For `pick_and_place`, choose `lift_height` so `src_pos[2]+lift_height` is **≥**
   that injected threshold but **never higher than `safe_carry_z_max_m`** in
   generated literals. If `safe_carry_end_effector_z_m` is null, add a conservative
   margin above the maximum available `geom_top_z_world`, still capped at 1.4 m.
10. Include **at least two** lines matching `# === EVO_PHASE: slug | goal ===` (see
   AtomicActionSkill system prompt rule 12). Split motion into stages; the last
   stage should conclude the intended action sequence.
11. EVO_PHASE goal text must be action-semantic (robot behavior/outcome), not implementation
    logic. Match API meaning strictly: `pick_and_place` => full pick+place+release macro,
    `push` => contact+push, `pull` => contact/grasp+pull, `press` => directional press,
    `open/close` => articulated open/close manipulation, `pour` => tilt-pour behavior.


Atomic task JSON:
{json.dumps(subtask, ensure_ascii=False, indent=2)}

Scene context JSON:
{json.dumps(scene_context, ensure_ascii=False, indent=2)}

Current code:
```python
{code}
```

Validation message: {msg}
Forbidden check: {forbidden_msg}
Missing action API: {not has_action_api}
Missing EVO_PHASE stage headers (need >= 2): {missing_phases}
"""
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are a strict Python robotics code repairer."},
                {"role": "user", "content": repair_prompt},
            ],
        )
        repaired = _extract_code_block(resp.output_text or "")
        valid2, msg2 = validate_code(repaired)
        bad2, bad_msg2 = self._has_forbidden_ops(repaired)
        if not valid2:
            raise ValueError(f"Repaired code still invalid: {msg2}")
        if bad2:
            raise ValueError(f"Repaired code contains forbidden ops: {bad_msg2}")
        if not any(fn in repaired for fn in _action_fns):
            raise ValueError("Repaired code has no executable action API call")
        if count_evo_phase_headers(repaired) < 2:
            raise ValueError(
                "Repaired code still missing `EVO_PHASE` stage headers (need at least two "
                "`# === EVO_PHASE: slug | goal ===` lines)."
            )
        return repaired


class AtomicActionSkill:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def generate_atomic_actions(
        self,
        subtask: dict[str, Any],
        scene_context: dict[str, Any],
        image_path: str = "logs/current_view.png",
        extra_user_context: str | None = None,
    ) -> str:
        image_data_url = file_to_data_url(image_path)
        atomic_api_signatures = _extract_public_api_signatures(_load_extra_atomic_api_text())
        system_prompt = """
You are a low-level robotics motion planner and Python code writer.
Generate executable atomic action code for ONE atomic task by ONLY calling
pre-registered runtime builtins. You MUST NOT (re)define helper
functions, MUST NOT use `def`, `class`, or `import`.

Primitive runtime APIs (pre-registered, just call them):
1. move_to(pos, quat, num_steps)           # pos (x,y,z); quat (w,x,y,z); num_steps default 100
2. move_ee(dx, dy, dz, droll, dpitch, dyaw, steps)  # relative delta pose; angles in degrees
3. gripper_control(value, delay)           # value 0..255 (0=open,255=closed); delay in ms
4. ee_pose() -> (pos, quat)                # current end-effector pose, quat is (w,x,y,z)

Composite atomic skills (pre-registered, just call them; DO NOT redefine):
- pick_and_place(object_pose, target_pose, direction_x, direction_y, direction_z,
                 approach_height, lift_height,
                 grasp_value, release_value, move_steps, grip_delay)
- push(target_pose, object_pose, push_distance, approach_height, grasp_value,
       move_steps, grip_delay)
- pull(target_pose, object_pose, pull_distance, approach_height, grasp_value,
       release_value, move_steps, grip_delay)
- press(object_pose, direction_x, direction_y, direction_z, grasp_value,
        move_steps, grip_delay)
- open(grasp_pose, rotation_radius, rotation_angle_deg, grasp_value, move_steps, grip_delay)
- close(grasp_pose, rotation_radius, rotation_angle_deg, grasp_value, move_steps, grip_delay)
- pour(object_pose, target_pose, direction_x, direction_y, direction_z,
       rot_x, rot_y, rot_z, approach_height, lift_height,
       grasp_value, release_value, move_steps, grip_delay)
- move_x(distance, steps) / move_y(distance, steps) / move_z(distance, steps)
- rotate_x(angle_deg, steps) / rotate_y(angle_deg, steps) / rotate_z(angle_deg, steps)

API action semantics for EVO_PHASE goal text (STRICT):
- `move_to` / `move_ee` / `move_x|y|z` / `rotate_x|y|z`: describe as arm motion intent
  (approach/align/raise/translate/rotate), not math or helper logic.
- `gripper_control(value<=30)`: describe as opening/releasing gripper.
- `gripper_control(value>=200)`: describe as closing/grasping gripper.
- `pick_and_place(...)`: one complete transfer macro = approach+grasp+lift+transport+place+release.
  **Translation-only:** fixed grasp orientation for the whole skill; motion is only
  vertical and horizontal world-frame segments; `target_pose` uses **xyz only** (quat ignored).
  If this API appears in a stage, that stage goal must reflect the full transfer outcome
  (or explicitly say this stage executes the complete pick-and-place transfer).
- `push(...)`: make contact then push object along target direction/distance.
- `pull(...)`: make contact/grasp then pull object toward target direction/distance.
- `press(...)`: approach then press along the provided direction.
- `open(...)` / `close(...)`: manipulate articulated part through the opening/closing arc.
- `pour(...)`: grasp source, move above/near target, tilt to pour, then recover/release as configured.

Pose helpers (pre-registered, just call them; DO NOT redefine):
- get_object_abs_pose(object_poses, object_name) -> (pos, quat)
- recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz,
    offset_pos_xyz, offset_rpy_deg) -> {"grasp_pos_world_xyz","grasp_quat_world_wxyz"}

Strict rules:
1. Output only one python code block. No prose outside the block.
2. ONLY call the pre-registered functions listed above. Do NOT define any
   helper function or class. Do NOT use `def`, `class`, `import`, `lambda`.
3. Quaternion order everywhere is (w, x, y, z).
4. Start with a single short top comment that restates the atomic task.
5. Prefer composite skills (pick_and_place/push/pull/press/open/close/pour) when
   the atomic task's primitive matches. Fall back to low-level move_to/move_ee
   only when no composite fits.
6. When a grasp is needed, read the relevant entry from the planner-provided
   `grasp_offset_candidates` list in scene context, recover the absolute
   grasp pose using recover_grasp_pose_from_offset(...), and pass it as a
   dict `{"pos": [...], "quat": [...]}` to the composite skill's
   `object_pose` argument. Hardcode the numeric offsets/object poses
   inline — generated runtime code must NOT reference variables named
   `scene_context`, the atomic-task dict, or `plan`.
7. If grasp offsets exist for an object, do not grasp object center by default.
8. Keep motion conservative and intentionally slow: always set explicit
   num_steps / steps values, typically in the 100~200 range. During
   object interaction phases (final approach, contact, grasp/press/push/pull,
   placement/release), prefer the upper end (e.g. 300) to reduce impact
   and improve stability. Ensure a small clearance above grasp targets.
9. Horizontal clearance (critical): whenever the gripper moves laterally, EE
   world z must clear **all tabletop solids by their actual geom envelope tops**,
   not by `objects[*].pos[2]` (that is only a placement reference, not top height).
   The planner injects `tabletop_clearance.safe_carry_end_effector_z_m`
   (= min(max(geom tops)+margin, `safe_carry_z_max_m`); see `safe_carry_was_capped`)
   and per-object `geom_top_z_world` / `geom_bottom_z_world` from MuJoCo parsing of
   `model/object/<Name>.xml` (boxes, meshes, etc.) with scene JSON scale/pose.
   For `pick_and_place`, `lift_height` is added to grasp z (`src_pos[2]+lift_height`);
   set carry z **≥** that threshold and **≤ `safe_carry_z_max_m` (1.4 m)** before any horizontal transfer.
   Same idea for `pour`/`open`/`close` and any lateral `move_to`/`move_ee`:
   raise z first using these geometry-based heights, translate, then descend.
10. If the user message begins with a section titled **prior_simulation_judge_feedback**,
   read its **task_result** and **analysis** carefully and revise the plan so those failure
    modes are explicitly resolved in the new code (do not ignore them).
11. **Phased execution markers (MANDATORY):** split motion into clear robotic stages
    (e.g. approach / pregrasp, descend & grasp, lift & carry, place / release, restore).
    Before each stage's Python lines, insert **exactly one** header line on its own:
    `# === EVO_PHASE: <slug_en> | <short goal> ===`
    The `<short goal>` must be a **robot action outcome description**, not internal
    code/function logic wording. Bad: "compute beaker grasp"; Good: "move and prepare
    to grasp beaker".
    The goal text must match the real action semantics of APIs used in that stage.
    Example: if using `pick_and_place`, do NOT label goal as only "grasp object" unless
    the stage truly excludes placement/release APIs.
    Use **at least 2** such headers (typically 3–6). Slug: ASCII letters/digits/underscore.
    The first header should appear near the top of executable motion code. The last
    stage should represent the final intended motion outcome.
    If the user message has **prior_segment_judge_feedback**, fix the cited stage while
    keeping all markers and the overall structure.
"""

        user_prompt = f"""
Atomic task JSON:
{json.dumps(subtask, ensure_ascii=False, indent=2)}

Scene context JSON:
{json.dumps(scene_context, ensure_ascii=False, indent=2)}

Pre-registered runtime functions (signatures only — already built-in, JUST CALL THEM, never redefine):
```python
{atomic_api_signatures}
```

Code style requirements:
- Extract numeric object poses and grasp offsets from scene context at
  planning time, then emit inline numeric constants in the runtime code.
- Primary call pattern for pick/place style atomic tasks:
    1) Recover absolute grasp pose via `recover_grasp_pose_from_offset(...)`
       using the best-matching entry from `grasp_offset_candidates`.
    2) Build an `object_pose` dict `{{"pos": grasp_pos, "quat": grasp_quat}}`.
    3) Call the matching composite skill (e.g. `pick_and_place(object_pose=..., target_pose=..., ...)`).
- Fall back to `candidate_poses` / object centroid only when no grasp offset
  candidate matches the target object.
- Object matching hints: exact `name` match first, then `source_key` suffix
  match (e.g. "object/Beaker" -> "Beaker"). If multiple pose candidates exist
  for one object, prefer the `pose_name` whose wording best matches the
  atomic task description.
- **Lateral motion vs table clutter:** use `tabletop_clearance.safe_carry_end_effector_z_m`
  (pre-capped at `safe_carry_z_max_m`, 1.4 m) and each object's `geom_top_z_world`
  (MuJoCo geom tops — **not** `pos[2]`). Inline that threshold in `lift_height` /
  `move_to` z; carry height must meet clearance but **must not exceed 1.4 m** world z.
- **EVO_PHASE headers:** include `# === EVO_PHASE: ... | ... ===` lines as required in
  system rule 12 so the runtime can execute and judge **stage-by-stage**.
- For each EVO_PHASE, write the goal as an **action description** (what robot does /
  should achieve), not implementation wording like "compute ...".
- Keep EVO_PHASE goal semantically consistent with called API:
  `pick_and_place` means full pick+transport+place+release, `push` means contact+push,
  `pull` means contact/grasp+pull, `press` means directional press, `open/close` means
  articulated manipulation, `pour` means tilt-pour behavior.
- Keep trajectories intentionally slow: explicitly set `num_steps` / `steps`,
  usually in `100~200`; for phases interacting with objects (approach/contact/
  grasp/release/push/pull/press), prefer around `200~300`.

Return ONLY the runnable python block.
"""

        if extra_user_context and str(extra_user_context).strip():
            user_prompt = f"{str(extra_user_context).strip()}\n\n---\n\n{user_prompt}"

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                },
            ],
        )
        return _extract_code_block(resp.output_text or "")


class TaskSuccessJudgeAgent:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def judge(
        self,
        task_prompt: str,
        image_path: str = "logs/finished_view.png",
        wrist_image_path: str | None = None,
        json_out: Path | str | None = None,
    ) -> tuple[bool, str]:
        """Run vision-language success judge; write structured JSON to disk and return (success, json_text).

        When ``wrist_image_path`` is set, the model receives **two** images (front table, then wrist),
        matching the stage judge convention.
        """
        image_data_url = file_to_data_url(image_path)
        wrist_image_data_url = file_to_data_url(wrist_image_path) if wrist_image_path else None
        if wrist_image_data_url:
            system_prompt = """
You are a robotics evaluator.
You see TWO camera images from the **end** of the movements (same instant):
- front camera view
- wrist camera view
Decide whether the task has been successfully completed.

Procedures:
1. Define the task final state. Especially, the object pose should be the final pose.
2. Compare the final state with the image.

Output rules (STRICT):
Return ONE JSON object only (optionally wrapped in a ```json code fence). No other text.
Schema:
{
  "task_result": "SUCCESS" or "FAIL",
  "analysis": "if task_result is FAIL, explain why (grasp, distance, clutter, occlusion, wrong object, etc.)"
}
"""
        else:
            system_prompt = """
You are a strict robotics evaluator.
Given the task description and one final camera image, decide whether the task has been successfully completed.

Procedures:
1. Define the task final state. Especially, the object pose should be the final pose.
2. Compare the final state with the image.

Output rules (STRICT):
Return ONE JSON object only (optionally wrapped in a ```json code fence). No other text.
Schema:
{
  "task_result": "SUCCESS" or "FAIL",
  "analysis": "if task_result is FAIL, explain why (grasp, distance, clutter, occlusion, wrong object, etc.)"
}
"""
        user_prompt = f"""
Task description:
{task_prompt}

Is the task final state achieved?
"""
        content_items: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": image_data_url},
        ]
        if wrist_image_data_url:
            content_items.append({"type": "input_image", "image_url": wrist_image_data_url})
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": content_items,
                },
            ],
        )

        raw_text = (resp.output_text or "").strip()
        record = _parse_judge_output(raw_text, task_prompt=task_prompt, image_path=image_path)
        record["model"] = self.model
        record["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if wrist_image_path:
            record["wrist_image_path"] = str(Path(wrist_image_path).resolve())

        out_path = Path(json_out) if json_out else Path(image_path).parent / f"{Path(image_path).stem}_judge.json"
        if json_out is None and str(Path(image_path).parent) in ("", "."):
            out_path = DEFAULT_JUDGE_JSON_PATH
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        success = str(record.get("task_result", record.get("verdict", ""))).upper() == "SUCCESS"
        return success, json.dumps(record, ensure_ascii=False, indent=2)


class SegmentSuccessJudgeAgent:
    """
    stage success judge (mid-rollout segment judge): after each ``EVO_PHASE`` block,
    checks the table camera image against that stage's goal.
    """

    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model

    def judge_segment(
        self,
        *,
        phase_slug: str,
        phase_goal: str,
        atomic_task_summary: str,
        image_path: str,
        wrist_image_path: str | None = None,
        json_out: Path | str | None = None,
    ) -> tuple[bool, str]:
        """Return (success, json_text) for one motion stage."""
        image_data_url = file_to_data_url(image_path)
        wrist_image_data_url = file_to_data_url(wrist_image_path) if wrist_image_path else None
        system_prompt = """
You are a strict robotics **stage** evaluator (stage judge).
You see TWO camera images captured **immediately after** a motion stage finished:
- front camera view
- wrist camera view
Decide whether **this stage's stated goal** is satisfied (not whether the whole task is done).

Output rules (STRICT):
Return ONE JSON object only (optionally wrapped in a ```json code fence). No other text.
Schema:
{
  "task_result": "SUCCESS" or "FAIL",
  "analysis": "if task_result is FAIL, explain why for this stage only (approach height, alignment, grasp gap, collisions, wrong object, etc.)"
}
"""
        user_prompt = f"""
Overall atomic task (for context only; do not require full task completion here):
{atomic_task_summary}

**Current stage** slug: {phase_slug}
**Current stage goal** (what this stage alone should achieve):
{phase_goal}
"""
        content_items: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": image_data_url},
        ]
        if wrist_image_data_url:
            content_items.append({"type": "input_image", "image_url": wrist_image_data_url})
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": content_items,
                },
            ],
        )
        raw_text = (resp.output_text or "").strip()
        task_key = f"{phase_slug}: {phase_goal}"
        record = _parse_judge_output(raw_text, task_prompt=task_key, image_path=image_path)
        record["model"] = self.model
        record["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["phase_slug"] = phase_slug
        record["phase_goal"] = phase_goal
        if wrist_image_path:
            record["wrist_image_path"] = str(Path(wrist_image_path).resolve())

        out_path = (
            Path(json_out)
            if json_out
            else Path(image_path).parent / f"{Path(image_path).stem}_segment_judge.json"
        )
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        success = str(record.get("task_result", record.get("verdict", ""))).upper() == "SUCCESS"
        return success, json.dumps(record, ensure_ascii=False, indent=2)


class EvoMAAgentPipeline:
    def __init__(self, client: OpenAI | None = None, model: str = MODEL_NAME):
        self.client = client or _get_client()
        self.supervisor = TaskSupervisorAgent(self.client, model=model)
        self.atomic_skill = AtomicActionSkill(self.client, model=model)
        self.verifier = CodeVerifierAgent(self.client, model=model)
        self.pose_reconstructor = ScenePoseReconstructor(PRE_SCENE_PATH)

    @staticmethod
    def _plan_to_report_text(plan: dict[str, Any]) -> str:
        if str(plan.get("status", "")).lower() == "completed":
            return REPORT_DONE_TOKEN
        atomic_tasks = TaskSupervisorAgent._atomic_tasks_from_payload(plan)
        if not atomic_tasks:
            return "Atomic task 1: Review scene and complete task"

        lines = []
        for idx, sub in enumerate(atomic_tasks, start=1):
            desc = str(sub.get("description", "")).strip() or "undefined atomic task"
            lines.append(f"Atomic task {idx}: {desc}")
        return "\n".join(lines)

    def run_report_generation(self, task_prompt: str) -> str:
        _ensure_logs_dir()
        scene_context = self.pose_reconstructor.build_context(task_prompt)
        plan = self.supervisor.analyze_task(
            task_prompt=task_prompt,
            scene_context=scene_context,
        )
        PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        SCENE_CONTEXT_PATH.write_text(json.dumps(scene_context, ensure_ascii=False, indent=2), encoding="utf-8")

        report = self._plan_to_report_text(plan)
        Path("logs/report.txt").write_text(report, encoding="utf-8")
        return report

    def run_atomic_action_generation(self, failure_feedback: str | None = None) -> str | None:
        _ensure_logs_dir()

        if not PLAN_PATH.exists():
            report_content = Path("logs/report.txt").read_text(encoding="utf-8").strip() if Path("logs/report.txt").exists() else ""
            if _is_task_completed(report_content):
                return None
            # Fallback if run_report_generation was not called first.
            fallback_plan = {
                "status": "in_progress",
                "summary": "fallback plan from report",
                "atomic_tasks": [
                    {
                        "id": 1,
                        "description": report_content or "complete the task",
                        "primitive": "other",
                        "source_object": None,
                        "target_object": None,
                        "constraints": ["avoid collision"],
                        "done_criteria": "task done",
                    }
                ],
            }
            PLAN_PATH.write_text(json.dumps(fallback_plan, ensure_ascii=False, indent=2), encoding="utf-8")

        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if str(plan.get("status", "")).lower() == "completed":
            return None

        if SCENE_CONTEXT_PATH.exists():
            scene_context = json.loads(SCENE_CONTEXT_PATH.read_text(encoding="utf-8"))
        else:
            scene_context = self.pose_reconstructor.build_context(task_prompt=None)

        atomic_tasks = TaskSupervisorAgent._atomic_tasks_from_payload(plan)
        if not atomic_tasks:
            return None

        first_subtask = atomic_tasks[0]
        extra_ctx = failure_feedback
        if extra_ctx and str(extra_ctx).strip().startswith("{"):
            s = str(extra_ctx).strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and (
                    parsed.get("task_result") is not None or parsed.get("analysis") is not None
                ):
                    extra_ctx = (
                        format_judge_feedback_for_atomic_skill(s)
                        + "\n\n---\n\n### judge_record_json\n"
                        + json.dumps(parsed, ensure_ascii=False, indent=2)
                    )
            except json.JSONDecodeError:
                pass

        raw_code = self.atomic_skill.generate_atomic_actions(
            subtask=first_subtask,
            scene_context=scene_context,
            extra_user_context=extra_ctx,
        )
        RAW_CODE_PATH.write_text(raw_code, encoding="utf-8")

        checked_code = self.verifier.verify_and_repair(
            code=raw_code,
            subtask=first_subtask,
            scene_context=scene_context,
        )
        ATOMIC_CODE_PATH.write_text(checked_code, encoding="utf-8")
        return checked_code

    def run_once(self, task_prompt: str) -> tuple[str, str | None]:
        report = self.run_report_generation(task_prompt)
        if _is_task_completed(report):
            return report, None
        code = self.run_atomic_action_generation()
        return report, code



def atomic_action_generation():
    pipeline = EvoMAAgentPipeline()
    pipeline.run_atomic_action_generation()


__all__ = [
    "REPORT_DONE_TOKEN",
    "EVO_PHASE_LINE_RE",
    "count_evo_phase_headers",
    "parse_evo_phase_segments",
    "format_judge_feedback_for_atomic_skill",
    "format_segment_failure_for_atomic_skill",
    "TaskSupervisorAgent",
    "TaskSuccessJudgeAgent",
    "SegmentSuccessJudgeAgent",
    "AtomicActionSkill",
    "EvoMAAgentPipeline",
    "atomic_action_generation",
]
