#!/usr/bin/env python3
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except Exception:
    mujoco = None

try:
    mujoco.mj_loadPluginLibrary("./libmjlab.so.3.3.0")
except Exception:
    pass

def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    if quat.shape != (4,):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_assets(scene_json_path: Path) -> list[dict]:
    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    assets = scene.get("assets", [])
    if not isinstance(assets, list):
        return []
    return [a for a in assets if isinstance(a, dict)]


def load_scene(scene_json_path: Path) -> dict:
    return json.loads(scene_json_path.read_text(encoding="utf-8"))


def make_world_xml_for_ur5e(base_pos: np.ndarray, base_quat: np.ndarray, ur5e_model_xml: Path) -> str:
    bp = " ".join(f"{float(v):.10f}" for v in base_pos.tolist())
    bq = " ".join(f"{float(v):.10f}" for v in base_quat.tolist())
    model_file = ur5e_model_xml.resolve().as_posix()
    return f"""
<mujoco model="ur5e_fk_world">
  <asset>
    <model name="ur5e_model" file="{model_file}" content_type="text/xml"/>
  </asset>
  <worldbody>
    <body name="robot_base_world" pos="{bp}" quat="{bq}">
      <attach model="ur5e_model" body="base" prefix="ur:"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _find_body_id(model, candidates: list[str]) -> int:
    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return body_id
    return -1


def _find_site_id(model, candidates: list[str]) -> int:
    for name in candidates:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return site_id
    return -1


def compute_robot_frames(scene: dict, project_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    robot = scene.get("robot", {})
    if not isinstance(robot, dict):
        return frames

    base_pos = np.array(robot.get("base_pos", [0.0, 0.0, 0.824]), dtype=float)
    base_quat = normalize_quat_wxyz(np.array(robot.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float))
    frames["robot_base"] = (base_pos, quat_wxyz_to_rotmat(base_quat))

    if mujoco is None:
        return frames

    joint_targets_raw = robot.get("joint_targets", [])
    if not isinstance(joint_targets_raw, list) or len(joint_targets_raw) < 6:
        return frames

    joint_targets = np.array([float(v) for v in joint_targets_raw[:6]], dtype=float)
    ur5e_model_xml = project_root / "model" / "robot" / "ur5e_gripper.xml"
    if not ur5e_model_xml.exists():
        return frames
    xml_text = make_world_xml_for_ur5e(base_pos, base_quat, ur5e_model_xml)

    with tempfile.TemporaryDirectory(prefix="plot_scene_frames_") as td:
        xml_path = Path(td) / "ur5e_fk_world.xml"
        xml_path.write_text(xml_text, encoding="utf-8")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)

        nq = int(model.nq)
        if nq > 0:
            nfill = min(6, nq)
            data.qpos[:nfill] = joint_targets[:nfill]
        mujoco.mj_forward(model, data)

        wrist_id = _find_body_id(model, ["ur:wrist_3_link", "wrist_3_link"])
        if wrist_id >= 0:
            pos = np.array(data.xpos[wrist_id], dtype=float)
            rot = np.array(data.xmat[wrist_id], dtype=float).reshape(3, 3)
            frames["wrist_3"] = (pos, rot)

        site_id = _find_site_id(model, ["ur:attachment_site", "attachment_site"])
        if site_id >= 0:
            pos = np.array(data.site_xpos[site_id], dtype=float)
            mat = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
            frames["gripper_attachment"] = (pos, mat)

        pinch_id = _find_site_id(model, ["ur:2f85:pinch", "2f85:pinch", "pinch"])
        if pinch_id >= 0:
            pos = np.array(data.site_xpos[pinch_id], dtype=float)
            mat = np.array(data.site_xmat[pinch_id], dtype=float).reshape(3, 3)
            frames["gripper_pinch"] = (pos, mat)

    return frames


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = float(np.max(maxs - mins) * 0.6)
    if half_range < 0.05:
        half_range = 0.05

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot object coordinate frames from an EvoBody scene JSON."
    )
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=Path("chemistry.json"),
        help="Path to scene json (default: chemistry.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/chemistry_frames.png"),
        help="Output image path (default: logs/chemistry_frames.png)",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.05,
        help="Length of each local axis arrow in meters (default: 0.05)",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide object name labels",
    )
    parser.add_argument(
        "--show-robot-frames",
        action="store_true",
        help="Show robot base/wrist/gripper frames from scene robot config",
    )
    args = parser.parse_args()

    if not args.scene_json.exists():
        raise FileNotFoundError(f"Scene json not found: {args.scene_json}")

    scene = load_scene(args.scene_json)
    assets = scene.get("assets", [])
    if not isinstance(assets, list):
        assets = []

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Object Frames: {args.scene_json}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    origins: list[np.ndarray] = []
    colors = ["r", "g", "b"]  # x, y, z

    for idx, asset in enumerate(assets, start=1):
        key = str(asset.get("key", f"asset_{idx}"))
        name = key.split("/")[-1]

        pos_raw = asset.get("pos", [0.0, 0.0, 0.0])
        quat_raw = asset.get("quat", [1.0, 0.0, 0.0, 0.0])
        if not isinstance(pos_raw, list) or len(pos_raw) != 3:
            continue
        if not isinstance(quat_raw, list) or len(quat_raw) != 4:
            continue

        origin = np.array([float(v) for v in pos_raw], dtype=float)
        quat = np.array([float(v) for v in quat_raw], dtype=float)
        rot = quat_wxyz_to_rotmat(quat)
        axes_world = rot @ np.eye(3, dtype=float)

        origins.append(origin)
        ax.scatter(origin[0], origin[1], origin[2], c="k", s=16)

        if not args.hide_labels:
            ax.text(origin[0], origin[1], origin[2], f" {name}", fontsize=8)

        for axis_idx in range(3):
            direction = axes_world[:, axis_idx]
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                length=args.axis_length,
                normalize=True,
                color=colors[axis_idx],
                linewidth=1.5,
            )

    if args.show_robot_frames:
        project_root = Path(__file__).resolve().parents[1]
        robot_frames = compute_robot_frames(scene, project_root)
        # Use larger axis arrows for robot frames for readability.
        robot_axis_length = max(args.axis_length * 1.35, args.axis_length + 0.01)
        for name, (origin, rot) in robot_frames.items():
            origins.append(origin)
            ax.scatter(origin[0], origin[1], origin[2], c="m", s=28)
            if not args.hide_labels:
                ax.text(origin[0], origin[1], origin[2], f" [{name}]", fontsize=9, color="m")

            for axis_idx in range(3):
                direction = rot[:, axis_idx]
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    length=robot_axis_length,
                    normalize=True,
                    color=colors[axis_idx],
                    linewidth=2.2,
                    alpha=0.95,
                )

    points_for_bounds = np.array(origins, dtype=float) if origins else np.zeros((0, 3), dtype=float)
    set_equal_3d_axes(ax, points_for_bounds)
    ax.view_init(elev=24, azim=-56)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"Saved frame visualization to: {args.output}")
    print("Axis colors: X=red, Y=green, Z=blue")
    if args.show_robot_frames:
        print("Robot frames enabled: robot_base + (wrist/gripper if Mujoco FK available)")


if __name__ == "__main__":
    main()
