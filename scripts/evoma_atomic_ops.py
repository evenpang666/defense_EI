"""Atomic motion library for EvoMA runtime APIs.

This file is designed to be executed in the same restricted environment as
execute_code_gradio.py, where the available runtime APIs are:
move_to, move_ee, gripper_control, ee_pose, np.

All functions defined here are pre-registered as runtime builtins by
`execute_code_runtime.execute_code_with_recording`. Generated code is expected
to CALL these functions directly rather than re-implement them.
"""


def get_object_abs_pose(object_poses, object_name):
    """Look up absolute world pose of an object by name from a poses list.

    Args:
        object_poses: List of dicts, each with at least {"name"|"source_key", "pos", "quat"}.
        object_name: Object short name or source key suffix (e.g. "Beaker").

    Returns:
        Tuple (pos, quat) in world frame. Falls back to ([0,0,0],[1,0,0,0])
        when no match is found.
    """
    if not isinstance(object_poses, list):
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    target = str(object_name or "").strip()
    target_suffix = target.split("/")[-1]
    for obj in object_poses:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip()
        key = str(obj.get("source_key", obj.get("key", ""))).strip()
        if name == target or key == target or key.endswith("/" + target_suffix) or name == target_suffix:
            pos = obj.get("pos", [0.0, 0.0, 0.0])
            quat = obj.get("quat", [1.0, 0.0, 0.0, 0.0])
            pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            quat = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            return pos, quat
    return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]


def _rpy_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    r = float(roll_deg) * np.pi / 180.0
    p = float(pitch_deg) * np.pi / 180.0
    y = float(yaw_deg) * np.pi / 180.0
    cr = np.cos(r * 0.5); sr = np.sin(r * 0.5)
    cp = np.cos(p * 0.5); sp = np.sin(p * 0.5)
    cy = np.cos(y * 0.5); sy = np.sin(y * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    yq = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return [float(w), float(x), float(yq), float(z)]


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _rotate_vec_by_quat(q, v):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # q * (0, v) * q_conj
    tw = -x * vx - y * vy - z * vz
    tx = w * vx + y * vz - z * vy
    ty = w * vy - x * vz + z * vx
    tz = w * vz + x * vy - y * vx
    rx = tw * (-x) + tx * w + ty * (-z) - tz * (-y)
    ry = tw * (-y) - tx * (-z) + ty * w + tz * (-x)
    rz = tw * (-z) + tx * (-y) - ty * (-x) + tz * w
    return [float(rx), float(ry), float(rz)]


def _quat_to_rotmat_wxyz(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-8:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _rotmat_to_quat_wxyz(rot):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    R = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    q = q / n
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _rotate_vec_about_axis(vec, axis_unit, angle_rad):
    """Rodrigues rotation for vector around a unit axis."""
    v = np.asarray(vec, dtype=float).reshape(3)
    k = np.asarray(axis_unit, dtype=float).reshape(3)
    kn = float(np.linalg.norm(k))
    if kn < 1e-8:
        return v.copy()
    k = k / kn
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg):
    """Recover absolute world grasp pose from relative offset.

    Args:
        object_pos_xyz: Object world position [x, y, z].
        object_quat_wxyz: Object world quaternion [w, x, y, z].
        offset_pos_xyz: Grasp offset in object-local frame [dx, dy, dz].
        offset_rpy_deg: Grasp offset rotation (roll, pitch, yaw) in degrees.

    Returns:
        Dict with keys:
          - "grasp_pos_world_xyz": [x, y, z]
          - "grasp_quat_world_wxyz": [w, x, y, z]
    """
    obj_pos = [float(object_pos_xyz[0]), float(object_pos_xyz[1]), float(object_pos_xyz[2])]
    obj_quat = [float(object_quat_wxyz[0]), float(object_quat_wxyz[1]),
                float(object_quat_wxyz[2]), float(object_quat_wxyz[3])]
    n = (obj_quat[0] ** 2 + obj_quat[1] ** 2 + obj_quat[2] ** 2 + obj_quat[3] ** 2) ** 0.5
    if n < 1e-8:
        obj_quat = [1.0, 0.0, 0.0, 0.0]
    else:
        obj_quat = [v / n for v in obj_quat]

    dpos_local = [float(offset_pos_xyz[0]), float(offset_pos_xyz[1]), float(offset_pos_xyz[2])]
    dpos_world = _rotate_vec_by_quat(obj_quat, dpos_local)
    grasp_pos = [
        obj_pos[0] + dpos_world[0],
        obj_pos[1] + dpos_world[1],
        obj_pos[2] + dpos_world[2],
    ]
    dq = _rpy_deg_to_quat_wxyz(offset_rpy_deg[0], offset_rpy_deg[1], offset_rpy_deg[2])
    grasp_quat = _quat_mul_wxyz(obj_quat, dq)
    return {
        "grasp_pos_world_xyz": grasp_pos,
        "grasp_quat_world_wxyz": grasp_quat,
    }


def _extract_pose(pose, fallback_pos, fallback_quat):
    """Parse an input pose into (pos, quat).

    Args:
        pose: Optional pose in dict form ({"pos"/"position", "quat"/"orientation"})
            or sequence form [x, y, z] / [x, y, z, qw, qx, qy, qz].
        fallback_pos: Default xyz used when pose is missing/incomplete.
        fallback_quat: Default quaternion (qw, qx, qy, qz) used when pose is
            missing/incomplete.

    Returns:
        Tuple (pos, quat), where pos is [x, y, z] and quat is [qw, qx, qy, qz].
    """
    pos = [float(fallback_pos[0]), float(fallback_pos[1]), float(fallback_pos[2])]
    quat = [float(fallback_quat[0]), float(fallback_quat[1]), float(fallback_quat[2]), float(fallback_quat[3])]

    if pose is None:
        return pos, quat

    # Dict-like pose: {"pos": [...], "quat": [...]} or aliases.
    try:
        pose_pos = pose.get("pos", None)
        if pose_pos is None:
            pose_pos = pose.get("position", None)
        if pose_pos is not None and len(pose_pos) >= 3:
            pos = [float(pose_pos[0]), float(pose_pos[1]), float(pose_pos[2])]

        pose_quat = pose.get("quat", None)
        if pose_quat is None:
            pose_quat = pose.get("orientation", None)
        if pose_quat is not None and len(pose_quat) >= 4:
            quat = [float(pose_quat[0]), float(pose_quat[1]), float(pose_quat[2]), float(pose_quat[3])]
        return pos, quat
    except:
        pass

    # Sequence-like pose: [x, y, z] or [x, y, z, qw, qx, qy, qz].
    try:
        if len(pose) >= 3:
            pos = [float(pose[0]), float(pose[1]), float(pose[2])]
        if len(pose) >= 7:
            quat = [float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])]
    except:
        pass

    return pos, quat


def _target_from_pose_or_offset(
    target_pose,
    base_pos,
    base_quat,
    direction_x,
    direction_y,
    direction_z,
    _extract_pose_fn=_extract_pose,
):
    """Resolve destination pose from target_pose, or from directional offsets.

    Args:
        target_pose: Absolute destination pose. If provided, offsets are ignored.
        base_pos: Base xyz used when computing offset destination.
        base_quat: Base quaternion copied to destination when using offsets.
        direction_x: Offset along x axis.
        direction_y: Offset along y axis.
        direction_z: Offset along z axis.
        _extract_pose_fn: Pose parser callback.

    Returns:
        Tuple (dst_pos, dst_quat).
    """
    if target_pose is not None:
        return _extract_pose_fn(target_pose, base_pos, base_quat)

    return (
        [
            float(base_pos[0] + direction_x),
            float(base_pos[1] + direction_y),
            float(base_pos[2] + direction_z),
        ],
        [float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])],
    )


def _move_arc(
    start_pos,
    end_pos,
    quat,
    arc_height=0.06,
    steps=60,
    _move_to=move_to,
):
    """Move end-effector along a smooth parabolic arc between two points.

    Args:
        start_pos: Arc start xyz.
        end_pos: Arc end xyz.
        quat: Quaternion kept during arc motion.
        arc_height: Peak height offset of the arc.
        steps: Number of interpolation segments.
        _move_to: Runtime absolute motion API.
    """
    num = max(2, int(steps))
    for i in range(1, num + 1):
        t = float(i) / float(num)
        x = (1.0 - t) * float(start_pos[0]) + t * float(end_pos[0])
        y = (1.0 - t) * float(start_pos[1]) + t * float(end_pos[1])
        z_linear = (1.0 - t) * float(start_pos[2]) + t * float(end_pos[2])
        z = z_linear + 4.0 * float(arc_height) * t * (1.0 - t)
        _move_to(pos=[x, y, z], quat=quat, num_steps=2)


def move_x(distance=0.05, steps=120, _move_ee=move_ee):
    """Translate end-effector along x axis.

    Args:
        distance: Translation distance on x (meters).
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=float(distance), dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, steps=int(steps))


def move_y(distance=0.05, steps=120, _move_ee=move_ee):
    """Translate end-effector along y axis.

    Args:
        distance: Translation distance on y (meters).
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=0.0, dy=float(distance), dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, steps=int(steps))


def move_z(distance=0.05, steps=120, _move_ee=move_ee):
    """Translate end-effector along z axis.

    Args:
        distance: Translation distance on z (meters).
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=0.0, dy=0.0, dz=float(distance), droll=0.0, dpitch=0.0, dyaw=0.0, steps=int(steps))


def rotate_x(angle_deg=10.0, steps=120, _move_ee=move_ee):
    """Rotate end-effector around x axis (roll).

    Args:
        angle_deg: Rotation angle in degrees.
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=0.0, dy=0.0, dz=0.0, droll=float(angle_deg), dpitch=0.0, dyaw=0.0, steps=int(steps))


def rotate_y(angle_deg=10.0, steps=120, _move_ee=move_ee):
    """Rotate end-effector around y axis (pitch).

    Args:
        angle_deg: Rotation angle in degrees.
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=float(angle_deg), dyaw=0.0, steps=int(steps))


def rotate_z(angle_deg=10.0, steps=120, _move_ee=move_ee):
    """Rotate end-effector around z axis (yaw).

    Args:
        angle_deg: Rotation angle in degrees.
        steps: Interpolation steps for smoothness.
        _move_ee: Runtime relative motion API.
    """
    _move_ee(dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=float(angle_deg), steps=int(steps))


def pick_and_place(
    object_pose=None,
    target_pose=None,
    direction_x=0.0,
    direction_y=0.0,
    direction_z=0.0,
    approach_height=0.08,
    lift_height=0.10,
    grasp_value=255,
    release_value=0,
    move_steps=120,
    grip_delay=80,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
    _target_from_pose_or_offset_fn=_target_from_pose_or_offset,
):
    """Pick-and-place(up and down): world-frame vertical + horizontal translation only (no rotation).

    End-effector orientation is fixed to the grasp quaternion from ``object_pose``
    for the entire macro. ``target_pose`` contributes **position (xyz) only**;
    any orientation in ``target_pose`` is ignored for motion. Segments are
    strictly vertical then horizontal (constant carry z) then vertical again.

    Sequence: open gripper → hover above object → descend to grasp pose → close
    → lift to safe height → translate in xy at carry z → descend to place xyz
    → open → lift to safe height.

    Args:
        object_pose: Source pose of manipulated object/contact point (pos+quat).
        target_pose: Destination **position**; orientation ignored. If None, use
            direction offsets from source position.
        direction_x: Destination offset on x when target_pose is None.
        direction_y: Destination offset on y when target_pose is None.
        direction_z: Destination offset on z when target_pose is None.
        approach_height: Vertical clearance above object/target for approach / retreat.
        lift_height: Z lift above source grasp height while carrying (safe transit height).
        grasp_value: Gripper close value (typically 255).
        release_value: Gripper open value (typically 0).
        move_steps: Motion interpolation steps.
        grip_delay: Gripper settling delay.
        _ee_pose/_move_to/_gripper_control: Runtime APIs.
        _extract_pose_fn/_target_from_pose_or_offset_fn: Helper callbacks.
    """
    cur_pos, cur_quat = _ee_pose()
    src_pos, src_quat = _extract_pose_fn(object_pose, cur_pos, cur_quat)
    dst_pos, _dst_quat_ignored = _target_from_pose_or_offset_fn(
        target_pose,
        src_pos,
        src_quat,
        direction_x,
        direction_y,
        direction_z,
    )

    carry_z = src_pos[2] + float(lift_height)
    retreat_z = max(carry_z, dst_pos[2] + float(approach_height))

    _gripper_control(float(release_value), delay=int(grip_delay))
    _move_to(pos=[src_pos[0], src_pos[1], src_pos[2] + float(approach_height)], quat=src_quat, num_steps=int(move_steps))
    _move_to(pos=src_pos, quat=src_quat, num_steps=int(move_steps))
    _gripper_control(float(grasp_value), delay=int(grip_delay))
    _move_to(pos=[src_pos[0], src_pos[1], carry_z], quat=src_quat, num_steps=int(move_steps))

    _move_to(pos=[dst_pos[0], dst_pos[1], carry_z], quat=src_quat, num_steps=int(move_steps))
    _move_to(pos=dst_pos, quat=src_quat, num_steps=int(move_steps))
    _gripper_control(float(release_value), delay=int(grip_delay))
    _move_to(pos=[dst_pos[0], dst_pos[1], retreat_z], quat=src_quat, num_steps=int(move_steps))


def push(
    target_pose=None,
    object_pose=None,
    push_distance=0.05,
    approach_height=0.05,
    grasp_value=255,
    move_steps=100,
    grip_delay=60,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _move_ee=move_ee,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
):
    """Push: move to target pose, close gripper, move along +EE z in world frame.

    The post-close translation is ``push_distance`` meters along the end-effector
    +z axis at contact time (tool frame), mapped to world deltas for ``move_ee``.

    Args:
        target_pose: Goal pose dict ``{"pos", "quat"}`` (preferred).
        object_pose: Used as target pose when ``target_pose`` is None.
        push_distance: Magnitude of motion along +tool z after closing (meters).
        approach_height: Hover this far above target z before descending (0 to skip).
        grasp_value: Gripper close command.
        move_steps: Interpolation steps for Cartesian moves.
        grip_delay: Gripper command dwell time.
        _ee_pose/_move_to/_move_ee/_gripper_control/_extract_pose_fn: Runtime APIs.
    """
    pose = target_pose if target_pose is not None else object_pose
    cur_pos, cur_quat = _ee_pose()
    tgt_pos, tgt_quat = _extract_pose_fn(pose, cur_pos, cur_quat)

    if float(approach_height) > 1e-9:
        _move_to(
            pos=[tgt_pos[0], tgt_pos[1], tgt_pos[2] + float(approach_height)],
            quat=tgt_quat,
            num_steps=int(move_steps),
        )
    _move_to(pos=tgt_pos, quat=tgt_quat, num_steps=int(move_steps))
    _gripper_control(float(grasp_value), delay=int(grip_delay))

    _, cur_quat_grasp = _ee_pose()
    local_push = [0.0, 0.0, abs(float(push_distance))]
    dworld = _rotate_vec_by_quat(cur_quat_grasp, local_push)
    _move_ee(
        dx=float(dworld[0]),
        dy=float(dworld[1]),
        dz=float(dworld[2]),
        droll=0.0,
        dpitch=0.0,
        dyaw=0.0,
        steps=int(move_steps),
    )


def pull(
    target_pose=None,
    object_pose=None,
    pull_distance=0.05,
    approach_height=0.05,
    grasp_value=255,
    release_value=0,
    move_steps=100,
    grip_delay=60,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _move_ee=move_ee,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
):
    """Pull: move to target pose, close gripper, move along -EE z in world frame, open.

    The post-grasp translation is ``pull_distance`` meters opposite to the end-effector
    +z axis at grasp time (tool frame), mapped to world deltas for ``move_ee``.

    Args:
        target_pose: Goal pose dict ``{"pos", "quat"}`` (preferred).
        object_pose: Used as target pose when ``target_pose`` is None.
        pull_distance: Magnitude of motion along -tool z after closing (meters).
        approach_height: Hover this far above target z before descending (0 to skip).
        grasp_value: Gripper close command.
        release_value: Gripper open command.
        move_steps: Interpolation steps for Cartesian moves.
        grip_delay: Gripper command dwell time.
        _ee_pose/_move_to/_move_ee/_gripper_control/_extract_pose_fn: Runtime APIs.
    """
    pose = target_pose if target_pose is not None else object_pose
    cur_pos, cur_quat = _ee_pose()
    tgt_pos, tgt_quat = _extract_pose_fn(pose, cur_pos, cur_quat)

    if float(approach_height) > 1e-9:
        _move_to(
            pos=[tgt_pos[0], tgt_pos[1], tgt_pos[2] + float(approach_height)],
            quat=tgt_quat,
            num_steps=int(move_steps),
        )
    _move_to(pos=tgt_pos, quat=tgt_quat, num_steps=int(move_steps))
    _gripper_control(float(grasp_value), delay=int(grip_delay))

    _, cur_quat_grasp = _ee_pose()
    local_pull = [0.0, 0.0, -abs(float(pull_distance))]
    dworld = _rotate_vec_by_quat(cur_quat_grasp, local_pull)
    _move_ee(
        dx=float(dworld[0]),
        dy=float(dworld[1]),
        dz=float(dworld[2]),
        droll=0.0,
        dpitch=0.0,
        dyaw=0.0,
        steps=int(move_steps),
    )
    _gripper_control(float(release_value), delay=int(grip_delay))


def press(
    object_pose=None,
    direction_x=0.0,
    direction_y=0.0,
    direction_z=-0.03,
    grasp_value=255,
    move_steps=90,
    grip_delay=60,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _move_ee=move_ee,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
):
    """Press: close gripper, then move end-effector down (relative delta).

    End-effector should already be at the contact height before calling; ``object_pose``
    is accepted for API compatibility with generated code but is not used here.

    Args:
        object_pose: Unused; callers may pass for consistency with other primitives.
        direction_x: Relative displacement on x after closing (meters).
        direction_y: Relative displacement on y after closing (meters).
        direction_z: Relative displacement on z after closing (meters; negative = down).
        grasp_value: Gripper close value.
        move_steps: Steps for the relative move.
        grip_delay: Gripper settling delay after close.
        _ee_pose/_move_to/_move_ee/_gripper_control/_extract_pose_fn: Runtime APIs (unused hooks kept for injection).
    """
    _gripper_control(float(grasp_value), delay=int(grip_delay))
    _move_ee(
        dx=float(direction_x),
        dy=float(direction_y),
        dz=float(direction_z),
        droll=0.0,
        dpitch=0.0,
        dyaw=0.0,
        steps=int(move_steps),
    )


def _resolve_arc_plane_normal(arc_plane, fallback_normal):
    """Resolve plane spec to a unit normal vector."""
    if isinstance(arc_plane, (list, tuple)) and len(arc_plane) >= 3:
        n = np.asarray([float(arc_plane[0]), float(arc_plane[1]), float(arc_plane[2])], dtype=float)
    elif isinstance(arc_plane, str):
        key = arc_plane.strip().lower()
        mapping = {
            "xy": [0.0, 0.0, 1.0],
            "xoy": [0.0, 0.0, 1.0],
            "yz": [1.0, 0.0, 0.0],
            "yoz": [1.0, 0.0, 0.0],
            "zx": [0.0, 1.0, 0.0],
            "xz": [0.0, 1.0, 0.0],
            "zox": [0.0, 1.0, 0.0],
        }
        n = np.asarray(mapping.get(key, fallback_normal), dtype=float)
    else:
        n = np.asarray(fallback_normal, dtype=float)
    n = n.reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-8:
        return np.asarray([0.0, 1.0, 0.0], dtype=float)
    return n / nn


def _orthonormal_tangent_frame(z_axis, plane_normal):
    """Build a stable frame from z-axis and arc plane normal."""
    z = np.asarray(z_axis, dtype=float).reshape(3)
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    zn = float(np.linalg.norm(z))
    if zn < 1e-8:
        z = np.asarray([0.0, 0.0, 1.0], dtype=float)
    else:
        z = z / zn

    x = np.cross(n, z)
    xn = float(np.linalg.norm(x))
    if xn < 1e-8:
        seed = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(seed, z))) > 0.95:
            seed = np.asarray([0.0, 1.0, 0.0], dtype=float)
        x = np.cross(seed, z)
        xn = float(np.linalg.norm(x))
    x = x / max(1e-8, xn)
    y = np.cross(z, x)
    y = y / max(1e-8, float(np.linalg.norm(y)))
    return np.stack([x, y, z], axis=1)


def _run_arc_motion(
    start_pos,
    start_quat,
    target_quat,
    arc_center,
    arc_plane,
    arc_angle_deg,
    move_steps,
    move_along_negative_ee_z,
    _move_to,
):
    """Arc interpolation with EE z-axis aligned to trajectory tangent."""
    start_pos_np = np.asarray(start_pos, dtype=float).reshape(3)
    center = np.asarray(arc_center, dtype=float).reshape(3)
    radius_vec = start_pos_np - center
    radius = float(np.linalg.norm(radius_vec))
    if radius < 1e-8:
        return

    target_rot = _quat_to_rotmat_wxyz(target_quat)
    start_rot = _quat_to_rotmat_wxyz(start_quat)
    plane_normal = _resolve_arc_plane_normal(arc_plane, target_rot[:, 1])

    # Keep radius in the specified plane.
    radius_vec = radius_vec - plane_normal * float(np.dot(radius_vec, plane_normal))
    radius = float(np.linalg.norm(radius_vec))
    if radius < 1e-8:
        return
    radius_vec = radius_vec / radius

    total_steps = max(2, int(move_steps))
    theta_total = float(arc_angle_deg) * np.pi / 180.0
    desired_dir0 = -start_rot[:, 2] if bool(move_along_negative_ee_z) else start_rot[:, 2]
    tangent0 = np.cross(plane_normal, radius_vec)
    tangent0 = tangent0 / max(1e-8, float(np.linalg.norm(tangent0)))
    direction_sign = 1.0 if float(np.dot(tangent0, desired_dir0)) >= 0.0 else -1.0

    z_t0 = -direction_sign * tangent0 if bool(move_along_negative_ee_z) else direction_sign * tangent0
    rot_tan0 = _orthonormal_tangent_frame(z_t0, plane_normal)
    rot_offset = rot_tan0.T @ target_rot

    for i in range(1, total_steps + 1):
        t = float(i) / float(total_steps)
        theta = theta_total * t
        radial = _rotate_vec_about_axis(radius_vec, plane_normal, direction_sign * theta)
        radial = radial / max(1e-8, float(np.linalg.norm(radial)))
        pos_t = center + radial * radius

        tangent_t = np.cross(plane_normal, radial)
        tangent_t = tangent_t / max(1e-8, float(np.linalg.norm(tangent_t)))
        tangent_t = tangent_t * direction_sign
        z_axis_t = -tangent_t if bool(move_along_negative_ee_z) else tangent_t
        rot_tan = _orthonormal_tangent_frame(z_axis_t, plane_normal)
        quat_t = _rotmat_to_quat_wxyz(rot_tan @ rot_offset)
        _move_to(
            pos=[float(pos_t[0]), float(pos_t[1]), float(pos_t[2])],
            quat=quat_t,
            num_steps=2,
        )


def open(
    grasp_pose=None,
    arc_center=None,
    arc_plane="zox",
    arc_angle_deg=90.0,
    rotation_radius=0.08,
    approach_distance=0.05,
    retreat_distance=0.02,
    grasp_value=255,
    release_value=0,
    move_steps=100,
    grip_delay=80,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
):
    """Open articulated handle by approach, pull-back, then arc motion.

    Inputs:
        grasp_pose: Target grasp pose dict/list.
        arc_center: Arc center xyz. If None, fallback to ``rotation_radius``.
        arc_plane: Arc plane ("xy"/"yz"/"zx"/"zox") or a normal vector.
        arc_angle_deg: Arc sweep angle in degrees (default 90).
    """
    cur_pos, cur_quat = _ee_pose()
    gpos, gquat = _extract_pose_fn(grasp_pose, cur_pos, cur_quat)
    base_rot = _quat_to_rotmat_wxyz(gquat)
    z0 = base_rot[:, 2]
    gpos_np = np.asarray(gpos, dtype=float).reshape(3)
    near_pos = gpos_np - z0 * float(approach_distance)
    _move_to(pos=[float(near_pos[0]), float(near_pos[1]), float(near_pos[2])], quat=gquat, num_steps=max(2, int(move_steps) // 2))
    _move_to(pos=gpos, quat=gquat, num_steps=int(move_steps))
    _gripper_control(float(grasp_value), delay=int(grip_delay))
    retreat_pos = gpos_np - z0 * float(retreat_distance)
    _move_to(
        pos=[float(retreat_pos[0]), float(retreat_pos[1]), float(retreat_pos[2])],
        quat=gquat,
        num_steps=max(2, int(move_steps) // 3),
    )

    start_pos = np.asarray(retreat_pos, dtype=float).reshape(3)
    center = arc_center
    if center is None:
        radius = max(1e-4, abs(float(rotation_radius)))
        center = start_pos + z0 * radius

    _run_arc_motion(
        start_pos=start_pos,
        start_quat=gquat,
        target_quat=gquat,
        arc_center=center,
        arc_plane=arc_plane,
        arc_angle_deg=arc_angle_deg,
        move_steps=move_steps,
        move_along_negative_ee_z=True,
        _move_to=_move_to,
    )
    _gripper_control(float(release_value), delay=int(grip_delay))


def close(
    *args,
    grasp_pose=None,
    arc_center=None,
    arc_plane="zox",
    arc_angle_deg=45.0,
    rotation_radius=0.08,
    approach_distance=0.05,
    grasp_value=255,
    move_steps=100,
    grip_delay=80,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
    **kwargs,
):
    """Close articulated handle by approach, then arc motion.

    Inputs:
        grasp_pose: Target grasp pose dict/list.
        arc_center: Arc center xyz. If None, fallback to ``rotation_radius``.
        arc_plane: Arc plane ("xy"/"yz"/"zx"/"zox") or a normal vector.
        arc_angle_deg: Arc sweep angle in degrees (default 45).
    """
    # Backward compatibility:
    # - old style positional first arg (object name / object_pose) is accepted
    # - legacy kwargs like articulation/motion/target_pose/object_pose are ignored
    #   unless they provide a usable fallback grasp pose.
    if grasp_pose is None:
        if "grasp_pose" in kwargs and kwargs.get("grasp_pose") is not None:
            grasp_pose = kwargs.get("grasp_pose")
        elif "object_pose" in kwargs and kwargs.get("object_pose") is not None:
            grasp_pose = kwargs.get("object_pose")
        elif len(args) >= 1 and isinstance(args[0], (dict, list, tuple)):
            grasp_pose = args[0]

    if "rotation_radius" in kwargs:
        try:
            rotation_radius = float(kwargs.get("rotation_radius"))
        except Exception:
            pass
    if "rotation_angle_deg" in kwargs:
        try:
            arc_angle_deg = float(kwargs.get("rotation_angle_deg"))
        except Exception:
            pass
    if "arc_angle_deg" in kwargs:
        try:
            arc_angle_deg = float(kwargs.get("arc_angle_deg"))
        except Exception:
            pass
    if "arc_center" in kwargs and kwargs.get("arc_center") is not None:
        arc_center = kwargs.get("arc_center")
    if "arc_plane" in kwargs and kwargs.get("arc_plane") is not None:
        arc_plane = kwargs.get("arc_plane")

    cur_pos, cur_quat = _ee_pose()
    gpos, gquat = _extract_pose_fn(grasp_pose, cur_pos, cur_quat)
    base_rot = _quat_to_rotmat_wxyz(gquat)
    z0 = base_rot[:, 2]
    gpos_np = np.asarray(gpos, dtype=float).reshape(3)
    near_pos = gpos_np - z0 * float(approach_distance)
    _move_to(pos=[float(near_pos[0]), float(near_pos[1]), float(near_pos[2])], quat=gquat, num_steps=max(2, int(move_steps) // 2))
    _gripper_control(float(grasp_value), delay=int(grip_delay))
    start_pos = np.asarray(near_pos, dtype=float).reshape(3)
    center = arc_center
    if center is None:
        radius = max(1e-4, abs(float(rotation_radius)))
        center = start_pos + z0 * radius

    _run_arc_motion(
        start_pos=start_pos,
        start_quat=gquat,
        target_quat=gquat,
        arc_center=center,
        arc_plane=arc_plane,
        arc_angle_deg=arc_angle_deg,
        move_steps=move_steps,
        move_along_negative_ee_z=False,
        _move_to=_move_to,
    )


def pour(
    object_pose=None,
    target_pose=None,
    direction_x=0.08,
    direction_y=0.0,
    direction_z=0.03,
    rot_x=0.0,
    rot_y=60.0,
    rot_z=0.0,
    approach_height=0.08,
    lift_height=0.12,
    grasp_value=255,
    release_value=0,
    move_steps=120,
    grip_delay=80,
    _ee_pose=ee_pose,
    _move_to=move_to,
    _move_ee=move_ee,
    _gripper_control=gripper_control,
    _extract_pose_fn=_extract_pose,
    _target_from_pose_or_offset_fn=_target_from_pose_or_offset,
):
    """Pour action: move -> grasp -> move -> rotate -> release.

    Args:
        object_pose: Source pose to pick/pour from.
        target_pose: Pour position/orientation. If None, use direction offsets.
        direction_x: Destination offset on x when target_pose is None.
        direction_y: Destination offset on y when target_pose is None.
        direction_z: Destination offset on z when target_pose is None.
        rot_x: Roll angle for pouring (degrees).
        rot_y: Pitch angle for pouring (degrees).
        rot_z: Yaw angle for pouring (degrees).
        approach_height: Safety approach height.
        lift_height: Lift height after grasping object.
        grasp_value: Gripper close value.
        release_value: Gripper open value.
        move_steps: Motion interpolation steps.
        grip_delay: Gripper settling delay.
        _ee_pose/_move_to/_move_ee/_gripper_control: Runtime APIs.
        _extract_pose_fn/_target_from_pose_or_offset_fn: Helper callbacks.
    """
    cur_pos, cur_quat = _ee_pose()
    src_pos, src_quat = _extract_pose_fn(object_pose, cur_pos, cur_quat)
    dst_pos, dst_quat = _target_from_pose_or_offset_fn(
        target_pose,
        src_pos,
        src_quat,
        direction_x,
        direction_y,
        direction_z,
    )

    _move_to(pos=[src_pos[0], src_pos[1], src_pos[2] + float(approach_height)], quat=src_quat, num_steps=int(move_steps))
    _move_to(pos=src_pos, quat=src_quat, num_steps=int(move_steps))
    _gripper_control(float(grasp_value), delay=int(grip_delay))
    _move_to(pos=[src_pos[0], src_pos[1], src_pos[2] + float(lift_height)], quat=src_quat, num_steps=int(move_steps))

    _move_to(pos=[dst_pos[0], dst_pos[1], dst_pos[2] + float(approach_height)], quat=dst_quat, num_steps=int(move_steps))
    _move_to(pos=dst_pos, quat=dst_quat, num_steps=int(move_steps))
    _move_ee(dx=0.0, dy=0.0, dz=0.0, droll=float(rot_x), dpitch=float(rot_y), dyaw=float(rot_z), steps=int(move_steps))
    _gripper_control(float(release_value), delay=int(grip_delay))


__all__ = [
    "move_x",
    "move_y",
    "move_z",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "pick_and_place",
    "push",
    "pull",
    "press",
    "open",
    "close",
    "pour",
    "get_object_abs_pose",
    "recover_grasp_pose_from_offset",
]
