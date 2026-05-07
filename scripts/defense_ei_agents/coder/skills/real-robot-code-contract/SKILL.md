---
name: real-robot-code-contract
description: Real UR7e runtime code contract for defense_ei_agents with Intel RealSense RGB feedback.
tags: [defense_ei_agents, atomic, codegen, runtime, ur7e, real-robot]
---

# Real-Robot Code Contract

The code is executed on `UR7eVectorController`. There is no MuJoCo scene and no
evobody dependency.

Primitive APIs:
1. `move_ee(dx, dy, dz, droll, dpitch, dyaw, velocity=0.04, acceleration=0.18, wait_after_arm_s=0.2)` - relative TCP delta, `dx/dy/dz` in millimeters, `droll/dpitch/dyaw` in radians. Do not pass `steps`; real hardware uses speed and acceleration.
2. `gripper_control(value, delay)` - Robotiq command, 0=open and 255=closed; delay in ms.
3. `ee_pose() -> pose_vec` - current TCP pose as `[x,y,z,rx,ry,rz]`.

Composite APIs:
- `pick_and_place(...)` and alias `pick_place(...)`
- `push(...)`
- `pull(...)`
- `press(...)`
- `open(...)`
- `close(...)`
- `pour(...)`
- `move_x(...)`, `move_y(...)`, `move_z(...)`
- `rotate_x(...)`, `rotate_y(...)`, `rotate_z(...)`
- `sleep(seconds)`

Generation rules:
- Output exactly one Python code block.
- Do not call simulation tools.
- Do not generate quaternion code. Avoid names or keys such as `quat`,
  `quaternion`, `wxyz`, or `xyzw`; use 6D Cartesian TCP pose vectors instead.
- Use millimeters for every `move_ee` translation. Examples: use `10` for
  10 mm, `50` for 5 cm, and `100` for 10 cm. Do not write `0.05` when you mean
  5 cm.
- Use radians for every `move_ee` rotation increment. Examples: `0.1745` is
  about 10 degrees and `1.5708` is about 90 degrees.
- Use `move_ee` for every incremental arm motion. Do not call `move_to`; it is
  not available in generated-code runtime API.
- If the atomic task names a primitive skill, follow `primitive-skill-contract`
  for the required phase order and motion decomposition.
- Do not invent exact object coordinates from RGB images alone.
- Use small relative motions, slow contact, and explicit gripper commands.
- Overall motion must be slow enough to protect real objects. Prefer many small
  `move_ee` increments and low `velocity`/`acceleration` over a single large
  move.
- Near-object interaction phases must be slower than transit phases: approach,
  descend, grasp, push/pull/press contact, placement, release, and retraction
  from a container should use smaller deltas, lower `velocity`/`acceleration`,
  and brief `sleep(...)` pauses.
- Add a short `sleep(...)` after every move command so each hardware motion has
  time to finish and settle before the next command.
- If the atomic task info includes a concrete 6D TCP pose vector, use it only as
  a reference with `ee_pose()` and guarded `move_ee` deltas. Do not call
  absolute-pose APIs from generated code.
- If feedback says the previous attempt failed, address the specific failure.

Coordinate frame for `move_ee`:

- Wrist image right is gripper +X, so `move_ee(+dx, 0, 0, 0, 0, 0)`
  moves toward the right side of the wrist image.
- Wrist image down is gripper +Y, so `move_ee(0, +dy, 0, 0, 0, 0)`
  moves toward the lower side of the wrist image.
- Wrist camera viewing direction is gripper +Z, so `move_ee(0, 0, +dz, 0, 0, 0)`
  moves along the wrist camera forward direction; use the opposite sign to move
  away from that direction.

Orientation reference for `move_ee`:

- `droll/dpitch/dyaw` are relative increments, not absolute target angles.
- `[0, 0, 0]` keeps the current gripper orientation.
- `[0, +1.5708, 0]` is the first reference increment for pitching a horizontal
  gripper toward a vertical-down grasp; use `[0, -1.5708, 0]` for the opposite
  pitch direction if the observed wrist image motion is reversed.
- `[0, -1.5708, 0]` is the first reference increment for pitching a vertical-down
  gripper back toward a horizontal gripper.
- `[0, 0, +1.5708]` rotates about the wrist image forward / gripper +Z axis.
- `[+1.5708, 0, 0]` rolls the gripper around the wrist image right / gripper +X
  axis.
