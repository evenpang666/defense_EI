---
name: primitive-skill-contract
description: Motion templates for defense_ei_agents primitive_skill code generation on a real UR7e.
tags: [defense_ei_agents, primitive_skill, pick_place, real-robot, codegen]
---

# Primitive Skill Contract

Use this contract when the atomic task info contains a `primitive`,
`primitive_skill`, or equivalent action label such as `pick_place`.

## Global Rules

- Implement the named primitive skill as a complete motion sequence, not as a
  single vague move.
- `move_ee(dx, dy, dz, droll, dpitch, dyaw, velocity=0.04, acceleration=0.18, wait_after_arm_s=0.2)`
  uses millimeters for `dx/dy/dz` and radians for `droll/dpitch/dyaw`. Do not
  pass `steps`; real hardware uses velocity and acceleration.
- Use the wrist-image/gripper frame for relative moves: wrist image right is
  gripper +X, wrist image down is gripper +Y, and the wrist image viewing
  direction is gripper +Z.
- Prefer vertical and horizontal Cartesian translations for pick/place style
  manipulation. Avoid diagonal shortcuts near objects, containers, rims, or
  clutter.
- All primitive skills must be object-safe and slow. Use conservative travel
  distances, split large moves into smaller segments, lower
  `velocity`/`acceleration`, and add short `sleep(...)` pauses before/after
  object contact.
- Add a short `sleep(...)` after every move command so the real robot has time
  to complete and settle.
- Interaction phases must be especially slow: final approach, descent to grasp,
  gripper close, initial lift, descent to target, release, and retraction from
  containers.
- Use `move_ee(...)` for all incremental arm motion. Do not call `move_to`;
  generated code cannot use absolute-pose APIs directly. With RGB-only
  observations, use conservative relative moves and make the intended phase
  explicit.
- Every stage must start with
  `# === DEFENSE_EI_PHASE: <slug> | <short goal> ===`.
- Keep the gripper state explicit before and after contact.

## `pick_place` / `pick_and_place`

`pick_place` is a composition of vertical and horizontal translations:

1. Move to a safe pose above the operated object.
2. Open the gripper.
3. Move vertically downward to the operated object grasp height slowly, using
   smaller deltas and lower `velocity`/`acceleration`.
4. Close the gripper on the operated object and pause briefly to let the grasp
   settle.
5. Move vertically upward to a safe carry height slowly at first, then continue
   to the final safe height.
6. Determine all objects between the operated object and the target position
   from the atomic task info and visual notes. The safe carry height must be
   higher than every intervening object's height. If exact heights are unknown,
   choose a conservative extra lift and state the reason in the phase goal.
7. Keep the gripper closed and translate horizontally to a pose above the target
   position. Do not lower during this transfer.
8. Move vertically downward to the target placement position slowly.
9. Open the gripper to release and pause briefly.
10. Retract vertically upward to a safe height slowly before any horizontal
    motion away.

Required phase order for generated code:

- `approach_source_above`
- `open_gripper`
- `descend_to_source`
- `grasp_source`
- `lift_to_safe_height`
- `transfer_above_target`
- `descend_to_target`
- `release_at_target`
- `retract_up`

For container targets such as beakers, cups, bins, or bowls:

- Transfer to above the opening/rim first.
- Lower slowly and vertically inside the opening.
- Release gently after the object is below the rim or clearly inside the target
  region.
- Retract vertically before any horizontal motion away from the container.

Do not skip `open_gripper` before descent or `lift_to_safe_height` before
horizontal transfer.
