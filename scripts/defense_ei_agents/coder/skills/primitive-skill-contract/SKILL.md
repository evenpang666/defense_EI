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
- Primitive names are never callable runtime APIs. Do not call `pick_place`,
  `pick_and_place`, `push`, `pull`, `press`, `open`, `close`, or `pour`; use
  the pseudocode templates below and emit real `move_ee`, `gripper_control`,
  and `sleep` calls.

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

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_source_above | align above source using small wrist-frame corrections ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: open_gripper | prepare for grasp ===
gripper_control(0, 120); sleep(...)
# === DEFENSE_EI_PHASE: descend_to_source | slow vertical descent ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: grasp_source | close on object and settle ===
gripper_control(255, 250); sleep(...)
# === DEFENSE_EI_PHASE: lift_to_safe_height | lift before horizontal transfer ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: transfer_above_target | horizontal transfer at safe height ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: descend_to_target | slow vertical placement ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: release_at_target | release object ===
gripper_control(0, 180); sleep(...)
# === DEFENSE_EI_PHASE: retract_up | retreat vertically ===
move_ee(...); sleep(...)
```

## `push`

Push is a contact translation using a closed or partially closed gripper as a
pusher. Required phase order:

- `approach_push_start`
- `set_push_contact_shape`
- `descend_or_advance_to_contact`
- `push_slowly`
- `retract_from_contact`

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_push_start | align gripper before contact ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: set_push_contact_shape | make a stable pushing surface ===
gripper_control(180, 120); sleep(...)
# === DEFENSE_EI_PHASE: descend_or_advance_to_contact | slow contact approach ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: push_slowly | translate object in small increments ===
move_ee(...); sleep(...)
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: retract_from_contact | back away from object ===
move_ee(...); sleep(...)
```

## `pull`

Pull requires contact or grasp before moving opposite the intended contact
direction. Required phase order:

- `approach_pull_contact`
- `grasp_or_hook`
- `pull_slowly`
- `release_if_grasped`
- `retract_from_contact`

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_pull_contact | align to pull feature ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: grasp_or_hook | secure contact before pulling ===
gripper_control(255, 220); sleep(...)
# === DEFENSE_EI_PHASE: pull_slowly | pull with short conservative increments ===
move_ee(...); sleep(...)
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: release_if_grasped | let go after pull ===
gripper_control(0, 150); sleep(...)
# === DEFENSE_EI_PHASE: retract_from_contact | retreat from manipulated object ===
move_ee(...); sleep(...)
```

## `press`

Press is a short approach, slow normal contact, brief hold, and retraction.
Required phase order:

- `approach_press_target`
- `set_press_contact_shape`
- `press_slowly`
- `hold_press`
- `retract_from_press`

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_press_target | align to button or target surface ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: set_press_contact_shape | close gripper for stable contact ===
gripper_control(255, 120); sleep(...)
# === DEFENSE_EI_PHASE: press_slowly | press along observed normal in small increments ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: hold_press | keep contact briefly ===
sleep(...)
# === DEFENSE_EI_PHASE: retract_from_press | move back along approach direction ===
move_ee(...); sleep(...)
```

## `open` / `close`

Open and close articulated objects with a grasp/hook, small rotational or arc
increments, then release. Required phase order:

- `approach_handle_or_lid`
- `grasp_or_hook_articulation`
- `articulate_slowly`
- `release_articulation`
- `retract_from_articulation`

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_handle_or_lid | align to articulated feature ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: grasp_or_hook_articulation | secure the feature ===
gripper_control(255, 220); sleep(...)
# === DEFENSE_EI_PHASE: articulate_slowly | move through small arc increments ===
move_ee(...); sleep(...)
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: release_articulation | release after target state is reached ===
gripper_control(0, 150); sleep(...)
# === DEFENSE_EI_PHASE: retract_from_articulation | retreat without bumping object ===
move_ee(...); sleep(...)
```

## `pour`

Pour is grasp, lift, transfer above target, tilt in small increments, pause,
untilt, then place/release or retreat according to the atomic task. Required
phase order:

- `approach_source_above`
- `grasp_source`
- `lift_to_pour_height`
- `transfer_above_target`
- `tilt_to_pour`
- `hold_pour`
- `untilt`
- `return_or_release_source`
- `retract_up`

Pseudocode template:

```python
# === DEFENSE_EI_PHASE: approach_source_above | align above source container ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: grasp_source | close and settle grasp ===
gripper_control(255, 250); sleep(...)
# === DEFENSE_EI_PHASE: lift_to_pour_height | lift clear of table and obstacles ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: transfer_above_target | move above receiving container ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: tilt_to_pour | tilt slowly over target opening ===
move_ee(...); sleep(...)
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: hold_pour | maintain tilt briefly ===
sleep(...)
# === DEFENSE_EI_PHASE: untilt | rotate back slowly ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: return_or_release_source | return source if required ===
move_ee(...); sleep(...)
# === DEFENSE_EI_PHASE: retract_up | retreat safely ===
move_ee(...); sleep(...)
```
