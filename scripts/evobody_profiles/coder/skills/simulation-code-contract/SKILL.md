---
name: simulation-code-contract
description: MuJoCo simulation runtime code contract for Evobody atomic action generation. Use only for simulation validation and scene-json rollouts.
tags: [evobody, atomic, codegen, runtime, mujoco, simulation]
---

# Evobody Simulation Code Contract

Generate ONE executable Python block per atomic task that the MuJoCo runtime can `exec` as-is.

## Simulation Runtime APIs

Primitive runtime APIs:
1. `move_to(pos, quat, num_steps)` - `pos` is `(x,y,z)`, `quat` is `(w,x,y,z)`.
2. `move_ee(dx, dy, dz, droll, dpitch, dyaw, steps)` - relative delta pose, angles in degrees.
3. `gripper_control(value, delay)` - `value` 0..255, where 0=open and 255=closed; delay in ms.
4. `ee_pose() -> (pos, quat)` - current end-effector pose, `quat` is `(w,x,y,z)`.

Composite skills:
- `pick_and_place(object_pose, target_pose, direction_x, direction_y, direction_z, approach_height, lift_height, grasp_value, release_value, move_steps, grip_delay)`
- `push(target_pose, object_pose, push_distance, approach_height, grasp_value, move_steps, grip_delay)`
- `pull(target_pose, object_pose, pull_distance, approach_height, grasp_value, release_value, move_steps, grip_delay)`
- `press(object_pose, direction_x, direction_y, direction_z, grasp_value, move_steps, grip_delay)`
- `open(grasp_pose, rotation_radius, rotation_angle_deg, grasp_value, move_steps, grip_delay)`
- `close(grasp_pose, rotation_radius, rotation_angle_deg, grasp_value, move_steps, grip_delay)`
- `pour(object_pose, target_pose, direction_x, direction_y, direction_z, rot_x, rot_y, rot_z, approach_height, lift_height, grasp_value, release_value, move_steps, grip_delay)`
- `move_x(distance, steps)` / `move_y(distance, steps)` / `move_z(distance, steps)`
- `rotate_x(angle_deg, steps)` / `rotate_y(angle_deg, steps)` / `rotate_z(angle_deg, steps)`

Pose helpers:
- `get_object_abs_pose(object_poses, object_name) -> (pos, quat)`
- `recover_grasp_pose_from_offset(object_pos_xyz, object_quat_wxyz, offset_pos_xyz, offset_rpy_deg) -> {"grasp_pos_world_xyz","grasp_quat_world_wxyz"}`

## Simulation Tool Sequence

Run this exact pipeline for simulation validation:

1. `runtime_api_catalog("simulation")` once per session.
2. `check_forbidden_tokens(code)` after drafting code.
3. `syntax_check_code_via_ast_tree(code)` after forbidden-token scan passes.
4. `load_mujoco_scene_from_json(scene_json_path)` once per scene path.
5. `execute_generated_code_in_loaded_scene(code, final_image_path, error_output_path)`.

If execution fails, revise only the failing phase and retry the checks. Keep retries low.

## Simulation Pose Rules

- Hardcode numeric poses from reconstructed scene context inline.
- Use grasp offsets when available; do not grasp object centers by default.
- Respect `tabletop_clearance.safe_carry_end_effector_z_m` and object `geom_top_z_world` for all lateral transfers.
- `pick_and_place` is translation-only with fixed grasp orientation. `target_pose` uses xyz only.
- Keep motion conservative: `move_steps` / `num_steps` usually 100-300, with higher values near contact.

