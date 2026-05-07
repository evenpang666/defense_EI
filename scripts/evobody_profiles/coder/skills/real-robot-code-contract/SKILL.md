---
name: real-robot-code-contract
description: Real UR7e runtime code contract for one-pass Evobody validation with Intel RealSense RGB inputs. Use only when the user asks for real-robot / 真机 validation.
tags: [evobody, atomic, codegen, runtime, ur7e, real-robot]
---

# Evobody Real-Robot Code Contract

Use this skill only for real-world UR7e validation. The code will be executed once on `UR7eVectorController`; there is no MuJoCo scene and no simulation retry loop.

## Real Runtime APIs

Primitive APIs exposed by `UR7eVectorController`:
1. `move_to(pos, quat, num_steps)` - absolute TCP target, `pos=[x,y,z]` meters, `quat=[w,x,y,z]`.
2. `move_ee(dx, dy, dz, droll, dpitch, dyaw, steps)` - relative TCP delta, translation in meters, rotation in degrees.
3. `gripper_control(value, delay)` - Robotiq command, 0=open and 255=closed; delay in ms.
4. `ee_pose() -> (pos, quat)` - current TCP pose, `quat=[w,x,y,z]`.

Composite APIs exposed by `UR7eVectorController`:
- `pick_and_place(...)` and alias `pick_place(...)`
- `push(...)`
- `pull(...)`
- `press(...)`
- `open(...)`
- `close(...)`
- `pour(...)`
- `move_x(...)`, `move_y(...)`, `move_z(...)`
- `rotate_x(...)`, `rotate_y(...)`, `rotate_z(...)`

The composite signatures mirror the simulation APIs where possible:
- `pick_and_place(object_pose, target_pose, direction_x, direction_y, direction_z, approach_height, lift_height, grasp_value, release_value, move_steps, grip_delay)`
- `push(target_pose, object_pose, push_distance, approach_height, grasp_value, move_steps, grip_delay)`
- `pull(target_pose, object_pose, pull_distance, approach_height, grasp_value, release_value, move_steps, grip_delay)`
- `press(object_pose, direction_x, direction_y, direction_z, grasp_value, move_steps, grip_delay)`

## Real-World Generation Rules

- Output exactly one Python code block. No prose outside the code block.
- Do not call any MuJoCo or scene-json tool. Do not request `load_mujoco_scene_from_json` or `execute_generated_code_in_loaded_scene`.
- Use `runtime_api_catalog("real")`, `check_forbidden_tokens(code)`, and `syntax_check_code_via_ast_tree(code)` only.
- Do not define functions/classes/imports. Call only the real runtime APIs listed above plus literal Python containers/numbers.
- Keep motions very conservative: prefer small relative moves (`0.01` to `0.05` m) unless a calibrated pose is explicitly provided.
- If no calibrated object pose is available from the prompt, do not invent precise object coordinates. Prefer guarded relative motions from the current TCP pose using `ee_pose()`, `move_ee`, and gripper commands.
- Quaternion order is always `[w, x, y, z]`.
- Include `# === EVO_PHASE: <slug> | <short goal> ===` before each robotic stage; use at least two phases.

## One-Pass Real Validation

The orchestrator will execute this sequence exactly once:

`supervisor -> coder -> real UR7e execution -> judger`

The coder should therefore generate a single cautious action program, not a multi-round repair strategy.

