---
name: atomic-code-contract
description: Runtime-safe code contract for defense_ei_agents atomic action generation.
tags: [defense_ei_agents, atomic, codegen, runtime]
---

# Atomic Code Contract

Generate one executable Python block for one atomic real-robot task.

Hard rules:
- Use `real-robot-code-contract`.
- Use `primitive-skill-contract` when the task names a primitive skill such as
  `pick_place`, `push`, `pull`, `press`, `open`, `close`, or `pour`.
- Call only runtime APIs from the selected contract and `runtime_api_catalog("real")`.
- Do not define helpers or import modules.
- Output exactly one fenced Python block; no prose outside it.
- Do not use quaternion rotations. Absolute TCP poses are 6D Cartesian vectors
  `[x, y, z, rx, ry, rz]`, with UR rotation-vector radians.
- For `move_ee`, use millimeters for `dx/dy/dz` and radians for
  `droll/dpitch/dyaw`. Do not pass `steps`; use the optional
  `velocity`/`acceleration` parameters for hardware speed control.
- Use `move_ee` for incremental motion. Do not call `move_to`; it is not
  available in the generated-code runtime API.
- Add `# === DEFENSE_EI_PHASE: <slug> | <short goal> ===` before each robotic stage.
- Keep actions conservative, slow, and explicit. Near-object interaction phases
  must use smaller relative deltas, low `velocity`/`acceleration`, and short
  pauses. Add a short `sleep(...)` after every move command.

Mandatory checks before final answer:
1. Call `runtime_api_catalog("real")`.
2. Call `check_forbidden_tokens(code)`.
3. Call `syntax_check_code_via_ast_tree(code)`.
