---
name: atomic-code-contract
description: General runtime-safe code contract for Evobody atomic action generation. Select simulation-code-contract or real-robot-code-contract for environment-specific APIs.
tags: [evobody, atomic, codegen, runtime]
---

# Evobody Atomic Code Contract

Generate one executable Python block for one atomic robotics task.

## Hard Rules

- Select exactly one environment-specific contract:
  - `simulation-code-contract` for MuJoCo simulation.
  - `real-robot-code-contract` for UR7e real-robot validation.
- Call only APIs from the selected contract and `runtime_api_catalog(...)`.
- Do not define helpers or import modules.
- Output exactly one fenced Python block; no prose outside it.
- Use quaternion order `(w, x, y, z)`.
- Add `# === EVO_PHASE: <slug> | <short goal> ===` before each robotic stage.
- Keep actions conservative and explicit.

## Mandatory Checks

Before final answer:

1. Call `runtime_api_catalog("simulation")` or `runtime_api_catalog("real")` according to the selected environment.
2. Call `check_forbidden_tokens(code)`.
3. Call `syntax_check_code_via_ast_tree(code)`.

Only simulation validation may call MuJoCo scene loading and execution tools.
Real-robot validation must not call simulation execution tools.

