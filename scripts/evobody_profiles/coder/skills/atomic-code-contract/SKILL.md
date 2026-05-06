---
name: atomic-code-contract
description: Runtime-safe, low-iteration code contract for Evobody atomic action generation. Produces directly executable Python that drives the loaded MuJoCo scene via pre-registered runtime APIs.
tags: [evobody, atomic, codegen, runtime, mujoco]
---

# Evobody Atomic Code Contract

Generate ONE executable Python block per atomic task that the runtime can `exec` as-is. Optimize for the **fewest LLM/tool turns** that still pass safety + execution.

---

## Hard rules for the generated code (must hold every turn)

- Call only the runtime APIs in `runtime_api_catalog()` (primitives, composites, helpers).
- No `def`, `class`, `import`, `from`, `lambda`, `open(`, `exec(`, `eval(`, `subprocess`, `os.`, `sys.`.
- Quaternion order is `(w, x, y, z)` everywhere.
- Insert `# === EVO_PHASE: <slug> | <short goal> ===` header lines (≥ 2; typical 3–6) before each robotic stage.
- Hardcode object/grasp poses inline. Never reference `scene_context`, `plan`, or atomic-task variables.
- Output the code inside a single ```python fenced block as the final answer; no prose outside the block.

---

## Tool sequence (deterministic, minimum-call)

Run this exact pipeline. Skip optional steps when their cached result is still valid.

| Step | Tool | When to call | Stop / next condition |
|---|---|---|---|
| 1 | `runtime_api_catalog()` | **Once per session.** Cache its output; do not re-query unless you forgot the API list. | Use cached APIs to write code. |
| 2 | `check_forbidden_tokens(code)` | Always, immediately after writing code. Cheap text scan. | If `ok=false` → strip the listed tokens and rewrite (do **not** call any other tool first). Re-run step 2. |
| 3 | `syntax_check_code_via_ast_tree(code)` | Always, after step 2 passes. | If `signal="NO_SYNTAX_ERROR"` → go to step 4. Else read `error`, fix only the cited lines, re-run step 3. |
| 4 | `load_mujoco_scene_from_json(scene_json_path)` | **Once per scene path.** Default `logs/pre_scene.json`; use the path provided in the user message if present. Do not reload on retries unless the scene path changes or step 5 reports "scene is not loaded". | If `signal="SCENE_LOADED"` → step 5. If error → return the error JSON and stop. |
| 5 | `execute_generated_code_in_loaded_scene(code, final_image_path, error_output_path)` | After steps 2–4 pass. Use `final_image_path="logs/finished_view.png"` and `error_output_path="logs/last_execution_error.txt"` unless overridden. | If `signal="EXECUTION_OK"` → emit final code block answer. Else read `error_log_path`, revise the failing stage, restart at step 2. |

---

## Iteration budget (hard caps)

- Forbidden-token rewrites: ≤ 2.
- Syntax-error rewrites: ≤ 3.
- Execution-failure rewrites: ≤ 3.
- After exceeding any cap, return the last error JSON verbatim and stop. Do not loop further.

---

## Failure-feedback handling

If the user message contains `### prior_simulation_judge_feedback` or `### prior_segment_judge_feedback`:

1. Parse `task_result` and `analysis` first.
2. Identify the specific failure mode (grasp offset, carry height, wrong direction, missing release, etc.).
3. Adjust **only** the offending phase(s); keep other EVO_PHASE markers and structure intact.
4. Then run steps 2 → 5 once. Do not regenerate end-to-end if a localized fix suffices.

---

## Final-answer rule

The final answer to the user MUST be exactly one ```python fenced block containing the executed code, ending with no trailing prose. Tool transcripts are not the final answer.

If all caps are exhausted without `EXECUTION_OK`, return:

```json
{"ok": false, "error": "<last error>", "error_log_path": "<path>"}
```

instead of code, so the orchestrator can mark the round failed without parsing.
