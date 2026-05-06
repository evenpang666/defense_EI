---
name: supervisor-contract
description: Single-pass JSON planning contract for the Evobody supervisor. Decomposes the user task into validated atomic tasks.
tags: [evobody, planning, robotics, json-output]
---

# Evobody Supervisor Contract

Plan the user task as an ordered list of atomic primitives, validate it once, and emit JSON as the final answer. Optimize for **one tool call total** in the happy path.

---

## Output shape (final answer)

Return **only** this JSON object — no markdown fences, no commentary:

```
{
  "status": "in_progress" | "completed",
  "summary": "<short sentence>",
  "atomic_tasks": [
    {
      "id": 1,
      "description": "...",
      "primitive_skill": "pick_place|push|pull|press|open|close|pour",
      "source_object": "<name or null>",
      "target_object": "<name or null>",
      "interactive_objects": ["<name>", ...],
      "motion_type": "rotation|translation|hybrid",
      "constraints": ["..."],
      "done_criteria": "..."
    }
  ]
}
```

Rules:
- `status="completed"` ⇒ `atomic_tasks` MUST be `[]`. Use this only when the front image already shows the goal state.
- `status="in_progress"` ⇒ `atomic_tasks` MUST be non-empty.
- Atomic tasks are ordered; `id` starts at 1 and increments by 1.
- Each atomic task must change at least one `interactive_objects` pose (per definition of an atomic primitive).

---

## Tool sequence (deterministic, one-shot fast path)

1. **Default path:** call `extract_and_validate_supervisor_json(raw_text)` with your draft JSON string.
   - On success use the returned `extracted_json` as the final answer. Stop.
   - On exception, fall through to step 2.
2. **Fallback (only if step 1 raised):**
   - Call `extract_supervisor_json(raw_text)` to recover the inner JSON object.
   - Call `validate_supervisor_json(extracted_json)`; if it raises, fix the offending field and re-validate.
3. **Schema lookup:** call `supervisor_output_schema()` **at most once per session** if you forgot the field set. Static data — do not re-query.

Scene preparation is owned by the `scene-loader` skill; do not load the scene from this skill.

---

## Decomposition rules

- One atomic task = one composite primitive run (`pick_place`, `push`, `pull`, `press`, `open`, `close`, `pour`).
- Do not combine two object transfers into one atomic task; emit two entries with sequential `id`s instead.
- `motion_type`:
  - `pick_place`, `push`, `pull` → `translation`
  - `open`, `close` → `rotation`
  - `pour`, `press` (with tilt or oblique approach) → `hybrid`
- `constraints` must call out: required carry clearance, fragile-object handling, gripper open/close value if non-default.
- `done_criteria` must be observable from a camera image (pose region, contact state, articulation angle), not internal robot state.

---

## Iteration budget

- Validation retries: ≤ 2. After two failures, return the last validator error JSON verbatim and stop.

---

## What NOT to do

- Do not wrap the final answer in ```json fences.
- Do not pass markdown text directly to `validate_supervisor_json` — extract first.
- Do not invent objects that are absent from the reconstructed scene context.
- Do not emit `status="completed"` together with non-empty `atomic_tasks`.
