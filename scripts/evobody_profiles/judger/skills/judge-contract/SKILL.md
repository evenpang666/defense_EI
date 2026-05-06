---
name: judge-contract
description: Strict, single-pass JSON output contract for Evobody final-task judging. Returns exactly {"task_result", "analysis"}.
tags: [evobody, judge, json-output]
---

# Judge Contract

Decide SUCCESS / FAIL from the final camera image(s) and the task description, then emit one JSON object as the final answer. Optimize for **one tool call total** in the happy path.

---

## Output shape (final answer)

Return **only** this JSON object — no markdown, no code fences, no commentary:

```
{"task_result": "SUCCESS" | "FAIL", "analysis": "<concise reason; empty string allowed if SUCCESS>"}
```

`task_result` is uppercase. `analysis` is a single short paragraph (≤ 80 words). For SUCCESS, `analysis` may be `""`.

---

## Tool sequence (deterministic, one-shot fast path)

1. **Default path (use this every turn):** call `extract_and_normalize_task_judge_json(raw_text)` with your draft JSON string.
   - On success it returns the canonical JSON; emit that string verbatim as your final answer. Stop.
   - On exception, fall through to step 2.
2. **Fallback (only if step 1 raised):**
   - Call `extract_task_judge_json(raw_text)` to recover the inner JSON object.
   - Call `normalize_task_judge_json(extracted_json)` to canonicalize keys and uppercase `task_result`.
3. **Schema lookup:** call `task_judge_schema()` **at most once per session** if you genuinely forgot the keys. It returns static data; do not call it on retries.

---

## Judging rules

- Compare the final image(s) against the implied final state of the task description (object position, gripper state, contact, articulation angle, pour completion, etc.).
- Treat the front image as primary evidence; if a wrist image is also attached, use it to disambiguate occlusion only.
- Do not invent measurements you cannot see.
- Common FAIL modes to name when present: wrong object grasped, object dropped, missed target, occluded target, incomplete pour/press, gripper still attached when it should have released.

---

## Iteration budget

- Normalization retries: ≤ 2. After two failures, return:

```
{"task_result": "FAIL", "analysis": "judge output normalization failed"}
```

so the orchestrator does not stall.

---

## What NOT to do

- Do not wrap the final answer in ```json fences.
- Do not call `normalize_task_judge_json` directly on markdown-wrapped text — extract first.
- Do not call any image-loading tool; images are pre-attached by the orchestrator.
- Do not output any text outside the single JSON object.
