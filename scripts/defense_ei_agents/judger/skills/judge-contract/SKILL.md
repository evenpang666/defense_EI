---
name: judge-contract
description: Strict JSON output contract for defense_ei_agents atomic-task judging.
tags: [defense_ei_agents, judge, json-output]
---

# Judge Contract

Return exactly this JSON object:

```
{"task_result": "SUCCESS" | "FAIL", "analysis": "<concise reason; empty string allowed if SUCCESS>"}
```

Rules:
- Judge one atomic task at a time.
- Compare the final front/wrist RGB images against the atomic task's target state
  and done criteria.
- Treat execution errors as failure unless the visual state clearly satisfies the
  atomic task anyway.
- For failure, name the concrete problem: wrong object, missed target, no visible
  motion, incomplete press/open/close/pour, object dropped, gripper still holding,
  unsafe collision, or occlusion.
- Do not output markdown or prose outside the JSON object.
