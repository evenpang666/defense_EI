---
name: planner-contract
description: JSON-only primitive-level decomposition contract for defense_ei_agents planner.
tags: [defense_ei_agents, planner, robotics, json-output]
---

# Planner Contract

Return exactly one JSON object:

```
{
  "status": "in_progress" | "completed",
  "summary": "<short summary>",
  "atomic_tasks": [
    {
      "id": 1,
      "description": "<one atomic task>",
      "primitive_skill": "pick_place|push|pull|press|open|close|pour",
      "source_object": "<object name or null>",
      "target_object": "<target object/region or null>",
      "reason": "<why this primitive is needed>"
    }
  ]
}
```

Rules:
- `completed` means no robot action is needed and `atomic_tasks` is empty.
- `in_progress` means at least one atomic task is needed.
- Use the smallest ordered list that still captures the task.
- Keep pose, clearance, target-state, and visual-verification details for the supervisor.
