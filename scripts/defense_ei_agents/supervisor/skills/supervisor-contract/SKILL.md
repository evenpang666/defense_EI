---
name: supervisor-contract
description: JSON-only atomic task information contract for defense_ei_agents supervisor.
tags: [defense_ei_agents, supervisor, robotics, json-output]
---

# Supervisor Contract

Return exactly one JSON object:

```
{
  "status": "in_progress" | "completed",
  "summary": "<short summary>",
  "atomic_task_info": [
    {
      "id": 1,
      "description": "<planner atomic task>",
      "primitive_skill": "pick_place|push|pull|press|open|close|pour",
      "operated_object": "<object name or null>",
      "source_object": "<object name or null>",
      "target_object": "<object name or null>",
      "target_state": "<observable desired end state>",
      "motion_type": "translation|rotation|hybrid",
      "constraints": ["<clearance/speed/contact/friction note>", "..."],
      "done_criteria": "<visual criterion for this atomic task>",
      "failure_risks": ["<risk>", "..."]
    }
  ]
}
```

Rules:
- Preserve planner order and ids.
- One `atomic_task_info` entry per planner atomic task.
- Use camera-observable `target_state` and `done_criteria`.
- Do not invent calibrated coordinates unless provided by the user.
- Keep the contract independent of any evobody paths, tools, or memory.
