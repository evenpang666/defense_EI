# defense_ei_agents Real UR7e Workflow

This workflow runs four DefenseAgent ReAct agents around a real UR7e:

1. `planner` decomposes the natural-language task into primitive-level atomic tasks.
2. `supervisor` turns each atomic task into execution-relevant information and visual done criteria.
3. `coder` generates restricted Python for exactly one atomic task.
4. `judger` evaluates the post-execution front/wrist RGB images and returns `SUCCESS` or `FAIL`.

The orchestrator is `scripts/evaluate_defense_agent_real.py`. It executes real robot code only; there is no dry-run or generate-only mode.

## Safety Notes

- Use this only with the robot in a supervised lab setting.
- Put the UR e-Series robot in Remote Control mode before running.
- Keep E-stop reachable.
- Test camera capture and a tiny free-space motion before object interaction.
- Generated code is statically checked before execution. It may call only:
  `ee_pose`, `move_ee`, `gripper_control`, `move_x`, `move_y`,
  `move_z`, `rotate_x`, `rotate_y`, `rotate_z`, `sleep`, and `print`.
- Primitive names such as `pick_place`, `push`, `pull`, `press`, `open`,
  `close`, and `pour` are labels only. The coder must expand them into
  low-level `move_ee` and `gripper_control` phases.

## Required Environment Variables

DefenseAgent resolves LLM settings from profile values first, then environment variables. The checked-in profiles intentionally leave `provider`, `model`, `base_url`, and `api_key` blank.

Recommended OpenAI-compatible setup:

```bash
export AGENT_LAB_LLM_PROVIDER="openai"
export LLM_MODEL_ID="gpt-5.4"
export LLM_BASE_URL="https://api.qnaigc.com"
export LLM_API_KEY="..."
```

Provider-specific variables also work when `LLM_*` is unset:

```bash
export AGENT_LAB_LLM_PROVIDER="openai"
export OPENAI_MODEL="gpt-5.4"
export OPENAI_BASE_URL="https://api.qnaigc.com"
export OPENAI_API_KEY="..."
```

Optional embedding variables are currently accepted by the CLI for future memory/RAG use. The real workflow now runs with memory disabled, so these are not required:

```bash
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_API_KEY="..."
export EMBEDDING_BASE_URL="https://api.qnaigc.com"
export EMBEDDING_DIMS="1536"
```

## Dependencies

Install the project dependencies, DefenseAgent, RealSense bindings, and UR RTDE support in the active Python environment:

```bash
pip install imageio pyyaml openai ur-rtde pyrealsense2
pip install -e third_party/DefenseAgent
```

If you use the repository-wide environment from the top-level README, make sure `ur-rtde`, `pyrealsense2`, `imageio`, and `pyyaml` are present.

## Hardware Inputs

The runtime uses:

- URScript primary socket: port `30003`
- URScript secondary socket: port `30002`
- RTDE receive: port `30004`
- Robotiq socket: port `63352`
- Two Intel RealSense D435i RGB streams, front and wrist

If only one RealSense stream is found, it is reused as both front and wrist input. For reliable judging, use two cameras and pass serials explicitly:

```bash
--camera-serials FRONT_SERIAL,WRIST_SERIAL
```

## Capture Images Only

Use this to verify camera wiring without connecting to the robot:

```bash
python scripts/evaluate_defense_agent_real.py \
  --task "capture test" \
  --capture-only \
  --camera-serials FRONT_SERIAL,WRIST_SERIAL
```

The command writes `current_front_rgb.png`, `current_wrist_rgb.png`, and `summary.json` under `logs/defense_agent_real/<timestamp>/`.

## Run The Full Workflow

Normal run with live initial capture:

```bash
python scripts/evaluate_defense_agent_real.py \
  --task "pick up the red block and place it into the bowl" \
  --robot-ip 169.254.26.10 \
  --camera-serials FRONT_SERIAL,WRIST_SERIAL \
  --max-attempts-per-atomic 3
```

Run from existing initial images while still using RealSense for post-execution feedback:

```bash
python scripts/evaluate_defense_agent_real.py \
  --task "pick up the red block and place it into the bowl" \
  --current-front-image logs/example/current_front_rgb.png \
  --current-wrist-image logs/example/current_wrist_rgb.png \
  --robot-ip 169.254.26.10 \
  --camera-serials FRONT_SERIAL,WRIST_SERIAL
```

When existing initial images are supplied, both paths must be provided. The orchestrator still opens the RealSense cameras so the judger receives fresh after-action images.

## Robotiq Fallback

By default, arm control can continue if the Robotiq socket is unavailable unless `--strict-gripper-connection` is set. If your setup requires URScript fallback definitions, pass:

```bash
--robotiq-urscript-defs-path /path/to/robotiq_defs.script
```

Use strict mode when gripper motion is required for the task:

```bash
--strict-gripper-connection
```

## Outputs

Each run creates a timestamped directory under `logs/defense_agent_real/` containing:

- `real_*_profile.yaml`: resolved profile copies for the run
- `defense_ei_plan.json`: planner output
- `defense_ei_atomic_task_info.json`: supervisor output
- `atomic_XX/attempt_YY/real_atomic_actions.py`: generated code
- `atomic_XX/attempt_YY/syntax_check.json`: syntax and runtime API validation
- `atomic_XX/attempt_YY/real_execution.json`: robot execution report
- `atomic_XX/attempt_YY/after_front_rgb.png` and `after_wrist_rgb.png`
- `atomic_XX/attempt_YY/real_atomic_judge.json`: judger result
- `summary.json`: final status summary

Exit code is `0` when all atomic tasks complete and `2` when the workflow stops after failed attempts.

## Profiles And Contracts

Profiles live in:

- `planner/profile.yaml`
- `supervisor/profile.yaml`
- `coder/profile.yaml`
- `judger/profile.yaml`

The coder contracts are the most safety-critical:

- `coder/prompts/system.md`
- `coder/skills/atomic-code-contract/SKILL.md`
- `coder/skills/real-robot-code-contract/SKILL.md`
- `coder/skills/primitive-skill-contract/SKILL.md`
- `coder/python_tools/atomic_runtime_contract.py`

If you add new runtime APIs, update all of these plus `make_real_runtime_api()` and the orchestrator's static runtime-call allowlist together.
