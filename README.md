# EvoBody (LeRobot-only Pipeline)

EvoBody is a self-evolving VLA workflow for atomic task chains in MuJoCo scenes.

The current pipeline is **pure PyTorch + LeRobot**:

1. Evaluate each atomic task in order.
2. If one atomic task fails, trigger EvoMa auto data generation.
3. Convert generated trajectories to LeRobot dataset.
4. Continual finetune a **primitive-specific LoRA expert**.
5. If expert already exists, run replay training first (default replay rate `0.2`), then current data finetuning.

---

## 1) Environment Setup

If your environment is already ready, skip.

Git submodules live under `third_party/`:

- **LeRobot** — [huggingface/lerobot](https://github.com/huggingface/lerobot) at `third_party/lerobot` (this pipeline’s training stack).
- **LIBERO** — [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) at `third_party/libero` (lifelong / multitask manipulation benchmark; optional unless you use LIBERO tasks or datasets).
- **DefenseAgent** — [yishu031031/DefenseAgent](https://github.com/yishu031031/DefenseAgent) at `third_party/DefenseAgent` (optional dependency for DefenseAgent-based workflows).

After clone, fetch submodule checkouts:

```bash
git submodule update --init --recursive
cp -r src/pi05 third_party/lerobot/src/lerobot/policies
```

Fresh clone with submodules in one step:

```bash
git clone --recurse-submodules <repository-url>
```

To register these paths in a new fork (this repo already lists them in `.gitmodules`):

```bash
git submodule add https://github.com/huggingface/lerobot.git third_party/lerobot
git submodule add https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git submodule add https://github.com/yishu031031/DefenseAgent.git third_party/DefenseAgent
git submodule update --init --recursive
cp -r src/pi05 third_party/lerobot/src/lerobot/policies
```

For LIBERO Python deps, datasets, and evaluation, follow the upstream [LIBERO README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md) (for example `cd third_party/libero && pip install -r requirements.txt && pip install -e .` after matching their Python/CUDA notes).

Then create the environment and install dependencies:

```bash
conda create -y -n evobody python=3.12
conda activate evobody
conda install ffmpeg=7.1.1 -c conda-forge

pip install uv
cd third_party/lerobot
uv pip install -e .[pi]
uv pip install torch==2.9.1 torchvision==0.24.1 gradio mujoco ffmpeg openai peft torchcodec imageio[ffmpeg] imageio[pyav] num2words ur-rtde pyrealsense2
```

Set LLM env vars for EvoMa generation and judging:

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="qwen/qwen3.6-plus"
export API_KEY="..."
```

---

## 2) End-to-End Closed Loop

Run the full pipeline with comma-separated atomic tasks:

```bash
python scripts/pipeline_lerobot.py \
  --scene-json chemistry.json \
  --task "pick up beaker and place to hot plate" \
  --max-rounds 8 \
  --eval-loops 1 \
  --eval-min-success 1 \
  --generate-success-target 1 \
  --train-steps 10 \
  --train-batch-size 8 \
  --replay-rate 0.2
```

### `scripts/pipeline_lerobot.py` Parameters

| Parameter | Type | Default | Description | Example |
|---|---|---|---|---|
| `--scene-json` | `str` | required | Input scene JSON path used for evaluation and generation. | `--scene-json chemistry.json` |
| `--task` | `str` | `""` | Atomic task chain as one string. Comma-separated tasks run in order. Ignored when `--atomic-tasks-json` is provided. | `--task "pick beaker,move beaker,place beaker"` |
| `--atomic-tasks-json` | `str` | `""` | JSON file containing atomic task list. Higher priority than `--task`. | `--atomic-tasks-json tasks.json` |
| `--max-rounds` | `int` | `8` | Max learn rounds per atomic task before marking failure. | `--max-rounds 8` |
| `--eval-loops` | `int` | `3` | Number of evaluation rollouts for each verify attempt. | `--eval-loops 3` |
| `--eval-min-success` | `int` | `2` | Minimum successful eval rollouts required to pass one atomic task. | `--eval-min-success 2` |
| `--generate-success-target` | `int` | `20` | Target successful trajectories for each generation call. | `--generate-success-target 20` |
| `--generate-max-attempts` | `int` | `0` | Max attempts in generation (`0` means unlimited). | `--generate-max-attempts 100` |
| `--generate-lerobot-root` | `str` | `"~/.cache/huggingface/lerobot"` | Local LeRobot dataset root. | `--generate-lerobot-root ~/.cache/huggingface/lerobot` |
| `--train-steps` | `int` | `3000` | Total train steps per failed round. Split into replay + current data when replay is enabled. | `--train-steps 3000` |
| `--train-batch-size` | `int` | `8` | Batch size used in training and validation calls. | `--train-batch-size 8` |
| `--policy-type` | `str` | `pi05` | LoRA-MoE backbone policy type. Choose from `pi05` or `smolvla`. | `--policy-type smolvla` |
| `--base-model-path` | `str` | auto | Optional backbone init path. Defaults to `checkpoints/pi05_base` for `pi05`, `checkpoints/smolvla_base` for `smolvla`. | `--base-model-path checkpoints/smolvla_base` |
| `--replay-rate` | `float` | `0.2` | Replay ratio for existing primitive expert. Replay steps = `train_steps * replay_rate`. | `--replay-rate 0.2` |
| `--val-max-batches` | `int` | `20` | Max batches for offline validation after each training round. | `--val-max-batches 20` |
| `--checkpoint-root` | `str` | `"checkpoints"` | Root directory for expert checkpoints and expert registry state. New LoRA parameters are saved under this directory. | `--checkpoint-root checkpoints` |
| `--loop-log-dir` | `str` | `"logs/closed_loop_lerobot"` | Root directory for loop outputs (`summary.json`, per-round artifacts). | `--loop-log-dir logs/closed_loop_lerobot` |

Notes:

- If both `--task` and `--atomic-tasks-json` are given, `--atomic-tasks-json` is used.
- `--task` supports comma splitting only (for example `"a,b,c"`).
- For existing primitive experts, replay and current training are both LoRA-mode continual finetuning.

### Pipeline Behavior

- `--task` supports one string with commas; tasks run in order.
- Each atomic task is verified first.
- On failure, `scripts/generate_cli.py` generates new data.
- Primitive is inferred from generated `repo_id` prefix (for example `pick_place/xxx` -> primitive `pick_place`).
- Expert state is tracked in `checkpoints/expert_registry.json` (or under `--checkpoint-root`).
- Training outputs for each primitive are stored under `checkpoints/experts/<primitive>/...` (or under `--checkpoint-root`) and validation reads the latest `pretrained_model` from this location.
- Training SFT startup rule: if current primitive already has an expert checkpoint in `checkpoints/experts/<primitive>/...`, training continues from that expert checkpoint.
- After each verified atomic task, pipeline retrains a lightweight task router from the accumulated verified task descriptions and primitive labels, and saves it under `checkpoints/router/router_state.json` (or under `--checkpoint-root`).
- If a target train output directory already exists, pipeline allocates a unique suffixed directory (for example `_v001`) to avoid overwrite/resume conflicts.
- Validation checkpoint resolution rule:
  - if no LoRA expert checkpoint exists for the primitive, validate with `checkpoints/pi05_base`
- if expert checkpoints exist, validate with all latest primitive experts under `checkpoints/experts/*` using the learned router, and force `checkpoints/pi05_base` as pretrained backbone
- pipeline validation passes `--force-full-moe`, so each run always scans the current `--checkpoint-root` and loads the full expert set
- The learned router is stored at `checkpoints/router/router_state.json` and is retrained from the verified task history after each atomic task round.
- For existing primitive experts:
  - replay training on old primitive dataset first
  - then finetune on current generated dataset

`scripts/pipeline.py` is now only a compatibility shim and forwards to `scripts/pipeline_lerobot.py`.

---

## 3) Train Script (LoRA-MoE Continual)

`scripts/lerobot_train_cli.py` supports LoRA expert training:

```bash
python scripts/lerobot_train_cli.py \
  --dataset-repo-id pick_place/pick_up_the_beaker_and_place_it_onto_the_hotplate \
  --dataset-root ~/.cache/huggingface/lerobot \
  --output-dir logs/lerobot_train_round_001 \
  --policy-path logs/prev_round/checkpoints/003000/pretrained_model \
  --finetune-mode lora_moe \
  --expert-id pick_place \
  --peft-target-modules all-linear \
  --peft-r 16
```

Useful args:

- `--finetune-mode`: `lora_moe` (default) or `full`
- `--expert-id`: primitive expert name (for bookkeeping)
- `--policy-path`: previous checkpoint for continual finetuning
- `--peft-target-modules`, `--peft-r`, `--peft-init-type`

---

## 4) Validate Script

Full MoE from current checkpoints (same strategy as pipeline):

```bash
python scripts/lerobot_validate_cli.py \
  --dataset-repo-id pick_place/pick_up_beaker \
  --dataset-root ~/.cache/huggingface/lerobot \
  --checkpoint-root checkpoints \
  --base-model-path checkpoints/pi05_base \
  --force-full-moe \
  --moe-router router \
  --router-state-json checkpoints/router/router_state.json \
  --batch-size 4 \
  --max-batches 20
```

When `--policy-path`/`--policy-paths-json` is omitted, the script auto-discovers policy from `--checkpoint-root`:

- prefers latest expert checkpoint under `checkpoints/experts/<primitive>/.../pretrained_model`
- falls back to `checkpoints/pi05_base` if no expert checkpoint exists
- router mode uses the learned router state to select one expert path for inference

Single-expert mode (optional):

```bash
python scripts/lerobot_validate_cli.py \
  --dataset-repo-id pick_place/pick_up_beaker \
  --dataset-root ~/.cache/huggingface/lerobot \
  --checkpoint-root checkpoints \
  --base-model-path checkpoints/pi05_base \
  --moe-router single \
  --primitive pick_place \
  --batch-size 4 \
  --max-batches 20
```

Multi-expert (MoE-style offline evaluation):

```bash
python scripts/lerobot_validate_cli.py \
  --dataset-repo-id pick_place/pick_up_beaker \
  --dataset-root ~/.cache/huggingface/lerobot \
  --policy-paths-json experts.json \
  --moe-router router \
  --router-state-json checkpoints/router/router_state.json \
  --result-json-out logs/validate_moe.json
```

`experts.json` example:

```json
[
  "logs/expert_pick_place/checkpoints/003000/pretrained_model",
  "logs/expert_transfer/checkpoints/003000/pretrained_model"
]
```

---

## 5) Standalone Components

### Scene Building UI

```bash
python scripts/build_gradio.py
```

### LIBERO-Long Continual Train + Eval (LoRA-MoE)

```bash
python scripts/libero/pipeline_libero_long_lora_moe.py \
  --dataset-repo-id HuggingFaceVLA/libero \
  --dataset-root ~/.cache/huggingface/lerobot \
  --task-suite libero_10 \
  --checkpoint-root checkpoints \
  --loop-log-dir logs/libero_long_pipeline
```

### Autonomous Data Generation Only

```bash
python scripts/generate_cli.py \
  --scene-json chemistry.json \
  --task "pick up the beaker and place it on the hot plate" \
  --overwrite
```

### Policy Evaluation Only

```bash
python scripts/evaluate_cli.py \
  --scene-json chemistry.json \
  --task "pick up the beaker and place it on the hot plate" \
  --inference-backend pytorch \
  --eval-loops 3 \
  --time-limit-s 30 \
  --control-repeat 10 \
  --num-steps 10
```

---

## 6) Outputs and Logs

- Closed-loop summary: `logs/closed_loop_lerobot/summary.json`
- Primitive expert registry: `checkpoints/expert_registry.json` (or `<checkpoint-root>/expert_registry.json`)
- Primitive checkpoint outputs: `checkpoints/experts/<primitive>/atomic_xxx_round_xxx_{replay|current}/checkpoints/*/pretrained_model`
- Per-round artifacts:
  - `evaluate_result.json`
  - `generate_result.json`
  - `train_result.json`
  - `validate_result.json`

---

## Notes

- This repository no longer uses OpenPI/JAX pipeline paths for training loop.
- Training and validation in the loop are LeRobot-based.
- Offline server requirement: place Paligemma files at `checkpoints/paligemma-3b-pt-224` (must contain `config.json`).
- Training/validation scripts force offline mode and rewrite legacy tokenizer config from `google/paligemma-3b-pt-224` to local checkpoint path.