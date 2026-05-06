# LIBERO Long Pipeline (LoRA-MoE PI05)

This folder provides a continual train+eval pipeline for measuring current PI05 LoRA-MoE performance on LIBERO Long (`libero_10`).

## What it does

1. Iterates over `libero_10` task ids (`0..9` by default).
2. For each task:
   - checks whether the corresponding primitive expert already exists;
   - if expert exists, runs replay stage first (same replay logic as main pipeline);
   - runs current task training using all trajectories of the current task.
3. After all tasks are trained, runs `lerobot_eval` on each long task and stores result file paths in one summary JSON.

## Run (full pipeline)

```bash
python scripts/libero/pipeline_libero_long_lora_moe.py \
  --dataset-repo-id HuggingFaceVLA/libero \
  --dataset-root ~/.cache/huggingface/lerobot \
  --task-suite libero_10 \
  --num-tasks 10 \
  --train-steps 3000 \
  --train-batch-size 8 \
  --replay-rate 0.2 \
  --eval-episodes 10 \
  --checkpoint-root checkpoints \
  --loop-log-dir logs/libero_long_pipeline
```

## Training script

```bash
python scripts/libero/train_libero_long_lora_moe.py \
  --dataset-repo-id HuggingFaceVLA/libero \
  --dataset-root ~/.cache/huggingface/lerobot \
  --task-suite libero_10 \
  --num-tasks 10 \
  --train-steps 3000 \
  --train-batch-size 8 \
  --replay-rate 0.2 \
  --checkpoint-root checkpoints \
  --loop-log-dir logs/libero_long_pipeline
```

Before training starts, the script extracts tasks from dataset metadata and builds
`task_prompt -> primitive` mapping via agent (or fallback). It is cached at:

- `logs/libero_long_pipeline/task_primitive_mapping.json` (default)
- or custom path via `--primitive-map-cache-json`

## Evaluation script

```bash
python scripts/libero/eval_libero_long_lora_moe.py \
  --dataset-repo-id HuggingFaceVLA/libero \
  --dataset-root ~/.cache/huggingface/lerobot \
  --task-suite libero_10 \
  --num-tasks 10 \
  --eval-episodes 10 \
  --checkpoint-root checkpoints \
  --loop-log-dir logs/libero_long_pipeline
```

Evaluation script reads existing experts from `checkpoints/expert_registry.json` and does not train.
It also reads primitive mapping JSON from `--primitive-map-cache-json` (or the default path above).

## Mapping options

- `--primitive-semantic-mapping`: enable agent-based semantic primitive inference
- `--primitive-agent-model`: model used by `TaskSupervisorAgent`
- `--primitive-map-cache-json`: mapping JSON path
- `--refresh-primitive-map`: force regenerate mapping JSON

## Outputs

- Expert checkpoints: `checkpoints/experts/<primitive>/...`
- Expert registry: `checkpoints/expert_registry.json`
- Pipeline summary: `logs/libero_long_pipeline/summary.json`
- Task-wise eval outputs: `logs/libero_long_pipeline/eval/task_XX/`

## Optional primitive mapping

By default primitive name is `libero_long_task_XX`.  
You can override mapping with `--primitive-map-json`:

```json
{
  "0": "pick_place",
  "1": "open_close"
}
```
