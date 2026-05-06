---
name: scene-loader
description: Idempotent bootstrap that guarantees logs/current_view.png exists before supervisor planning. Skips when a fresh image is already on disk.
tags: [evobody, supervisor, scene, image, bootstrap]
---

# Scene Loader Bootstrap Skill

Guarantee a usable front-camera image at the path the orchestrator passes (default `logs/current_view.png`) before any planning step. Optimize for **zero tool calls** when the image already exists and is fresh.

---

## Tool

- `scene_loader_script(scene_json_path, current_view_path, randomize_texture=False)` — loads the MuJoCo scene and saves the front-camera image. Returns JSON with `ok`, `signal`, `scene_json_path`, `current_view_path`.

---

## Decision flow

1. **Skip-if-fresh:** if the user message states `current_view.png` exists, or the orchestrator already attached the image, do not call the tool. Treat the image as ready.
2. **Single-shot load:** otherwise call `scene_loader_script(scene_json_path, current_view_path, randomize_texture=False)` exactly once.
   - Use `scene_json_path` from the user message (default `chemistry.json`).
   - Use `current_view_path` from the user message (default `logs/current_view.png`).
3. **Verify response:** require `ok=true` and `signal="CURRENT_VIEW_READY"`. The returned `current_view_path` is now the canonical image for planning.
4. **Failure handling:** if `ok=false`, return the raw JSON error verbatim and stop. Do not retry, do not invent a plan.

---

## Iteration budget

- `scene_loader_script` calls per session: ≤ 1 in the happy path, ≤ 2 only if the orchestrator explicitly requests a re-render with a new `scene_json_path`.

---

## What NOT to do

- Do not call this tool on every turn — once the image exists for the active scene, treat it as immutable for the rest of the session.
- Do not enable `randomize_texture` unless the user message explicitly asks for randomized textures.
- Do not produce planning output from this skill; that belongs to `supervisor-contract`.
