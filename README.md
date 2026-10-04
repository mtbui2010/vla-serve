# vla-serve

Serve a trained Vision-Language-Action policy locally with one command, and drive a robot with it.

```bash
vla-serve inspect keti-pi05-openarm-handover        # cameras, state/action, chunk, licence — no GPU
vla-serve check   keti-pi05-openarm-handover --profile openarm-bimanual
vla-serve serve   keti-pi05-openarm-handover        # http://127.0.0.1:11450
```

```python
from vlaserve.client import PolicyClient, ChunkExecutor

client = PolicyClient(model="keti-pi05-openarm-handover")
chunk = client.act(images={"base": base, "left_wrist": lw, "right_wrist": rw},
                   state=q, task="hand over the cup")      # chunk.actions: (30, 16), absolute

ChunkExecutor(client, get_observation=read_robot, send_action=robot.command, fps=30).run(
    task="hand over the cup", seconds=60)
```

The sibling of [VisionServe](../vision_serve): same "one command, local, no cloud" idea, but
in Python, because for VLAs generality matters more than a single static binary
(see `CLAUDE.md` §Decisions).

## What it handles for you

| Problem when deploying a VLA | What vla-serve does |
|---|---|
| Every policy has its own loading code, tokenizer and normalisation stats | Loads LeRobot-format checkpoints from their own `config.json` + processor files, through LeRobot's factories, so any policy LeRobot implements is in reach. No manifest to write. **Verified so far: pi0.5** (two checkpoints, bit-exact vs LeRobot) and **SmolVLA** (kaair_sim open-drawer, bit-exact vs in-process and closed-loop in MuJoCo). Others (GR00T, X-VLA, ACT, ...) go through the same code path but are untested, and some need their LeRobot extra installed (`lerobot[groot]`, ...). |
| A checkpoint saved by an older LeRobot no longer loads | `compat.py` aliases renamed processor steps without touching the files (e.g. KETI's pi0.5 OpenArm checkpoints, saved with 0.5.x). |
| The robot's cameras / joints don't match what the policy was trained on, and nothing errors | `vla-serve check` compares a **robot profile** with the policy: camera names, state/action size, joint order. |
| Inference is slower than one control period | `ChunkExecutor` runs the robot at a fixed rate, requests the next chunk early, and drops actions that went stale during inference. |
| Is my serving path the same as LeRobot's? | GPU parity test against LeRobot's own policy server, same seed. |
| What licence am I shipping? | Every model reports one; PaliGemma derivatives are detected as `gemma-terms` and warned about. Nothing is blocked. |

## Install (this machine)

```bash
conda activate vlaserve                 # Python 3.12, torch 2.10.0+cu128, lerobot 0.6.1
pip install -e '.[test]' --no-deps
export HF_HOME=/mnt/nas/huggingface     # checkpoints live on the NAS cache
```

On a fresh machine: create a Python 3.12 env, install the torch build matching your CUDA driver
first, then `pip install -e '.[test]'`.

## Models

`vla-serve list` shows aliases (`vlaserve/data/models.yaml`, plus your own in
`~/.config/vla-serve/models.yaml`). Any LeRobot checkpoint also works directly by HF repo id or
local path:

```bash
vla-serve serve /data/checkpoints/my_run/pretrained_model
vla-serve serve lerobot/pi05_libero_finetuned
```

## HTTP API

| | |
|---|---|
| `GET /health` | |
| `GET /v1/spec?model=X` | spec + licence from `config.json`; no weights loaded |
| `POST /v1/models/load` `{"model", "device"?}` | |
| `POST /v1/models/unload` / `reset` `{"model"}` | `reset` at the start of each episode |
| `GET /v1/models` | loaded models |
| `POST /v1/act` (msgpack) | `{"model", "observation": {"images": {cam: HxWx3 uint8}, "state", "task"}, "seed"?}` → `{"actions": (H, D) float32, "action_names", "n_action_steps", "timings_ms"}` |

Images go raw (bit-exact) at any resolution; the server resizes exactly as LeRobot does.
Errors in the observation return 400 with the reason, never 500.

## Tests

```bash
pytest                                            # fast, no GPU
HF_HOME=/mnt/nas/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 pytest -m gpu -s
```

## Documentation

`website/` is a MkDocs Material site (guides, how it works, code map, API reference, and an
8-chapter **Learn VLAs** course illustrated with figures from real runs). Preview locally:

```bash
pip install -r website/requirements.txt && mkdocs serve -f website/mkdocs.yml
```

`.github/workflows/docs.yml` publishes it to GitHub Pages on every push to `main` that touches
`website/` (repository Settings → Pages → Source: "GitHub Actions"). Figures are regenerated from
recorded data with `website/tools/` (see its README).

## Makefile

`make help` lists every command: setup, inspect/check, serve, container, tests, simulation.

## Status

See `HANDOFF.md`.
