# Code map

One section per module: what it is for, the few names worth knowing, and its tests. Links go to
the source on GitHub.

## `spec.py` — what a policy needs

Turns a checkpoint directory into a `PolicySpec`: camera names, state and action sizes, chunk
length, whether a task sentence is required. Reads the normalisation statistics' header first and
`config.json` second ([why](../learn/04-normalisation.md)). Also validates an incoming
observation against the spec.

| name | does |
|---|---|
| `PolicySpec.from_checkpoint(path)` | build the spec without loading weights |
| `PolicySpec.validate(images, state, task)` | refuse a bad observation with a reason |
| `safetensors_shapes(path)` | tensor shapes from a `.safetensors` header, no data read |

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/spec.py) · tests: `tests/test_spec_codec_license.py`

## `registry.py` and `licenses.py` — names and licences

`resolve(name)` accepts an alias, a local directory or an HF repo id and returns the directory
plus a `LicenseInfo`. Licences come from the registry entry, then the checkpoint, then the
backbone (PaliGemma → `gemma-terms`).

[registry.py](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/registry.py) · [licenses.py](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/licenses.py) ·
bundled aliases: `vlaserve/data/models.yaml`

## `compat.py` — old checkpoints

Registers processor-step names that newer LeRobot versions renamed, as subclasses of their
replacements. Today: `delta_actions_processor` → `relative_actions_processor`.

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/compat.py) · tests: `tests/test_compat.py`

## `backends/` — running the policy

`Backend` is the interface (`predict(obs, seed) → ActionChunk`, `reset()`, `close()`).
`LeRobotBackend` implements it: frames to tensors (no resize), LeRobot preprocessor,
`predict_action_chunk`, postprocessor per step. Eager PyTorch unless `compile: true`.

[base.py](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/backends/base.py) ·
[lerobot_backend.py](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/backends/lerobot_backend.py) ·
tests: `tests/test_gpu_parity.py` (opt-in, `-m gpu`)

## `codec.py` — the wire format

msgpack with numpy arrays as `{dtype, shape, bytes}`. Every header is checked before memory is
allocated; bodies over 64 MB are refused.

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/codec.py)

## `server.py` — HTTP

FastAPI app plus `ModelManager` (load / unload / per-model lock). Caller mistakes are typed
errors mapped to 4xx; inference runs in a worker thread so `/health` stays responsive; no GPU
work starts for a client that has disconnected.

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/server.py) · tests: `tests/test_server_client.py` · API:
[reference](../reference/api.md)

## `client.py` — the robot side

| name | does |
|---|---|
| `PolicyClient` | `load`, `spec`, `reset`, `act(images, state, task, seed)` |
| `ActionQueue` | pure logic: drop stale actions, latest chunk wins |
| `ChunkExecutor` | fixed-rate control loop + background inference requests |
| `RunStats` | stalls, startup ticks, stale actions dropped, round-trip times |

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/client.py) · tests: `tests/test_server_client.py`,
`tests/test_client_is_light.py` (client must import without torch)

## `profiles.py` — the robot

`RobotProfile` (fps, camera name mapping, state/action size, joint names) and `check(spec)`,
which lists every mismatch. Bundled profiles: `vlaserve/data/profiles/`.

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/profiles.py) · tests: `tests/test_profiles.py`

## `cli.py` — the command line

`list`, `inspect`, `check`, `serve`, `bench`. `serve` and `bench` print an install hint if the
server dependencies are missing.

[Source](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/cli.py) · [reference](../reference/cli.md)

## Outside the package

| path | what |
|---|---|
| `deploy/Dockerfile`, `deploy/README.md` | the GPU server image |
| `examples/robot_loop.py` | a control loop with stub robot I/O |
| `examples/kaair_sim_eval.py` | closed-loop evaluation in kaair_sim |
| `website/` | this site; `website/tools/figures.py` renders its figures from recorded runs |
| `Makefile` | every command on one page — `make help` |
