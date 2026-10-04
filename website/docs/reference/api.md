# HTTP API

Default address `http://127.0.0.1:11450`. Control endpoints use JSON; `/v1/act` uses msgpack.

| method & path | body | returns |
|---|---|---|
| `GET /health` | — | `{"status": "ok", "models": [...]}` |
| `GET /v1/spec?model=X` | — | spec + licence from the checkpoint metadata (no weights loaded) |
| `GET /v1/models` | — | loaded models: spec, device, licence, call count |
| `POST /v1/models/load` | `{"model": X, "device"?: "cuda:0"}` | the model's description |
| `POST /v1/models/unload` | `{"model": X}` | `{"unloaded": X}` |
| `POST /v1/models/reset` | `{"model": X}` | `{"reset": X}` — call at each episode start |
| `POST /v1/act` | msgpack, below | msgpack, below |

## `/v1/act`

Request (`content-type: application/msgpack`):

```python
{
  "model": "pi05-libero",
  "observation": {
    "images": {"image": <uint8 H×W×3>, "image2": <uint8 H×W×3>},   # names from the spec
    "state":  <float32 (state_dim,)>,
    "task":   "put the bowl on the plate",
  },
  "seed": 0,                                                       # optional
}
```

An array travels as `{"__nd__": true, "dtype": "uint8", "shape": [H, W, 3], "data": <bytes>}`;
`vlaserve.codec.packb` does this for you. Images may be any resolution.

Response:

```python
{
  "model": "pi05-libero",
  "actions": <float32 (H, D)>,          # absolute, in robot units
  "action_names": [...] | None,
  "n_action_steps": 50,
  "timings_ms": {"preprocess": 3.5, "inference": 300.8, "postprocess": 2.7, "total": 307.0},
}
```

## Errors

Errors are JSON `{"error": <type>, "detail": <reason>}`.

| status | error | when |
|---|---|---|
| 400 | `ObservationError` | missing/unknown camera, wrong shape or dtype, NaN in state, no task, malformed body |
| 400 | `ProfileMismatch` | robot profile does not fit the policy |
| 404 | `ModelNotFound` | not an alias, a directory, or a cached/available HF repo |
| 409 | `ModelNotLoaded` | `act`/`reset`/`unload` for a model that is not loaded |
| 422 | `UnsupportedCheckpoint` | not a LeRobot checkpoint |
| 500 | anything else | a bug — please report it |
