"""Parity against LeRobot's own policy server, on real checkpoints. Opt in: pytest -m gpu

The reference is `PolicyServer._predict_action_chunk` from lerobot.async_inference, driven with
the same policy and processor objects, the same raw observation and the same seed. Our backend
must produce the same chunk. Frames are sent at the declared resolution: there LeRobot's resize
is an identity, and ours is absent by design (see LeRobotBackend._image).

    HF_HOME=/mnt/nas/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 \
        python -m pytest -m gpu -s tests/test_gpu_parity.py
"""
import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

MODELS = os.environ.get("VLASERVE_PARITY_MODELS", "pi05-libero,keti-pi05-openarm-handover").split(",")
DEVICE = os.environ.get("VLASERVE_TEST_DEVICE", "cuda")


@pytest.fixture(scope="module", params=MODELS)
def loaded(request):
    from vlaserve.backends import load_backend
    from vlaserve.registry import resolve
    r = resolve(request.param)
    b = load_backend(r.entry.backend, r.path, DEVICE)
    print(f"\n[{request.param}] loaded in {b.load_seconds:.1f}s")
    yield request.param, b
    b.close()


def _observation(spec, rng, task):
    images = {}
    for cam, shape in spec.cameras.items():
        assert shape, f"{cam}: no declared shape; LeRobot's reference would resize to a stale one"
        c, h, w = shape
        images[cam] = rng.integers(0, 256, (h, w, c), dtype=np.uint8)
    state = rng.standard_normal(spec.state_dim).astype(np.float32) if spec.state_dim else None
    return images, state


def _reference_chunk(backend, images, state, task, seed):
    import torch
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.utils.feature_utils import hw_to_dataset_features

    spec = backend.spec
    hw = {f"joint_{i}.pos": float for i in range(spec.state_dim or 0)}
    hw.update({cam: img.shape for cam, img in images.items()})
    srv = PolicyServer.__new__(PolicyServer)        # skip gRPC setup; we only need the method
    srv.policy, srv.preprocessor, srv.postprocessor = backend.policy, backend.pre, backend.post
    srv.lerobot_features = hw_to_dataset_features(hw, "observation")
    srv.actions_per_chunk = spec.chunk_size
    srv.logger = logging.getLogger("lerobot-reference")
    srv.config = SimpleNamespace(environment_dt=0.05)
    raw = {f"joint_{i}.pos": float(v) for i, v in enumerate(state)} if state is not None else {}
    raw.update(images)
    raw["task"] = task
    torch.manual_seed(seed)
    timed = srv._predict_action_chunk(TimedObservation(timestamp=0.0, timestep=0, observation=raw))
    return torch.stack([t.get_action() for t in timed]).float().cpu().numpy()


@pytest.mark.parametrize("seed", [0, 1])
def test_chunk_matches_lerobot_policy_server(loaded, seed):
    from vlaserve.backends import Observation
    name, backend = loaded
    rng = np.random.default_rng(100 + seed)
    task = "put the red block in the bowl"
    images, state = _observation(backend.spec, rng, task)

    backend.reset()
    ref = _reference_chunk(backend, images, state, task, seed)
    backend.reset()
    ours = backend.predict(Observation(images=images, state=state, task=task), seed=seed)

    assert ours.actions.shape == ref.shape == (backend.spec.chunk_size, backend.spec.action_dim)
    diff = float(np.abs(ours.actions - ref).max())
    scale = float(np.abs(ref).max())
    print(f"[{name}] seed={seed} chunk={ref.shape} max|diff|={diff:.3e} (max|a|={scale:.3f}) "
          f"timings={ {k: round(v, 1) for k, v in ours.timings_ms.items()} }")
    # Same policy object, eager, same seed: expected bit-exact (both pi0.5 checkpoints are).
    assert diff <= 1e-5 * max(1.0, scale)


def test_seed_changes_the_chunk(loaded):
    """Flow-matching policies sample noise: different seeds must give different chunks, or the
    seed is not reaching the sampler and 'deterministic' runs are an illusion."""
    from vlaserve.backends import Observation
    _, backend = loaded
    rng = np.random.default_rng(7)
    images, state = _observation(backend.spec, rng, "t")
    obs = Observation(images=images, state=state, task="pick")
    a = backend.predict(obs, seed=0).actions
    b = backend.predict(obs, seed=0).actions
    c = backend.predict(obs, seed=1).actions
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
