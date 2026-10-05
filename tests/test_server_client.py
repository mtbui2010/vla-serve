"""Server + client against a fake backend: the HTTP contract without a GPU."""
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vlaserve.backends.base import ActionChunk, Backend
from vlaserve.client import ActionQueue, ChunkExecutor, PolicyClient, ServerError
from vlaserve.licenses import LicenseInfo
from vlaserve.registry import Entry, ResolvedModel
from vlaserve.server import ModelManager, create_app
from vlaserve.spec import PolicySpec


class FakeBackend(Backend):
    """Action i of a chunk = state[0] + i, so tests can see which observation produced it."""

    def __init__(self, spec, delay=0.0):
        self.spec, self.delay, self.resets, self.closed = spec, delay, 0, False

    def predict(self, obs, seed=None):
        self.spec.validate(obs.images, obs.state, obs.task)
        self.last = obs
        time.sleep(self.delay)
        h = self.spec.chunk_size
        a = np.repeat((obs.state[0] + np.arange(h, dtype=np.float32))[:, None], self.spec.action_dim, 1)
        return ActionChunk(actions=a, timings_ms={"total": 1e3 * self.delay})

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


@pytest.fixture
def app_and_client(ckpt):
    spec = PolicySpec.from_checkpoint(ckpt)
    made = {}

    def resolver(name):
        return ResolvedModel(name, Path(ckpt), Entry(name, str(ckpt)), LicenseInfo("unknown", "none"))

    def factory(resolved, device):
        made[resolved.name] = FakeBackend(spec, delay=0.0)
        return made[resolved.name]

    manager = ModelManager(default_device="cpu", factory=factory, resolver=resolver)
    http = TestClient(create_app(manager))
    client = PolicyClient("http://testserver", model="m")
    client._http = http
    return manager, client, made


def test_client_resizes_to_network_input_and_sends_jpeg(ckpt):
    """A SmolVLA-like checkpoint: frames arrive resized-with-pad to the declared input."""
    import json
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg.update(type="smolvla", resize_imgs_with_padding=[40, 30])      # (W, H) as SmolVLA stores it
    (ckpt / "config.json").write_text(json.dumps(cfg))
    spec = PolicySpec.from_checkpoint(ckpt)
    made = {}

    def resolver(name):
        return ResolvedModel(name, Path(ckpt), Entry(name, str(ckpt)), LicenseInfo("unknown", "none"))

    def factory(resolved, device):
        made["b"] = FakeBackend(spec)
        return made["b"]

    http = TestClient(create_app(ModelManager("cpu", factory=factory, resolver=resolver)))
    frames = {"front": np.full((60, 120, 3), 200, np.uint8), "wrist": np.full((90, 60, 3), 90, np.uint8)}

    client = PolicyClient("http://testserver", model="m"); client._http = http
    client.load()
    client.act(frames, state=[0, 0, 0, 0], task="t")
    got = made["b"].last.images
    assert got["front"].shape == got["wrist"].shape == (30, 40, 3)
    assert got["front"][:8, :, 0].max() <= 3           # 60x120 -> 20x40 + 10 black rows on top (JPEG edge: 2)
    assert abs(int(got["front"][20, 20, 0]) - 200) <= 3  # JPEG: close, not exact

    raw = PolicyClient("http://testserver", model="m", resize="off", encoding="raw"); raw._http = http
    raw.act(frames, state=[0, 0, 0, 0], task="t")
    got = made["b"].last.images
    assert got["front"].shape == (60, 120, 3) and np.array_equal(got["front"], frames["front"])

    with pytest.raises(ValueError):
        PolicyClient(encoding="png")


def images():
    return {"front": np.zeros((32, 48, 3), np.uint8), "wrist": np.zeros((16, 16, 3), np.uint8)}


def test_act_roundtrip(app_and_client):
    _, client, _ = app_and_client
    with pytest.raises(ServerError) as e:
        client.act(images(), state=[0, 0, 0, 0], task="t")
    assert e.value.status == 409                     # not loaded yet
    info = client.load()
    assert info["license_warning"] and info["spec"]["action_dim"] == 3
    chunk = client.act(images(), state=[5, 0, 0, 0], task="t")
    assert chunk.actions.shape == (10, 3) and chunk.actions[0, 0] == 5 and chunk.actions[9, 0] == 14
    assert chunk.n_action_steps == 8 and chunk.action_names == ["a", "b", "gripper"]


def test_bad_observation_is_400_not_500(app_and_client):
    _, client, _ = app_and_client
    client.load()
    with pytest.raises(ServerError) as e:
        client.act({"front": np.zeros((32, 48, 3), np.uint8)}, state=[0, 0, 0, 0], task="t")
    assert e.value.status == 400 and "wrist" in str(e.value)
    r = client._http.post("/v1/act", content=b"not msgpack")
    assert r.status_code == 400


def test_reset_unload_and_models(app_and_client):
    _, client, made = app_and_client
    client.load()
    client.reset()
    assert made["m"].resets == 1
    assert client._http.get("/v1/models").json()["models"][0]["model"] == "m"
    assert client._http.post("/v1/models/unload", json={"model": "m"}).status_code == 200
    assert made["m"].closed
    assert client._http.post("/v1/models/unload", json={"model": "m"}).status_code == 409


# -- ActionQueue / ChunkExecutor ---------------------------------------------------------------

def _executed(q, n):
    for _ in range(n):
        q.pop()


def test_queue_drops_actions_executed_during_inference():
    q = ActionQueue(execute_steps=8)
    q.on_chunk(np.arange(10.0)[:, None], executed_at_obs=0)
    _executed(q, 3)                                   # robot moves 3 steps while next chunk computes
    q.on_chunk(100 + np.arange(10.0)[:, None], executed_at_obs=0)
    assert len(q) == 5 and q.pop()[0] == 103 and q.dropped_stale == 3


def test_queue_idle_time_does_not_age_the_plan():
    """Robot idle while waiting (first chunk, stalls): the observation is still current."""
    q = ActionQueue(execute_steps=8)
    q.on_chunk(np.arange(10.0)[:, None], executed_at_obs=0)
    assert len(q) == 8 and q.pop()[0] == 0 and q.dropped_stale == 0


def test_queue_chunk_older_than_its_whole_length_leaves_nothing():
    q = ActionQueue(execute_steps=4)
    q.on_chunk(np.arange(10.0)[:, None], 0)
    _executed(q, 4)
    q.on_chunk(np.arange(10.0)[:, None], 0)
    assert len(q) == 0 and q.pop() is None


class FakeClient:
    """Chunk computed from the step at which the observation was taken, with a fixed latency."""

    def __init__(self, latency, h=10):
        self.latency, self.h, self.calls = latency, h, 0

    def act(self, images=None, state=None, task=None):
        self.calls += 1
        time.sleep(self.latency)
        return type("C", (), {"actions": (state[0] + np.arange(self.h, dtype=float))[:, None],
                              "n_action_steps": 8, "roundtrip_ms": 1e3 * self.latency})()


def test_executor_sends_time_consistent_actions():
    """The k-th action sent must be the one planned for the k-th move (action value == k)."""
    sent = []
    # The observation carries how many actions have run; the fake policy plans value == move index.
    ex = ChunkExecutor(FakeClient(latency=0.03),
                       get_observation=lambda: {"state": [len(sent)]},
                       send_action=lambda a: sent.append((len(sent), a[0])), fps=100.0)
    ex.run(task="t", seconds=0.6)

    assert ex.stats.chunks >= 3 and ex.stats.dropped_stale > 0
    # +-1: an action may be sent between reading the counter and taking the observation.
    # Replaying stale actions instead would be off by latency x fps = 3.
    errors = [abs(k - v) for k, v in sent]
    assert sent and max(errors) <= 1, f"max error {max(errors)}"


def test_executor_slow_first_chunk_is_executed_not_discarded():
    sent = []
    client = FakeClient(latency=0.15)                 # 15 ticks at 100 Hz before anything arrives
    ex = ChunkExecutor(client, get_observation=lambda: {"state": [len(sent)]},
                       send_action=lambda a: sent.append(a[0]), fps=100.0, refill_at=0)
    ex.run(seconds=0.25)
    assert ex.stats.startup_ticks >= 10
    assert sent[:5] == [0, 1, 2, 3, 4] and ex.stats.dropped_stale == 0


def test_executor_surfaces_inference_errors():
    class Boom:
        def act(self, **kw):
            raise ServerError(400, "bad camera")
    ex = ChunkExecutor(Boom(), get_observation=lambda: {}, send_action=lambda a: None, fps=50)
    with pytest.raises(ServerError, match="bad camera"):
        ex.run(seconds=1.0)


def test_executor_rejects_bad_fps():
    with pytest.raises(ValueError):
        ChunkExecutor(None, lambda: {}, lambda a: None, fps=0)
