import json

import msgpack
import numpy as np
import pytest

from vlaserve import codec
from vlaserve.errors import ObservationError, UnsupportedCheckpoint
from vlaserve.licenses import resolve as resolve_license
from vlaserve.spec import PolicySpec


def obs_images():
    return {"front": np.zeros((32, 48, 3), np.uint8), "wrist": np.zeros((100, 80, 3), np.uint8)}


def test_spec_from_checkpoint(ckpt):
    s = PolicySpec.from_checkpoint(ckpt)                 # no stats: falls back to config.json
    assert s.source == "config"
    assert s.cameras == {"front": (3, 32, 48), "wrist": (3, 16, 16)}
    assert s.policy_fills == ["empty_camera_0"]
    assert (s.state_dim, s.action_dim, s.chunk_size, s.n_action_steps) == (4, 3, 10, 8)
    assert s.language is True


def test_spec_prefers_dataset_stats_over_stale_config(renamed_ckpt):
    s = PolicySpec.from_checkpoint(renamed_ckpt)
    assert s.source == "stats"
    assert s.cameras == {"head": None, "wrist": None}     # dataset names; base model's 256x256 dropped
    assert s.state_dim == 8 and s.action_dim == 8          # config.json says 6
    assert s.policy_fills == ["camera3", "empty_camera_0"]
    s.validate({"head": np.zeros((480, 848, 3), np.uint8), "wrist": np.zeros((480, 848, 3), np.uint8)},
               np.zeros(8, np.float32), "open drawer")
    with pytest.raises(ObservationError, match="unknown camera"):
        s.validate({"head": np.zeros((4, 4, 3), np.uint8), "wrist": np.zeros((4, 4, 3), np.uint8),
                    "camera1": np.zeros((4, 4, 3), np.uint8)}, np.zeros(8, np.float32), "t")


def test_safetensors_header_rejects_implausible_length(tmp_path):
    import struct
    from vlaserve.spec import safetensors_shapes
    (tmp_path / "x.safetensors").write_bytes(struct.pack("<Q", 1 << 40))
    with pytest.raises(UnsupportedCheckpoint):
        safetensors_shapes(tmp_path / "x.safetensors")


def test_spec_dict_roundtrip(ckpt):
    s = PolicySpec.from_checkpoint(ckpt)
    assert PolicySpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def test_spec_rejects_non_lerobot_dir(tmp_path):
    with pytest.raises(UnsupportedCheckpoint):
        PolicySpec.from_checkpoint(tmp_path)


def test_validate_accepts_any_resolution_without_padding_camera(ckpt):
    PolicySpec.from_checkpoint(ckpt).validate(obs_images(), np.zeros(4, np.float32), "do it")


@pytest.mark.parametrize("mutate, msg", [
    (lambda i, s, t: i.pop("wrist"), "missing camera"),
    (lambda i, s, t: i.__setitem__("top", np.zeros((8, 8, 3), np.uint8)), "unknown camera"),
    (lambda i, s, t: i.__setitem__("front", np.zeros((8, 8), np.uint8)), "HxWx3"),
    (lambda i, s, t: i.__setitem__("front", np.zeros((8, 8, 3), np.float32)), "uint8"),
    (lambda i, s, t: s.__setitem__(0, np.nan), "NaN"),
])
def test_validate_rejects(ckpt, mutate, msg):
    spec = PolicySpec.from_checkpoint(ckpt)
    images, state = obs_images(), np.zeros(4, np.float32)
    mutate(images, state, None)
    with pytest.raises(ObservationError, match=msg):
        spec.validate(images, state, "task")


def test_validate_needs_state_shape_and_task(ckpt):
    spec = PolicySpec.from_checkpoint(ckpt)
    with pytest.raises(ObservationError, match="state"):
        spec.validate(obs_images(), np.zeros(5, np.float32), "t")
    with pytest.raises(ObservationError, match="task"):
        spec.validate(obs_images(), np.zeros(4, np.float32), "")


def test_codec_roundtrip_is_bit_exact():
    img = np.random.default_rng(0).integers(0, 256, (7, 5, 3), dtype=np.uint8)
    act = np.random.default_rng(1).standard_normal((50, 7)).astype(np.float32)
    out = codec.unpackb(codec.packb({"img": img, "act": act, "n": np.int64(3)}))
    assert np.array_equal(out["img"], img) and out["img"].dtype == np.uint8
    assert np.array_equal(out["act"], act) and out["n"] == 3


@pytest.mark.parametrize("bad, msg", [
    ({"__nd__": True, "dtype": "uint8", "shape": [2, 2], "data": b"\0" * 3}, "needs 4 bytes"),
    ({"__nd__": True, "dtype": "object", "shape": [1], "data": b"\0"}, "not allowed"),
    ({"__nd__": True, "dtype": "uint8", "shape": [-1], "data": b""}, "bad array shape"),
    ({"__nd__": True, "dtype": "uint8", "shape": [1, 1, 1, 1, 1], "data": b"\0"}, "bad array shape"),
])
def test_codec_rejects_malformed_arrays(bad, msg):
    with pytest.raises(ObservationError, match=msg):
        codec.unpackb(msgpack.packb({"x": bad}, use_bin_type=True))


def test_codec_rejects_garbage():
    with pytest.raises(ObservationError):
        codec.unpackb(b"\xc1\xc1\xc1")


def test_license_inference():
    assert resolve_license({"paligemma_variant": "gemma_2b"}, None).license == "gemma-terms"
    assert resolve_license({}, None).license == "unknown"
    assert resolve_license({}, None).warning
    assert resolve_license({}, "MIT").warning is None
    assert resolve_license({"license": "apache-2.0"}, None).source == "checkpoint"
    # A permissive registry claim on a PaliGemma fine-tune is flagged, not trusted silently.
    lic = resolve_license({"paligemma_variant": "gemma_2b"}, "apache-2.0")
    assert lic.permissive and "gemma-terms" in lic.warning
