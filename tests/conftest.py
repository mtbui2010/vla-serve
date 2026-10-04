import json
import struct
from pathlib import Path

import pytest


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    """A config.json-only LeRobot checkpoint: 2 cameras + 1 padding camera, 4-d state, 3-d action."""
    cfg = {
        "type": "fake",
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 48]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 16, 16]},
            "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 16, 16]},
            "observation.state": {"type": "STATE", "shape": [4]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [3]}},
        "chunk_size": 10,
        "n_action_steps": 8,
        "action_feature_names": ["a", "b", "gripper"],
        "license": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(
        {"name": "policy_preprocessor", "steps": [{"registry_name": "tokenizer_processor", "config": {}}]}))
    return tmp_path


def write_safetensors_header(path: Path, shapes: dict[str, list[int]]) -> None:
    """A .safetensors file whose header lists these float32 tensors (data zero-filled)."""
    meta, off = {}, 0
    for k, shp in shapes.items():
        n = 4
        for x in shp:
            n *= x
        meta[k] = {"dtype": "F32", "shape": shp, "data_offsets": [off, off + n]}
        off += n
    head = json.dumps(meta).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0" * off)


@pytest.fixture
def renamed_ckpt(tmp_path: Path) -> Path:
    """Like kaair_sim's open-drawer SmolVLA: config.json inherited from smolvla_base (camera1..3,
    6-d state), dataset cameras head/wrist and an 8-d state, bridged by a rename_map."""
    cfg = {
        "type": "smolvla",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [8]}},
        "chunk_size": 50, "n_action_steps": 50,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rename = {"observation.images.head": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}
    steps = [{"registry_name": "rename_observations_processor", "config": {"rename_map": rename}},
             {"registry_name": "tokenizer_processor", "config": {}},
             {"registry_name": "normalizer_processor", "config": {},
              "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors"}]
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    write_safetensors_header(tmp_path / steps[2]["state_file"], {
        "observation.state.mean": [8], "action.mean": [8], "frame_index.mean": [1],
        "observation.images.head.mean": [3, 1, 1], "observation.images.wrist.mean": [3, 1, 1]})
    return tmp_path
