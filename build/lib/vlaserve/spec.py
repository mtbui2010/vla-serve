"""What a policy consumes and produces, read from its checkpoint's config.json alone.

Reading the spec never imports torch or touches the weights, so `vla-serve inspect` is instant
even for an 8 GB checkpoint on a network drive.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ObservationError, UnsupportedCheckpoint

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
# LeRobot's convention for padding cameras: the policy fills a missing one with a blank image
# (pi05's `_preprocess_images`), so a client never has to send it.
PADDING_CAMERA_PREFIX = "empty_camera"


@dataclass
class PolicySpec:
    policy_type: str
    cameras: dict[str, tuple[int, int, int]]  # short name -> (C, H, W) the checkpoint declares
    state_dim: int | None
    action_dim: int
    chunk_size: int | None
    n_action_steps: int | None
    action_names: list[str] | None = None
    padding_cameras: list[str] = field(default_factory=list)
    language: bool = True  # whether the processor pipeline tokenises a task string
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def required_cameras(self) -> list[str]:
        return [c for c in self.cameras if c not in self.padding_cameras]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cameras"] = {k: list(v) for k, v in self.cameras.items()}
        d["required_cameras"] = self.required_cameras
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicySpec":
        """Inverse of to_dict (what GET /v1/spec returns), for clients."""
        d = {k: v for k, v in d.items() if k != "required_cameras"}
        d["cameras"] = {k: tuple(v) for k, v in d["cameras"].items()}
        return cls(**d)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PolicySpec":
        path = Path(path)
        cfg_file = path / "config.json"
        if not cfg_file.is_file():
            raise UnsupportedCheckpoint(f"{path} has no config.json; not a LeRobot policy checkpoint")
        cfg = json.loads(cfg_file.read_text())
        if "type" not in cfg or "input_features" not in cfg or "output_features" not in cfg:
            raise UnsupportedCheckpoint(f"{cfg_file} lacks type/input_features/output_features")

        cameras, padding, state_dim = {}, [], None
        for key, feat in cfg["input_features"].items():
            if key.startswith(IMAGE_PREFIX):
                name = key[len(IMAGE_PREFIX):]
                cameras[name] = tuple(int(x) for x in feat["shape"])
                if name.startswith(PADDING_CAMERA_PREFIX):
                    padding.append(name)
            elif key == STATE_KEY:
                state_dim = int(feat["shape"][0])
        if "action" not in cfg["output_features"]:
            raise UnsupportedCheckpoint(f"{cfg_file}: no 'action' output feature")
        action_dim = int(cfg["output_features"]["action"]["shape"][0])

        language = False
        pre = path / "policy_preprocessor.json"
        if pre.is_file():
            steps = json.loads(pre.read_text()).get("steps", [])
            language = any("tokenizer" in s.get("registry_name", "") for s in steps)

        extra = {k: cfg[k] for k in ("paligemma_variant", "vlm_model_name", "dtype",
                                     "num_inference_steps", "use_relative_actions") if k in cfg}
        return cls(
            policy_type=cfg["type"],
            cameras=cameras,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=cfg.get("chunk_size"),
            n_action_steps=cfg.get("n_action_steps"),
            action_names=cfg.get("action_feature_names"),
            padding_cameras=padding,
            language=language,
            extra=extra,
        )

    # -- observation validation ------------------------------------------------------------
    def validate(self, images: dict[str, np.ndarray], state: np.ndarray | None,
                 task: str | None) -> None:
        """Reject an observation the policy cannot consume, before it reaches the GPU.

        Images are HWC uint8 (what a camera driver gives you). Their resolution may differ from
        the declared one; the backend resizes, as LeRobot's own policy server does.
        """
        missing = [c for c in self.required_cameras if c not in images]
        if missing:
            raise ObservationError(f"missing camera(s) {missing}; policy expects "
                                   f"{self.required_cameras}")
        unknown = [c for c in images if c not in self.cameras]
        if unknown:
            raise ObservationError(f"unknown camera(s) {unknown}; policy expects {list(self.cameras)}")
        for name, img in images.items():
            if img.ndim != 3 or img.shape[2] != self.cameras[name][0]:
                raise ObservationError(f"camera '{name}': expected HxWx{self.cameras[name][0]} "
                                       f"image, got shape {tuple(img.shape)}")
            if img.shape[0] > 8192 or img.shape[1] > 8192:
                raise ObservationError(f"camera '{name}': {img.shape[:2]} exceeds 8192 px")
            if img.dtype != np.uint8:
                raise ObservationError(f"camera '{name}': expected uint8, got {img.dtype}")
        if self.state_dim is not None:
            if state is None:
                raise ObservationError(f"policy needs a state vector of length {self.state_dim}")
            if state.shape != (self.state_dim,):
                raise ObservationError(f"state: expected shape ({self.state_dim},), got {state.shape}")
            if not all(math.isfinite(float(x)) for x in state):
                raise ObservationError("state contains NaN or Inf")
        if self.language and not task:
            raise ObservationError("policy is language-conditioned; send a non-empty 'task'")
