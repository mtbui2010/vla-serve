"""Serve any LeRobot-format policy checkpoint.

The checkpoint is self-describing: `config.json` names the policy class, and
`policy_preprocessor.json` / `policy_postprocessor.json` carry tokenisation, normalisation
statistics and relative/absolute action conversion. We load all three through LeRobot's own
factories and reproduce the inference path of LeRobot's async `policy_server.py`
(`_predict_action_chunk`) step for step, so a chunk served here equals a chunk served there.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .. import compat
from ..spec import IMAGE_PREFIX, STATE_KEY, PolicySpec
from .base import ActionChunk, Backend, Observation

log = logging.getLogger(__name__)


class LeRobotBackend(Backend):
    def __init__(self, path: Path, device: str = "cuda", compile: bool = False):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        added = compat.install()
        if added:
            log.info("compat: registered legacy processor names %s", added)

        self.spec = PolicySpec.from_checkpoint(path)
        self._torch = torch
        self.device = device

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        # Eager by default, whatever the checkpoint says. Some checkpoints ship
        # compile_model=true + max-autotune (lerobot/pi05_libero_finetuned): the first call then
        # takes ~46 s, and autotuning picks kernels by benchmark, so two loads of the same
        # checkpoint differ by ~1e-3 in bf16. Opt in per model with `options: {compile: true}`.
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = compile
        t0 = time.perf_counter()
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        self.load_seconds = time.perf_counter() - t0
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    # -- helpers ---------------------------------------------------------------------------
    def _image(self, img: np.ndarray, chw: tuple[int, int, int]):
        """HWC uint8 -> 1xCxHxW float in [0,1], resized to the declared resolution.

        Same as LeRobot's `resize_robot_observation_image` + `prepare_image`: bilinear,
        align_corners=False, applied to the uint8 tensor and only then scaled to [0,1].
        Resizing in float instead differs by up to 1 grey level, so the order matters for parity.
        """
        torch = self._torch
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
        if tuple(t.shape[-2:]) != tuple(chw[1:]):
            t = torch.nn.functional.interpolate(t, size=chw[1:], mode="bilinear", align_corners=False)
        return (t.float() / 255.0).contiguous()

    def _batch(self, obs: Observation) -> dict:
        torch = self._torch
        batch = {f"{IMAGE_PREFIX}{name}": self._image(img, self.spec.cameras[name])
                 for name, img in obs.images.items()}
        if obs.state is not None:
            batch[STATE_KEY] = torch.as_tensor(np.asarray(obs.state, dtype=np.float32)).unsqueeze(0)
        if obs.task is not None:
            batch["task"] = obs.task
        return batch

    # -- Backend ---------------------------------------------------------------------------
    def predict(self, obs: Observation, seed: int | None = None) -> ActionChunk:
        torch = self._torch
        self.spec.validate(obs.images, obs.state, obs.task)
        t0 = time.perf_counter()
        batch = self.pre(self._batch(obs))
        t1 = time.perf_counter()
        with torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            chunk = self.policy.predict_action_chunk(batch)          # (1, H, D), normalised
            if chunk.is_cuda:
                torch.cuda.synchronize(chunk.device)
            t2 = time.perf_counter()
            # The postprocessor is written for (B, D); LeRobot's server applies it per step.
            steps = [self.post(chunk[:, i, :]) for i in range(chunk.shape[1])]
            actions = torch.stack(steps, dim=1).squeeze(0).float().cpu().numpy()
        t3 = time.perf_counter()
        return ActionChunk(actions=actions, timings_ms={
            "preprocess": 1e3 * (t1 - t0), "inference": 1e3 * (t2 - t1),
            "postprocess": 1e3 * (t3 - t2), "total": 1e3 * (t3 - t0)})

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def close(self) -> None:
        del self.policy
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
