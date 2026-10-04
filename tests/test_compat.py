"""The legacy-name shim must load an old pipeline JSON without changing how new ones serialise."""
import json

import torch

from vlaserve import compat


def _pipeline(tmp_path, name, steps):
    (tmp_path / f"{name}.json").write_text(json.dumps({"name": name, "steps": steps}))


def test_delta_actions_loads_as_relative_and_pairs_with_absolute(tmp_path):
    from lerobot.policies.factory import _reconnect_relative_absolute_steps
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.relative_action_processor import (AbsoluteActionsProcessorStep,
                                                             RelativeActionsProcessorStep)

    compat.install()
    cfg = {"enabled": True, "exclude_joints": ["gripper"], "action_names": ["j1", "j2", "gripper"]}
    _pipeline(tmp_path, "pre", [{"registry_name": "delta_actions_processor", "config": cfg}])
    _pipeline(tmp_path, "post", [{"registry_name": "absolute_actions_processor", "config": {"enabled": True}}])
    pre = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="pre.json")
    post = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="post.json")
    step = pre.steps[0]
    assert isinstance(step, RelativeActionsProcessorStep)
    assert step.exclude_joints == ["gripper"] and step.action_names == ["j1", "j2", "gripper"]

    _reconnect_relative_absolute_steps(pre, post)
    absolute = post.steps[0]
    assert isinstance(absolute, AbsoluteActionsProcessorStep) and absolute.relative_step is step

    # The renamed class still serialises under its own, new name.
    assert RelativeActionsProcessorStep._registry_name == "relative_actions_processor"


def test_install_is_idempotent():
    compat.install()
    assert compat.install() == []
