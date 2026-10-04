import numpy as np
import pytest

from vlaserve.errors import ProfileMismatch
from vlaserve.profiles import RobotProfile
from vlaserve.spec import PolicySpec


def profile(**kw):
    base = dict(name="r", fps=10, cameras={"front": "cam0", "wrist": "cam1"}, state_dim=4, action_dim=3)
    base.update(kw)
    return RobotProfile(**base)


def test_compatible_profile(ckpt):
    assert profile().check(PolicySpec.from_checkpoint(ckpt)) == []


def test_every_mismatch_is_reported(ckpt):
    spec = PolicySpec.from_checkpoint(ckpt)
    problems = profile(cameras={"front": "cam0", "top": "cam2"}, state_dim=5, action_dim=2,
                       action_names=["b", "a", "gripper"]).check(spec)
    text = "\n".join(problems)
    for needle in ["needs camera(s) ['wrist']", "'top'", "state_dim", "action_dim", "joint order"]:
        assert needle in text
    with pytest.raises(ProfileMismatch):
        profile(state_dim=5).require(spec)


def test_bundled_profiles_load():
    assert RobotProfile.load("libero").cameras["image"] == "agentview_image"
    assert len(RobotProfile.load("openarm-bimanual").action_names) == 16


def test_to_policy_images_renames_and_reports_missing():
    p = profile()
    a = np.zeros((2, 2, 3), np.uint8)
    assert set(p.to_policy_images({"cam0": a, "cam1": a, "cam9": a})) == {"front", "wrist"}
    with pytest.raises(ProfileMismatch, match="cam1"):
        p.to_policy_images({"cam0": a})
