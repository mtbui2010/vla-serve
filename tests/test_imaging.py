"""Client-side resize must equal the policy's own resize; JPEG must be bounded and round-trip.

The references are LeRobot's functions themselves (common/vla_utils), run on float [0,1] frames
exactly as the policies run them. Ours works on uint8 and rounds, so it may differ by < 0.5 grey
level per pixel, never more.
"""
import json

import numpy as np
import pytest
import torch

from vlaserve import codec, imaging
from vlaserve.errors import ObservationError
from vlaserve.spec import PolicySpec, image_input_rule

SHAPES = [(480, 848), (720, 1280), (256, 256), (100, 700), (700, 100), (37, 53), (480, 640)]


def frame(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def torch_float(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0


@pytest.mark.parametrize("h,w", SHAPES)
@pytest.mark.parametrize("th,tw", [(224, 224), (512, 512), (300, 400)])
def test_bilinear_matches_torch(h, w, th, tw):
    img = frame(h, w)
    ref = torch.nn.functional.interpolate(torch_float(img), size=(th, tw), mode="bilinear",
                                          align_corners=False)
    ref = (ref[0].permute(1, 2, 0).numpy() * 255.0)
    ours = imaging.bilinear(img, th, tw).astype(np.float64)
    assert np.abs(ours - ref).max() <= 0.5 + 1e-3


@pytest.mark.parametrize("h,w", SHAPES)
def test_smolvla_resize_with_pad_matches_lerobot(h, w):
    from lerobot.policies.common.vla_utils import resize_with_pad
    img = frame(h, w, 1)
    ref = resize_with_pad(torch_float(img), 512, 512, pad_value=0)[0].permute(1, 2, 0).numpy() * 255.0
    ours = imaging.resize_with_pad(img, 512, 512, pad="top_left")
    assert ours.shape == ref.shape
    assert np.abs(ours.astype(np.float64) - ref).max() <= 0.5 + 1e-3
    # what the policy then does to it: nothing (same size -> returned unchanged)
    again = resize_with_pad(torch_float(ours), 512, 512, pad_value=0)
    assert torch.equal(again, torch_float(ours))


@pytest.mark.parametrize("h,w", SHAPES)
def test_pi_resize_with_pad_matches_lerobot(h, w):
    from lerobot.policies.common.vla_utils import resize_with_pad_torch
    img = frame(h, w, 2)
    ref = resize_with_pad_torch(torch_float(img), 224, 224)[0].permute(1, 2, 0).numpy() * 255.0
    ours = imaging.resize_with_pad(img, 224, 224, pad="center")
    assert ours.shape == ref.shape
    assert np.abs(ours.astype(np.float64) - ref).max() <= 0.5 + 1e-3


def test_image_input_rules():
    assert image_input_rule({"type": "smolvla", "resize_imgs_with_padding": [512, 384]}) == \
        {"size": [384, 512], "pad": "top_left"}            # SmolVLA stores (W, H)
    assert image_input_rule({"type": "xvla", "resize_imgs_with_padding": [384, 512]}) == \
        {"size": [384, 512], "pad": "top_left"}            # X-VLA passes (H, W)
    assert image_input_rule({"type": "pi05", "image_resolution": [224, 224]}) == \
        {"size": [224, 224], "pad": "center"}
    assert image_input_rule({"type": "act"}) is None


def test_spec_carries_rule_and_old_servers_are_tolerated(renamed_ckpt):
    cfg = json.loads((renamed_ckpt / "config.json").read_text())
    cfg["resize_imgs_with_padding"] = [512, 512]
    (renamed_ckpt / "config.json").write_text(json.dumps(cfg))
    spec = PolicySpec.from_checkpoint(renamed_ckpt)
    assert spec.image_input == {"size": [512, 512], "pad": "top_left"}
    d = spec.to_dict(); d.pop("image_input"); d["something_new"] = 1
    assert PolicySpec.from_dict(d).image_input is None


def test_jpeg_roundtrip_through_codec():
    img = frame(48, 64)
    body = codec.packb({"images": {"head": codec.jpeg(imaging.encode_jpeg(img, 90))}})
    out = codec.unpackb(body)["images"]["head"]
    assert out.shape == (48, 64, 3) and out.dtype == np.uint8 and out.flags.writeable


def test_jpeg_bad_and_oversized_are_400(monkeypatch):
    bad = codec.packb({"x": codec.jpeg(b"not a jpeg")})
    with pytest.raises(ObservationError, match="bad JPEG"):
        codec.unpackb(bad)
    monkeypatch.setattr(imaging, "MAX_SIDE", 32)
    big = codec.packb({"x": codec.jpeg(imaging.encode_jpeg(frame(40, 40)))})
    with pytest.raises(ObservationError, match="exceeds"):
        codec.unpackb(big)
