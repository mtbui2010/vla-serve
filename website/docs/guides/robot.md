# Drive a robot

## 1. Describe your robot in a profile

A **policy** says what it was trained on. A **profile** says what *your* robot produces and
under which names. Keeping them separate lets one robot run many policies and one policy be
checked against many robot setups.

```yaml title="kaair.yaml"
name: kaair
fps: 30
cameras:                     # policy camera name -> your driver's camera name
  head: femto_color
  wrist: d405_color
state_dim: 8
action_dim: 8
```

## 2. Check before the robot moves

```bash
make check MODEL=/ckpt/.../pretrained_model PROFILE=kaair.yaml
```

```text
2 problem(s) between profile 'kaair' and ...:
  - policy needs camera(s) ['wrist']; profile maps ['head', 'top']
  - state_dim: robot 7, policy 8
```

These are the mistakes that make a working policy fail on a new setup **without any error**: a
renamed camera, a missing gripper dimension, joints in a different order (checked when both
sides list `action_names`).

## 3. Run the control loop

```python
from vlaserve.client import PolicyClient, ChunkExecutor
from vlaserve.profiles import RobotProfile
from vlaserve.spec import PolicySpec

client = PolicyClient("http://gpu-box:11450", model=MODEL)
client.load()                                       # no-op if already loaded
spec = PolicySpec.from_dict(client.spec()["spec"])
profile = RobotProfile.load("kaair.yaml")
profile.require(spec)                               # raises on any mismatch

def read_robot():
    frames = {"femto_color": femto.read(), "d405_color": d405.read()}   # H×W×3 uint8 RGB
    return {"images": profile.to_policy_images(frames), "state": arm.joint_positions()}

client.reset()                                      # every new episode
stats = ChunkExecutor(client, get_observation=read_robot,
                      send_action=arm.command_position, fps=profile.fps).run(
    task="open drawer", seconds=60)
print(stats.summary())
```

`stats.summary()` tells you whether the loop kept up:

| field | meaning | healthy value |
|---|---|---|
| `stalls` | ticks with no action available after the first chunk | 0 |
| `startup_ticks` | ticks spent waiting for the very first chunk | ≈ latency × fps |
| `dropped_stale` | actions skipped because the robot had already moved past them | ≈ latency × fps per chunk |
| `overruns` | ticks where `send_action` itself took longer than one period | 0 |
| `roundtrip_ms_*` | request time including transfer | below chunk duration |

## 4. Tune the two knobs

* `execute_steps` — how many actions of each chunk to play (default: the checkpoint's
  `n_action_steps`). Fewer = more reactive, more requests.
* `refill_at` — request the next chunk when this many actions remain (default: half the chunk).
  It must exceed `latency × fps`, or the robot stalls waiting.

Example: 300 ms latency at 30 Hz is 9 ticks; with 30-step chunks the default `refill_at = 15`
leaves margin.

!!! warning "Safety"
    vla-serve does not limit velocities, workspace or forces. Keep your robot's own limits and
    an emergency stop. A policy that worked in simulation can still do something unexpected on
    hardware. Keep `fps` equal to the rate the training data was recorded at (the actions are
    spaced for it), and start with the robot controller's own speed scaling turned down.

The reasoning behind chunks, queues and stale actions is in
[Real-time control](../learn/06-real-time.md).
