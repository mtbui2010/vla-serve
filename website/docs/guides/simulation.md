# Test in simulation

Matching LeRobot number-for-number proves the *server* is right. It does not prove the *policy*
works on your setup. For that you need closed-loop runs: the policy acts, the world changes, the
policy sees the result. Simulation lets you do hundreds of those before risking hardware.

## The pattern

Keep your simulator's own evaluation loop — scenes, seeds, success criterion — and swap only the
policy for an HTTP call. Then "served vs in-process" is the only thing that differs.

```python
class ServedPolicy:
    """Stands in for (policy, pre, post) inside an existing LeRobot-style eval loop."""
    def __init__(self, client):
        self.client, self.queue = client, []

    def reset(self):                               # start of each episode
        self.queue.clear(); self.client.reset()

    def select_action(self, obs):                  # same semantics as LeRobot's select_action
        if not self.queue:
            chunk = self.client.act(**obs)
            self.queue = list(chunk.actions[: chunk.n_action_steps])
        return self.queue.pop(0)
```

A simulator usually pauses physics while the policy thinks (synchronous), so latency cannot
change the outcome: what you measure is the served policy alone.

## Worked example: kaair_sim, open the drawer

[`examples/kaair_sim_eval.py`](https://github.com/mtbui2010/vla-serve/blob/main/examples/kaair_sim_eval.py)
does exactly this for a MuJoCo kitchen-cabinet task with a SmolVLA policy.

```bash
make docker-run GPU=3 CKPT_ROOT=/mnt/nas/lerobot_shared/kaair_sim/checkpoints \
  MODEL=/ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model
make kaair-eval N=50
```

!!! measured "Measured (2026-10-04)"
    | run | drawers opened |
    |---|---|
    | through vla-serve | **47 / 50** (94 %) |
    | in-process LeRobot, same scenes | 49 / 50 (98 %) |

    The difference is within noise (Fisher exact p = 0.62). The hardest scene (seed 4) failed
    **5/5** in-process when re-run alone, and 4/5 through the server: a hard scene, not a
    serving error. Before running closed loop, served and in-process chunks were confirmed
    identical for the same observation and seed.

## Read results honestly

* **Compare against a baseline you ran today**, on the same scenes. A number from last month was
  produced by last month's simulator code.
* **Seed the policy's noise per episode** if you want paired comparisons. Flow-matching policies
  draw random noise for every chunk; if the evaluator does not seed it, an episode's outcome
  depends on how many random numbers the previous episodes consumed. See
  [Evaluating a policy](../learn/07-evaluation.md).
* **50 episodes cannot separate 94 % from 98 %.** Report counts and an interval, not just a
  percentage.
