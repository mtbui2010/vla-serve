"""Closed-loop test of a vla-serve server on kaair_sim's MuJoCo "open drawer" task.

Reuses kaair_sim's own evaluation (`/sim/scripts/eval_policy.py`: scenes, seeds, success
criterion, results file) unchanged, and swaps only the policy: instead of loading LeRobot in
process, every chunk comes from the vla-serve HTTP server. Run inside the kaair_sim container:

    # host: server in its container, checkpoint dir mounted read-only
    docker run -d --gpus '"device=3"' -p 127.0.0.1:11450:11450 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v /mnt/nas/lerobot_shared/kaair_sim/checkpoints:/ckpt:ro \
      --name vla-serve-kaair vla-serve:gpu-cu128 serve --host 0.0.0.0 \
      /ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model

    # kaair_sim_humble (network=host), client package copied to /tmp/vlaclient
    PYTHONPATH=/tmp/vlaclient:/tmp/vlaclient/deps /sim/.venv/bin/python \
      /tmp/vlaclient/kaair_sim_eval.py \
      --server-model /ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model \
      --ckpt /nas/checkpoints/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model \
      --task task_open_drawer_generic.yaml --tag vlaserve --videos 0 --n 50

Execution semantics match the in-process eval exactly: the sim is paused while the policy thinks
(synchronous), and each chunk's first `n_action_steps` actions are executed one per control step
before the next request — what LeRobot's `select_action` does with its internal queue. Wall-clock
latency therefore cannot change the outcome here; the test is of the served policy itself.
"""
import os
import sys
import time

import numpy as np
import torch

SIM = os.environ.get("SIM_ROOT", "/sim")
sys.path.insert(0, os.path.join(SIM, "scripts"))
import eval_policy  # noqa: E402

from vlaserve.client import PolicyClient  # noqa: E402


class ServedPolicy:
    """Stands in for (policy, pre, post) inside eval_policy.run()."""

    def __init__(self, client: PolicyClient):
        self.client = client
        self.queue: list[np.ndarray] = []
        self.calls = 0
        self.roundtrip_ms: list[float] = []

    # policy.reset() at the start of every episode
    def reset(self):
        self.queue.clear()
        self.client.reset()

    # pre(obs): eval_policy builds LeRobot-style tensors; turn them back into what a robot has
    def pre(self, obs):
        def hwc(t):
            return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return {"images": {"head": hwc(obs["observation.images.head"]),
                           "wrist": hwc(obs["observation.images.wrist"])},
                "state": obs["observation.state"].numpy(),
                "task": obs["task"]}

    def select_action(self, obs):
        if not self.queue:
            chunk = self.client.act(**obs)
            n = chunk.n_action_steps or len(chunk.actions)
            self.queue = list(chunk.actions[:n])
            self.calls += 1
            self.roundtrip_ms.append(chunk.roundtrip_ms)
        return torch.from_numpy(self.queue.pop(0)).unsqueeze(0)

    @staticmethod
    def post(action):
        return action


def main():
    url = "http://127.0.0.1:11450"
    server_model = None
    argv = []
    it = iter(sys.argv[1:])
    for a in it:                      # strip our two options, pass the rest to eval_policy
        if a == "--url":
            url = next(it)
        elif a == "--server-model":
            server_model = next(it)
        else:
            argv.append(a)
    if not server_model:
        sys.exit("--server-model is required (the model name/path as the server sees it)")

    client = PolicyClient(url, model=server_model, timeout=120)
    info = client.load()
    print(f"server: {info['model']} on {info['device']}, cameras {info['spec']['required_cameras']}, "
          f"state {info['spec']['state_dim']}, licence {info['license']}", flush=True)
    served = ServedPolicy(client)
    eval_policy.load_policy = lambda ckpt, device: (served, served.pre, served.post)
    sys.argv = [sys.argv[0]] + argv
    t0 = time.time()
    eval_policy.main()
    rt = np.array(served.roundtrip_ms)
    print(f"vla-serve: {served.calls} chunk requests, round trip median {np.median(rt):.1f} ms, "
          f"p95 {np.percentile(rt, 95):.1f} ms, wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
