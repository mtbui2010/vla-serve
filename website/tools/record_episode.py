"""Record one kaair_sim open-drawer episode driven through vla-serve, for the docs figures.

Runs inside kaair_sim_humble. Same scene construction as eval_policy.main(); policy = HTTP.
Saves /tmp/docsfig/episode.npz: full-res first frames, every-3rd-step thumbnails, states,
executed actions, each chunk with the step it was requested at.
"""
import os, sys, time
import numpy as np, torch, yaml, mujoco

SIM = "/sim"
sys.path.insert(0, f"{SIM}/scripts")
import eval_policy as ev, check_sim as cs, drawer_ready as dr, randomize  # noqa: E402
from vlaserve.client import PolicyClient  # noqa: E402

MODEL = "/ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model"
SEED = int(os.environ.get("SEED", 10 ** 9))


class Recorder:
    def __init__(self, client):
        self.client, self.queue, self.step = client, [], 0
        self.states, self.actions, self.chunks, self.first = [], [], [], None

    def reset(self):
        self.queue.clear(); self.client.reset()

    def pre(self, obs):
        hwc = lambda t: (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        o = {"images": {"head": hwc(obs["observation.images.head"]), "wrist": hwc(obs["observation.images.wrist"])},
             "state": obs["observation.state"].numpy(), "task": obs["task"]}
        if self.first is None:
            self.first = o
        self.states.append(o["state"].copy())
        return o

    def select_action(self, o):
        if not self.queue:
            c = self.client.act(**o, seed=self.step)          # seeded: the figure is reproducible
            self.chunks.append((self.step, c.actions.copy(), c.timings_ms.get("total", np.nan), c.roundtrip_ms))
            self.queue = list(c.actions[: c.n_action_steps])
        a = self.queue.pop(0)
        self.actions.append(a.copy())
        self.step += 1
        return torch.from_numpy(a).unsqueeze(0)

    @staticmethod
    def post(a):
        return a


def main():
    cfg = yaml.safe_load(open(f"{SIM}/config/mjcf.yaml"))
    task = yaml.safe_load(open(f"{SIM}/config/task_open_drawer_generic.yaml"))
    rcfg = yaml.safe_load(open(f"{SIM}/config/randomization.yaml"))
    ref = cs.UrdfFK(f"{SIM}/urdf/raw/kaair_slamtec.urdf")
    tcp_T = ref.fk(dr.ready_q(task, task["mode"], cfg["home"]))["tool_tcp_link"]
    rng = np.random.default_rng(SEED)
    rand = randomize.sample(rng, rcfg)
    xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
    path = f"{SIM}/mjcf/_docsfig_scene_{os.getpid()}.xml"
    open(path, "w").write(xml)
    try:
        m = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    d = mujoco.MjData(m)
    rec = Recorder(PolicyClient(model=MODEL, timeout=120))
    frames = []
    t0 = time.time()
    r = ev.run(m, d, task, cfg, rand, rec, rec.pre, rec.post, 15.0, 0.20, frames, None, rng)
    print("result", r, "wall", round(time.time() - t0), "s", "steps", rec.step, "chunks", len(rec.chunks))
    os.makedirs("/tmp/docsfig", exist_ok=True)
    np.savez_compressed(
        "/tmp/docsfig/episode.npz",
        head0=rec.first["images"]["head"], wrist0=rec.first["images"]["wrist"], state0=rec.first["state"],
        thumbs=np.stack(frames[::3]), thumb_every=3,
        states=np.stack(rec.states), actions=np.stack(rec.actions),
        chunk_steps=np.array([c[0] for c in rec.chunks]), chunks=np.stack([c[1] for c in rec.chunks]),
        chunk_total_ms=np.array([c[2] for c in rec.chunks]), chunk_rt_ms=np.array([c[3] for c in rec.chunks]),
        success=r["success"], opened=r["opened"], t_success=r["t_success"] or -1, seed=SEED, hz=30)


if __name__ == "__main__":
    main()
