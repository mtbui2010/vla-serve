"""Record one kaair_sim open-drawer episode driven through vlaserve (docs figures + video).

Runs inside kaair_sim_humble, against a running vlaserve server. Scene construction is the
same as kaair_sim's eval_policy.main(); only the policy is an HTTP client.

    python record_episode.py --model /ckpt/<run>/checkpoints/020000/pretrained_model \
        --task task_open_drawer.yaml --rand-config randomization_real.yaml --camera-model \
        --seed 1000000000 --out /tmp/docsfig

Writes <out>/episode.npz (first observation at full resolution, thumbnails every 3rd step,
states, executed actions, every chunk with its request step and timing, drawer opening per step)
and <out>/episode.mp4 (head | wrist at full resolution, 30 fps, as the policy saw them).
"""
import argparse, os, sys, time
import imageio, mujoco, numpy as np, torch, yaml, cv2

SIM = "/sim"
sys.path.insert(0, f"{SIM}/scripts")
import cabinet, check_sim as cs, drawer_ready as dr, eval_policy as ev, randomize  # noqa: E402
from vlaserve.client import PolicyClient  # noqa: E402


class Recorder:
    def __init__(self, client, video, drawer):
        self.client, self.video, self.drawer = client, video, drawer   # drawer() -> opening [m]
        self.queue, self.step = [], 0
        self.states, self.actions, self.chunks, self.opened, self.thumbs = [], [], [], [], []
        self.first = None

    def reset(self):
        self.queue.clear(); self.client.reset()

    def pre(self, obs):
        hwc = lambda t: (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        o = {"images": {"head": hwc(obs["observation.images.head"]), "wrist": hwc(obs["observation.images.wrist"])},
             "state": obs["observation.state"].numpy(), "task": obs["task"]}
        if self.first is None:
            self.first = o
        self.states.append(o["state"].copy())
        self.opened.append(self.drawer())
        frame = np.concatenate([o["images"]["head"], o["images"]["wrist"]], 1)
        self.video.append_data(frame)
        if self.step % 3 == 0:
            self.thumbs.append(cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA))
        return o

    def select_action(self, o):
        if not self.queue:
            c = self.client.act(**o, seed=self.step)          # seeded: reproducible figures
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name/path as the server sees it")
    ap.add_argument("--url", default="http://127.0.0.1:11450")
    ap.add_argument("--task", default="task_open_drawer.yaml")
    ap.add_argument("--rand-config", default="randomization_real.yaml")
    ap.add_argument("--camera-model", action="store_true")
    ap.add_argument("--seed", type=int, default=10 ** 9)
    ap.add_argument("--out", default="/tmp/docsfig")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = yaml.safe_load(open(f"{SIM}/config/mjcf.yaml"))
    task = yaml.safe_load(open(f"{SIM}/config/{a.task}"))
    rcfg = yaml.safe_load(open(f"{SIM}/config/{a.rand_config}"))
    cams = None
    if a.camera_model:
        import camera_model
        cams = camera_model.load_camera_models()
    ref = cs.UrdfFK(f"{SIM}/urdf/raw/kaair_slamtec.urdf")
    tcp_T = ref.fk(dr.ready_q(task, task["mode"], cfg["home"]))["tool_tcp_link"]

    rng = np.random.default_rng(a.seed)                 # identical to eval_policy.main()
    rand = randomize.sample(rng, rcfg)
    if cams is not None:
        for c in cams.values():
            c.episode_jitter(rng)
    xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
    path = f"{SIM}/mjcf/_docsfig_scene_{os.getpid()}.xml"
    open(path, "w").write(xml)
    try:
        m = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    d = mujoco.MjData(m)
    target = rand.get("target", cabinet.STYLES[task["cabinet"]]["target"])
    adr = m.jnt_qposadr[m.joint(f"cab_drawer{target}_slide").id]

    video = imageio.get_writer(f"{a.out}/episode.mp4", fps=30, quality=8, macro_block_size=1)
    rec = Recorder(PolicyClient(a.url, model=a.model, timeout=120), video,
                   drawer=lambda: float(d.qpos[adr] - rand["drawer0"]))
    t0 = time.time()
    try:
        r = ev.run(m, d, task, cfg, rand, rec, rec.pre, rec.post, 15.0, 0.20, None, cams, rng)
    finally:
        video.close()
    print("result", r, "wall", round(time.time() - t0), "s", "steps", rec.step, "chunks", len(rec.chunks))
    np.savez_compressed(
        f"{a.out}/episode.npz",
        head0=rec.first["images"]["head"], wrist0=rec.first["images"]["wrist"], state0=rec.first["state"],
        thumbs=np.stack(rec.thumbs), thumb_every=3, task=r["prompt"],
        states=np.stack(rec.states), actions=np.stack(rec.actions), opened=np.array(rec.opened),
        chunk_steps=np.array([c[0] for c in rec.chunks]), chunks=np.stack([c[1] for c in rec.chunks]),
        chunk_total_ms=np.array([c[2] for c in rec.chunks]), chunk_rt_ms=np.array([c[3] for c in rec.chunks]),
        success=r["success"], opened_final=r["opened"], t_success=r["t_success"] or -1, seed=a.seed, hz=30)


if __name__ == "__main__":
    main()
