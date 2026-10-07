"""Data for the docs figures, from the recorded episode's first observation.

1. flow.npz     — x_t at each of the 10 Euler steps (in-process LeRobot, GPU), unnormalised
2. seeds.npz    — chunks for seeds 0..7 from the running vlaserve server
3. timeline.npz — a real ChunkExecutor run against the server (30 Hz, stub robot)
"""
import sys, time, threading
import numpy as np, torch

W = sys.argv[1] if len(sys.argv) > 1 else "."   # data dir: episode.npz in; flow/seeds/timeline.npz out
RUN = "smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model"
CK = f"/mnt/nas/lerobot_shared/kaair_sim/checkpoints/{RUN}"     # host path (in-process flow probe)
SERVED = f"/ckpt/{RUN}"                                          # the same checkpoint as the server sees it
PARTS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"seeds", "timeline", "flow"}
ep = np.load(f"{W}/episode.npz")
TASK = str(ep["task"]) if "task" in ep.files else "open drawer"
obs = {"images": {"head": ep["head0"], "wrist": ep["wrist0"]}, "state": ep["state0"], "task": TASK}

# --- 2 + 3 first (server), so the in-process model does not compete for the GPU -------------
from vlaserve.client import PolicyClient, ChunkExecutor
client = PolicyClient(model=SERVED, timeout=120)
client.reset()
if "seeds" in PARTS:
    seeds = np.stack([client.act(**obs, seed=s).actions for s in range(8)])
    np.savez(f"{W}/seeds.npz", chunks=seeds)

if "timeline" in PARTS:
    events = {"req": [], "ret": [], "send": []}
    t0 = time.monotonic()
    sent = []


    class Logged:
        def act(self, **kw):
            k = len(sent)
            events["req"].append((time.monotonic() - t0, k))
            c = client.act(**kw)
            events["ret"].append((time.monotonic() - t0, len(sent), k))
            return c


    def send(a):
        sent.append(a)
        events["send"].append(time.monotonic() - t0)


    ex = ChunkExecutor(Logged(), get_observation=lambda: {"images": obs["images"], "state": obs["state"]},
                       send_action=send, fps=30, execute_steps=30)
    st = ex.run(task=TASK, seconds=4.0)
    np.savez(f"{W}/timeline.npz", req=np.array(events["req"]), ret=np.array(events["ret"]),
             send=np.array(events["send"]), fps=30, execute_steps=30, refill_at=ex.refill_at, **{
                 k: v for k, v in st.summary().items() if isinstance(v, (int, float))})
    print("executor", st.summary(), "refill_at", ex.refill_at)

# --- 1. flow matching steps, in process ---------------------------------------------------
if "flow" in PARTS:
    import lerobot.policies.smolvla.modeling_smolvla as ms
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(CK); cfg.device = "cuda"
    pol = get_policy_class(cfg.type).from_pretrained(CK, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=CK,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}},
                                         postprocessor_overrides={"device_processor": {"device": "cpu"}})
    traj = []
    orig = ms.euler_integrate


    def recording(denoise_fn, noise, num_steps, **kw):
        def fn(x_t, t):
            traj.append((float(t.flatten()[0]), x_t.detach().clone()))
            return denoise_fn(x_t, t)
        out = orig(fn, noise, num_steps, **kw)
        traj.append((0.0, out.detach().clone()))
        return out


    ms.euler_integrate = recording
    chw = lambda a: torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
    b = pre({"observation.state": torch.from_numpy(obs["state"]), "observation.images.head": chw(obs["images"]["head"]),
             "observation.images.wrist": chw(obs["images"]["wrist"]), "task": TASK})
    with torch.inference_mode():
        torch.manual_seed(0)
        final = pol.predict_action_chunk(b)
        D = final.shape[-1]
        steps = []
        for t, x in traj:
            x = x[:, :, :D]
            steps.append(torch.stack([post(x[:, i, :]) for i in range(x.shape[1])], 1).squeeze(0).float().numpy())
    np.savez(f"{W}/flow.npz", times=np.array([t for t, _ in traj]), steps=np.stack(steps))
    print("flow steps", len(traj), "times", [round(t, 2) for t, _ in traj])
