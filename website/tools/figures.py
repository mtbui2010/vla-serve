"""Render the documentation figures from data recorded on real runs.

    python website/tools/figures.py <data-dir>

<data-dir> holds the .npz files written by the recording scripts (see website/tools/README.md):
episode.npz (one kaair_sim episode through vla-serve), flow.npz, seeds.npz, timeline.npz.
Writes PNGs into website/docs/assets/img/. Colours: the validated categorical slots 1-3 and the
blue ordinal ramp (steps 250-650) of the dataviz reference palette.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
JOINTS = [f"joint {i}" for i in range(1, 8)] + ["gripper"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


ep = np.load(DATA / "episode.npz")
HZ = int(ep["hz"])
J = int(np.argmax(np.ptp(ep["actions"], axis=0)))          # the joint that moves most (joint 7)

# 1. what the policy sees -------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, key, title in zip(axs, ["head0", "wrist0"], ["head camera", "wrist camera"]):
    ax.imshow(ep[key]); ax.set_axis_off()
    h, w = ep[key].shape[:2]
    ax.set_title(f"{title}  ·  {w}×{h} RGB", loc="left")
TASK = str(ep["task"]) if "task" in ep.files else "open drawer"
fig.suptitle(f'task: "{TASK}"     state: 7 joint angles + gripper width (8 numbers)',
             x=0.01, ha="left", y=0.02, fontsize=9, color=INK2)
save(fig, "observation.png")

# 2. filmstrip -----------------------------------------------------------------------------
every = int(ep["thumb_every"])
thumbs = ep["thumbs"]
t_succ = float(ep["t_success"])
times = [round(t_succ * f, 1) for f in (0, 0.25, 0.5, 0.75)] + [t_succ]
fig, axs = plt.subplots(1, len(times), figsize=(13, 2.2))
for ax, t in zip(axs, times):
    i = min(int(round(t * HZ / every)), len(thumbs) - 1)
    ax.imshow(thumbs[i][:, : thumbs.shape[2] // 2]); ax.set_axis_off()
    label = f"t = {t:.1f} s" + ("  ·  success" if t == t_succ else "")
    ax.set_title(label, fontsize=9, fontweight="normal", color=INK2, loc="left")
save(fig, "episode-filmstrip.png")

# 3. chunks over an episode ------------------------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(12, 4.6), sharex=True)
t = np.arange(len(ep["actions"])) / HZ
for d, ax in enumerate(axs.flat):
    for s in ep["chunk_steps"]:
        ax.axvline(s / HZ, color=AXIS, lw=0.8, zorder=0)
    ax.plot(t, ep["states"][:, d], color=ORANGE, lw=1.6, label="measured state")
    ax.plot(t, ep["actions"][:, d], color=BLUE, lw=1.6, label="commanded action")
    ax.set_title(JOINTS[d], fontsize=9, loc="left")
for ax in axs[1]:
    ax.set_xlabel("time [s]")
axs[0, 0].legend(loc="upper left")
fig.suptitle(f"One episode through vla-serve: {len(ep['chunk_steps'])} chunks (grey lines = a new chunk arrives), 50 actions each",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
save(fig, "episode-actions.png")

# 4. flow matching: noise -> actions -------------------------------------------------------
fl = np.load(DATA / "flow.npz")
steps, times_fm = fl["steps"], fl["times"]
fig, ax = plt.subplots(figsize=(8, 3.6))
k = np.arange(steps.shape[1]) / HZ
show = [0, 2, 4, 6, 8, 10]
for n, i in enumerate(show):
    c = RAMP[min(int(n * (len(RAMP) - 1) / (len(show) - 1)), len(RAMP) - 1)]
    ax.plot(k, steps[i, :, J], color=c, lw=2.2 if i == show[-1] else 1.3)
for i, txt in [(show[0], "t=1  noise"), (show[-1], "t=0  action chunk")]:
    ax.annotate(txt, (k[-1], steps[i, -1, J]), xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=INK2, va="center")
ax.text(0.0, -0.24, "lighter = earlier step (t = 1.0, 0.8, 0.6, 0.4, 0.2, 0.0)", transform=ax.transAxes,
        fontsize=8, color=MUTED)
ax.set_xlabel("time within the chunk [s]"); ax.set_ylabel(f"{JOINTS[J]} target [rad]")
ax.set_title("Flow matching: 10 Euler steps turn random noise (t=1) into an action chunk (t=0)", loc="left")
save(fig, "flow-matching.png")

# 5. seeds ---------------------------------------------------------------------------------
sd = np.load(DATA / "seeds.npz")["chunks"]
fig, ax = plt.subplots(figsize=(8, 3.4))
for s in sd:
    ax.plot(k, s[:, J], color=BLUE, lw=1.2, alpha=0.6)
spread = float(np.ptp(sd[:, :, J], axis=0).max())
ax.set_xlabel("time within the chunk [s]"); ax.set_ylabel(f"{JOINTS[J]} target [rad]")
ax.set_title(f"Same observation, 8 random seeds: up to {spread:.2f} rad apart", loc="left")
save(fig, "seeds.png")

# 6. executor timeline ----------------------------------------------------------------------
if (DATA / "timeline.npz").exists():
    tl = np.load(DATA / "timeline.npz")
    req, ret, send = tl["req"], tl["ret"], tl["send"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    for (tr, _), (tt, now, at_obs) in zip(req, ret):
        ax.barh(1, tt - tr, left=tr, height=0.5, color=BLUE)
        stale = int(now - at_obs)
        if stale:
            ax.annotate(f"{stale} stale", (tt, 1.3), fontsize=7, color=INK2, ha="center")
    ax.vlines(send, -0.25, 0.25, color=ORANGE, lw=0.8)
    ax.set_yticks([0, 1], ["action sent\n(30 Hz)", "inference\nrequest"])
    ax.set_ylim(-0.6, 1.7); ax.set_xlabel("wall-clock time [s]"); ax.grid(axis="y", visible=False)
    first = send[0] if len(send) else 0
    ax.set_title(f"ChunkExecutor on a live server: {len(send)} actions, 0 stalls; first action after {first:.2f} s",
                 loc="left")
    save(fig, "executor-timeline.png")

# 7. latency --------------------------------------------------------------------------------
if (DATA / "latency.json").exists():
    lat = json.loads((DATA / "latency.json").read_text())
    names = list(lat)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    y = np.arange(len(names))
    ax.barh(y, [lat[n] for n in names], height=0.5, color=BLUE)
    for yi, n in zip(y, names):
        ax.annotate(f"{lat[n]:.0f} ms", (lat[n], yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlabel("server time per chunk, median [ms]")
    ax.grid(axis="y", visible=False)
    ax.set_title("Server time per chunk on one RTX A6000 (eager PyTorch, batch 1, 20 runs)", loc="left")
    save(fig, "latency.png")

# 8. evaluation -----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


rows = [("office · vla-serve, CPU", 42), ("office · in-process LeRobot, CPU", 42), ("office · vla-serve, GPU", 36),
        ("plain · vla-serve, GPU", 47), ("plain · in-process LeRobot, CPU", 49)]
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (name, k_) in enumerate(rows):
    lo, hi = wilson(k_, 50)
    ax.plot([100 * lo, 100 * hi], [i, i], color=AXIS, lw=3, solid_capstyle="round")
    ax.plot(100 * k_ / 50, i, "o", ms=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{k_}/50", (100 * k_ / 50, i), xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(55, 100); ax.set_xlabel("drawers opened [%]  (bar = 95 % Wilson interval)")
ax.grid(axis="y", visible=False)
ax.set_title("kaair_sim open-drawer, SmolVLA: 50 identical scenes per row (4–5 Oct 2026)", loc="left")
save(fig, "evaluation.png")

# 9. normalisation --------------------------------------------------------------------------
st = np.load(DATA / "stats.npz")
s0 = ep["state0"]
z = (s0 - st["mean"]) / (st["std"] + 1e-8)
fig, axs = plt.subplots(1, 2, figsize=(11, 2.8))
x = np.arange(8)
axs[0].bar(x, s0, color=ORANGE, width=0.6)
axs[0].set_title("what the robot reports (rad, m)", loc="left")
axs[1].bar(x, z, color=BLUE, width=0.6)
axs[1].set_title("what the network sees: (x − mean) / std", loc="left")
for ax in axs:
    ax.set_xticks(x, ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "grip"])
    ax.axhline(0, color=AXIS, lw=0.8); ax.grid(axis="x", visible=False)
save(fig, "normalisation.png")
