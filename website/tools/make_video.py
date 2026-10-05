"""Render the tutorial video: step-by-step usage with real outputs, then the policy in the sim.

    python website/tools/make_video.py <data-dir> [out.mp4]

<data-dir>: episode.npz + episode.mp4 from record_episode.py (real-texture kaair scene),
out/*.txt|json captured from the real commands, results.json with the evaluation counts.
Terminal panels show captured output verbatim; long checkpoint paths are abbreviated to $CKPT.
1280x720, 30 fps, H.264 via ffmpeg.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "video" / "vla-serve-tutorial.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H, FPS = 1280, 720, 30

FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
def font(name, size): return ImageFont.truetype(str(FD / name), size)
SANS, BOLD, MONO, MONOB = "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"

BG, INK, INK2, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78), (137, 135, 129)
TERM, TERM_INK, TERM_DIM = (26, 26, 25), (236, 236, 232), (160, 158, 150)
BLUE, ORANGE, AQUA, GRID = (42, 120, 214), (235, 104, 52), (27, 175, 122), (225, 224, 217)
PROMPT = (110, 180, 255)

CKPT_HOST = "/mnt/nas/lerobot_shared/kaair_sim/checkpoints/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model"
CKPT = "/ckpt/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model"


def short(s: str) -> str:
    return s.replace(CKPT_HOST, "$CKPT").replace(CKPT, "$CKPT")


def txt(name: str) -> list[str]:
    return [short(l.rstrip()) for l in (DATA / "out" / name).read_text().splitlines()]


# --- drawing helpers ---------------------------------------------------------------------
def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur); cur = w_
    return lines + ([cur] if cur else [])


def header(d, step, title):
    if step:
        d.text((48, 34), step.upper(), font=font(BOLD, 18), fill=BLUE)
    d.text((48, 58), title, font=font(BOLD, 34), fill=INK)


def caption(d, text):
    f = font(SANS, 22)
    for i, line in enumerate(wrap(d, text, f, W - 96)):
        d.text((48, 640 + 28 * i - 14 * (len(wrap(d, text, f, W - 96)) - 1)), line, font=f, fill=INK2)


def terminal(d, cmds, t, box=(48, 120, W - 48, 610), size=17):
    """cmds: list of (command, output_lines). Commands type out, then their output appears."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, 12, fill=TERM)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((x0 + 16 + 20 * i, y0 + 14, x0 + 28 + 20 * i, y0 + 26), fill=c)
    f, fb = font(MONO, size), font(MONOB, size)
    lh = int(size * 1.45)
    y, clock = y0 + 44, 0.0
    maxw = int((x1 - x0 - 40) / d.textlength("m", font=f))
    for cmd, out in cmds:
        n = len(cmd)
        typed = int(np.clip((t - clock) / 0.9 * n, 0, n))
        if t < clock:
            break
        for k, chunk in enumerate([cmd[i:i + maxw - 2] for i in range(0, max(n, 1), maxw - 2)]):
            shown = chunk[: max(0, typed - k * (maxw - 2))]
            if k == 0:
                d.text((x0 + 20, y), "$", font=fb, fill=PROMPT)
            d.text((x0 + 20 + d.textlength("$ ", font=f), y), shown, font=fb, fill=TERM_INK)
            y += lh
        clock += 1.1
        for j, line in enumerate(out):
            if t < clock + 0.06 * j:
                return
            for piece in [line[i:i + maxw] for i in range(0, max(len(line), 1), maxw)] or [""]:
                if y > y1 - lh:
                    return
                color = TERM_INK if not line.lstrip().startswith(("#", "INFO", "WARNING")) else TERM_DIM
                if "OK:" in line or '"status":"ok"' in line:
                    color = (120, 220, 150)
                if "problem" in line or "Error" in line:
                    color = (255, 150, 120)
                d.text((x0 + 20, y), piece, font=f, fill=color)
                y += lh
        clock += 0.06 * len(out) + 0.9
        y += 6


def arrow(d, a, b, color, width=4):
    d.line([a, b], fill=color, width=width)
    v = np.array(b, float) - np.array(a, float); v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    p = np.array(b, float)
    d.polygon([tuple(p), tuple(p - 16 * v + 8 * n), tuple(p - 16 * v - 8 * n)], fill=color)


# --- scenes ------------------------------------------------------------------------------
def s_title(t):
    im, d = canvas()
    d.text((48, 230), "vla-serve", font=font(BOLD, 84), fill=INK)
    d.text((52, 340), "Serve a trained robot policy with one command,", font=font(SANS, 34), fill=INK2)
    d.text((52, 384), "and drive a robot with it.", font=font(SANS, 34), fill=INK2)
    d.text((52, 470), "Every output in this video is real — recorded 5 Oct 2026 on one RTX A6000.",
           font=font(SANS, 20), fill=MUTED)
    return im


def s_overview(t):
    im, d = canvas()
    header(d, "", "The idea")
    boxes = [((70, 190, 520, 470), "Robot computer", ["pip install vla-serve  (~80 MB)", "", "reads cameras + joints",
                                                       "plays actions at 30 Hz", "(ChunkExecutor)"]),
             ((760, 190, 1210, 470), "GPU machine", ["docker run vla-serve", "", "loads a LeRobot checkpoint",
                                                     "(pi0.5, SmolVLA, …)", "returns chunks of actions"])]
    for (x0, y0, x1, y1), title, lines in boxes:
        d.rounded_rectangle((x0, y0, x1, y1), 16, outline=GRID, width=3, fill=(255, 255, 255))
        d.text((x0 + 28, y0 + 24), title, font=font(BOLD, 28), fill=INK)
        for i, l in enumerate(lines):
            d.text((x0 + 28, y0 + 80 + 34 * i), l, font=font(MONO if "install" in l or "docker" in l else SANS, 20),
                   fill=INK2)
    if t > 0.8:
        arrow(d, (530, 280), (750, 280), BLUE)
        d.text((548, 236), "frames, joints, task", font=font(SANS, 18), fill=BLUE)
    if t > 1.6:
        arrow(d, (750, 380), (530, 380), ORANGE)
        d.text((552, 392), "50 future actions", font=font(SANS, 18), fill=ORANGE)
    caption(d, "The policy runs on a GPU machine. The robot sends what it sees, the server sends back the "
               "next second or two of motion, and the robot plays it smoothly while the next part is computed.")
    return im


def json_lines(name, keep):
    js = json.loads((DATA / "out" / name).read_text())
    sp = js["spec"]
    pick = {"policy_type": sp["policy_type"], "cameras": list(sp["cameras"]), "state_dim": sp["state_dim"],
            "action_dim": sp["action_dim"], "chunk_size": sp["chunk_size"], "language": sp["language"],
            "source": sp["source"], "license": js["license"]}
    return ["{"] + [f'  "{k}": {json.dumps(v)},' for k, v in pick.items() if k in keep] + ["  …", "}"]


def step_scene(step, title, cmds, cap):
    def f(t):
        im, d = canvas()
        header(d, step, title)
        terminal(d, cmds, t)
        caption(d, cap)
        return im
    return f


def s_sim_factory():
    ep = np.load(DATA / "episode.npz")
    cap = cv2.VideoCapture(str(DATA / "episode.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    opened = ep["opened"] * 100
    steps, ms = list(ep["chunk_steps"]), list(ep["chunk_total_ms"])
    t_succ, task = float(ep["t_success"]), str(ep["task"])
    n = len(frames)

    def f(t):
        i = min(int(t * 30), n - 1)
        im, d = canvas()
        header(d, "Step 5", "Run it: the robot opens the drawer")
        fr = frames[i]
        head = Image.fromarray(fr[:, : fr.shape[1] // 2]).resize((768, 435))
        wrist = Image.fromarray(fr[:, fr.shape[1] // 2:]).resize((420, 238))
        im.paste(head, (48, 118)); im.paste(wrist, (830, 118))
        d.text((56, 124), "head camera", font=font(BOLD, 16), fill=(255, 255, 255))
        d.text((838, 124), "wrist camera", font=font(BOLD, 16), fill=(255, 255, 255))
        # info panel
        x, y = 830, 372
        k = max(j for j, s in enumerate(steps) if s <= i)
        fresh = i - steps[k] < 8
        d.text((x, y), f'task  "{task}"', font=font(MONO, 18), fill=INK)
        d.text((x, y + 30), f"time  {i / 30:4.1f} s     step {i:3d}", font=font(MONO, 18), fill=INK2)
        d.rounded_rectangle((x, y + 62, x + 420, y + 96), 8, fill=BLUE if fresh else (232, 240, 251))
        d.text((x + 12, y + 68), f"chunk {k + 1}/{len(steps)} · server {ms[k]:.0f} ms",
               font=font(MONOB if fresh else MONO, 18), fill=(255, 255, 255) if fresh else BLUE)
        # drawer bar
        o = max(0.0, float(opened[min(i, len(opened) - 1)]))   # contact jitter can read -0.1 cm
        d.text((x, y + 110), f"drawer opened  {o:4.1f} cm", font=font(MONO, 18), fill=INK)
        d.rounded_rectangle((x, y + 140, x + 420, y + 160), 6, fill=(240, 239, 236))
        d.rounded_rectangle((x, y + 140, x + max(12, int(420 * min(o, 28) / 28)), y + 160), 6, fill=ORANGE)
        gx = x + int(420 * 20 / 28)
        d.line([(gx, y + 134), (gx, y + 166)], fill=INK, width=2)
        d.text((gx - 30, y + 168), "20 cm = success", font=font(SANS, 14), fill=MUTED)
        if i / 30 >= t_succ:
            d.rounded_rectangle((48, 118, 230, 160), 8, fill=AQUA)
            d.text((60, 125), "✓ success", font=font(BOLD, 22), fill=(255, 255, 255))
        caption(d, "Same checkpoint, now driving the simulated kaair robot in a room textured from photos of the "
                   "real office. A new chunk of 50 actions every 1.7 s of robot time (blue flash); the "
                   "simulator pauses while the server thinks.")
        return im
    return f, n / 30 + 1.5


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - r) / dd, 100 * (c + r) / dd


def s_results(t):
    res = json.loads((DATA / "results.json").read_text())
    im, d = canvas()
    header(d, "Result", "Same policy, served vs in-process")
    rows = res["rows"]
    y0 = 150
    for j, r in enumerate(rows):
        y = y0 + 76 * j
        d.text((48, y), r["label"], font=font(SANS, 22), fill=INK)
        d.text((48, y + 30), r["sub"], font=font(SANS, 16), fill=MUTED)
        x0, x1 = 560, 1180
        d.line([(x0, y + 22), (x1, y + 22)], fill=GRID, width=2)
        for v in (60, 70, 80, 90, 100):
            gx = x0 + (x1 - x0) * (v - 60) / 40
            d.text((gx - 12, y0 + 76 * len(rows) - 10), f"{v}%", font=font(SANS, 14), fill=MUTED) if j == 0 else None
        p = 100 * r["k"] / r["n"]
        lo, hi = wilson(r["k"], r["n"])
        X = lambda v: x0 + (x1 - x0) * (v - 60) / 40
        a = min(1.0, t / 1.0)
        d.line([(X(max(lo, 60)), y + 22), (X(hi), y + 22)], fill=(195, 194, 183), width=8)
        if a >= 1:
            c = BLUE if r["served"] else MUTED
            d.ellipse((X(p) - 10, y + 12, X(p) + 10, y + 32), fill=c, outline=BG, width=3)
            d.text((X(p) - 22, y - 16), f'{r["k"]}/{r["n"]}', font=font(BOLD, 18), fill=INK)
    d.text((560, y0 + 76 * len(rows) + 14), "dot = success rate · bar = 95 % interval (Wilson)",
           font=font(SANS, 14), fill=MUTED)
    caption(d, res["caption"])
    return im


def s_end(t):
    im, d = canvas()
    d.text((48, 210), "Try it", font=font(BOLD, 56), fill=INK)
    for i, (k, v) in enumerate([("Docs and VLA course", "mtbui2010.github.io/vla-serve"),
                                ("Code", "github.com/mtbui2010/vla-serve"),
                                ("One command", "make help")]):
        d.text((52, 320 + 62 * i), k, font=font(SANS, 24), fill=MUTED)
        d.text((380, 318 + 62 * i), v, font=font(MONOB, 26), fill=BLUE)
    return im


# --- timeline ----------------------------------------------------------------------------
def main():
    client_out = txt("client.txt")
    sim, sim_len = s_sim_factory()
    scenes = [
        (s_title, 4.0),
        (s_overview, 7.0),
        (step_scene("Step 1", "Build the server image (once)",
                    [("make docker", ["…", "#13 DONE 6.8s"]),
                     ("docker images vla-serve", txt("images.txt"))],
                    "One image holds PyTorch, CUDA and LeRobot in versions that work together. "
                    "No model weights inside: checkpoints are mounted when you run it."), 6.5),
        (step_scene("Step 2", "Look at a checkpoint before loading it",
                    [("vla-serve inspect $CKPT",
                      json_lines("inspect.json", {"policy_type", "cameras", "state_dim", "action_dim",
                                                  "chunk_size", "source", "license"}))],
                    "Instant, no GPU: it reads the checkpoint's metadata. This SmolVLA wants two cameras "
                    "(head, wrist), 8 joint values and a sentence, and returns 50 actions."), 7.5),
        (step_scene("Step 3", "Check the robot matches the policy",
                    [("vla-serve check $CKPT --profile kaair", txt("check_ok.txt")),
                     ("vla-serve check $CKPT --profile libero", txt("check_bad.txt"))],
                    "A robot profile names your cameras and joints. A mismatch is caught here, "
                    "before anything moves — not after the arm does something strange."), 8.5),
        (step_scene("Step 4", "Start the server and ask for actions",
                    [("make docker-run GPU=3 CKPT_ROOT=/mnt/nas/…/checkpoints MODEL=$CKPT",
                      txt("serverlog.txt")[-2:]),
                     ("make health", txt("health.txt")),
                     ("python ask.py      # client.act(images=…, state=…, task=…)", client_out)],
                    "The client sends camera frames, joint positions and the instruction. A missing camera "
                    "is refused with the reason, before it costs GPU time."), 11.0),
        (sim, sim_len),
        (s_results, 6.0),
        (s_end, 4.0),
    ]
    total = sum(d for _, d in scenes)
    print(f"{len(scenes)} scenes, {total:.1f} s")
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
    fade = 0.3
    for fn, dur in scenes:
        nf = int(dur * FPS)
        for k in range(nf):
            t = k / FPS
            im = fn(t)
            a = min(1.0, t / fade, (dur - t) / fade)
            if a < 1:
                im = Image.blend(Image.new("RGB", (W, H), BG), im, max(a, 0))
            ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
