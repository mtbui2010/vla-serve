"""Render the narrated tutorial video: step-by-step usage with real outputs, then the policy in
the simulator, with an English voice-over (tts.py) and timed subtitles.

    python website/tools/make_video.py <data-dir> [out.mp4]

<data-dir> holds
  episode.npz, episode.mp4   one office-scene episode recorded through vla-serve (record_episode.py)
  out/*.txt|json             outputs captured from the real commands (see website/tools/README.md)
  results.json               evaluation counts for the results scene
  voice/<scene>.wav, voice/timing.json   narration from tts.py (narration.json)
  executor-timeline.png      the docs figure of a live ChunkExecutor run

Terminal panels show captured output verbatim; long checkpoint paths are abbreviated to $CKPT.
Each scene lasts as long as its narration plus a short pause. 1280x720, 30 fps, H.264 + AAC.
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "video" / "vla-serve-tutorial.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H, FPS, SR = 1280, 720, 30, 24000
LEAD, TAIL = 0.4, 0.7                       # silence before / after each scene's narration [s]

FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
def font(name, size): return ImageFont.truetype(str(FD / name), size)
SANS, BOLD, MONO, MONOB = "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"

BG, INK, INK2, MUTED = (252, 252, 251), (11, 11, 11), (82, 81, 78), (137, 135, 129)
TERM, TERM_INK, TERM_DIM = (26, 26, 25), (236, 236, 232), (160, 158, 150)
BLUE, ORANGE, AQUA, GRID = (42, 120, 214), (235, 104, 52), (27, 175, 122), (225, 224, 217)
PROMPT = (110, 180, 255)

CKPT_HOST = "/mnt/nas/lerobot_shared/kaair_sim/checkpoints/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model"
CKPT = "/ckpt/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model"
TIMING = json.loads((DATA / "voice" / "timing.json").read_text())


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
        d.text((48, 30), step.upper(), font=font(BOLD, 18), fill=BLUE)
    d.text((48, 54), title, font=font(BOLD, 34), fill=INK)


# Narration spells numbers out so the voice reads them well; subtitles show them as numbers.
DISPLAY = [("two point four megabytes", "2.4 MB"), ("one hundred kilobytes", "100 kB"),
           ("eighty megabytes", "80 MB"), ("one point seven seconds", "1.7 s"),
           ("twenty five centimeters", "25 cm"), ("thirty three milliseconds", "33 ms"),
           ("every thirty three", "every 33 ms"), ("a few hundred milliseconds", "a few hundred ms"),
           ("Forty two out of fifty", "42 out of 50"), ("forty four", "44"),
           ("fifty simulated episodes", "50 simulated episodes"), ("fifty actions", "50 actions"),
           ("fifty seconds", "50 seconds"), ("eight joint values", "8 joint values"),
           ("four minutes", "4 minutes"), ("under ten seconds", "under 10 seconds"),
           ("three clear reasons", "3 clear reasons"), ("Step one.", "Step 1."), ("Step two.", "Step 2."),
           ("Step three.", "Step 3."), ("Step four.", "Step 4."), ("Step five.", "Step 5."),
           ("Step six.", "Step 6."), ("fine tuning", "fine-tuning"), ("in process", "in-process"),
           ("read only", "read-only")]


def display(text: str) -> str:
    for spoken, shown in DISPLAY:
        text = text.replace(spoken, shown)
    return text


def subtitles(im, sid, t):
    """The sentence being spoken at scene time t, in a band at the bottom."""
    tt = t - LEAD
    cur = next((s for s, a, b in TIMING[sid]["sentences"] if a - 0.05 <= tt <= b + 0.15), None)
    if not cur:
        return
    cur = display(cur)
    d = ImageDraw.Draw(im, "RGBA")
    f = font(SANS, 22)
    lines = wrap(d, cur, f, W - 160)[:3]
    h = 30 * len(lines) + 18
    y0 = H - h - 12
    d.rounded_rectangle((60, y0, W - 60, H - 12), 10, fill=(20, 20, 20, 200))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=f)
        d.text(((W - w) / 2, y0 + 9 + 30 * i), line, font=f, fill=(255, 255, 255))


def terminal(d, cmds, t, box=(48, 112, W - 48, 590), size=17, pace=1.0):
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
        if t < clock:
            break
        n = len(cmd)
        typed = int(np.clip((t - clock) / (0.9 * pace) * n, 0, n))
        for k, chunk in enumerate([cmd[i:i + maxw - 2] for i in range(0, max(n, 1), maxw - 2)]):
            shown = chunk[: max(0, typed - k * (maxw - 2))]
            if k == 0:
                d.text((x0 + 20, y), "$", font=fb, fill=PROMPT)
            d.text((x0 + 20 + d.textlength("$ ", font=f), y), shown, font=fb, fill=TERM_INK)
            y += lh
        clock += 1.1 * pace
        for j, line in enumerate(out):
            if t < clock + 0.06 * j:
                return
            for piece in [line[i:i + maxw] for i in range(0, max(len(line), 1), maxw)] or [""]:
                if y > y1 - lh:
                    return
                color = TERM_INK if not line.lstrip().startswith(("#", "INFO", "WARNING")) else TERM_DIM
                if "OK:" in line or '"status":"ok"' in line or "defaults" in line:
                    color = (120, 220, 150)
                if "problem" in line or "Error" in line:
                    color = (255, 150, 120)
                d.text((x0 + 20, y), piece, font=f, fill=color)
                y += lh
        clock += (0.06 * len(out) + 1.2) * pace
        y += 6


def code_panel(d, lines, box, size=16):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, 12, fill=(245, 244, 240), outline=GRID, width=2)
    f = font(MONO, size)
    for i, l in enumerate(lines):
        col = MUTED if l.lstrip().startswith("#") else INK
        d.text((x0 + 18, y0 + 14 + int(size * 1.45) * i), l, font=f, fill=col)


def arrow(d, a, b, color, width=4):
    d.line([a, b], fill=color, width=width)
    v = np.array(b, float) - np.array(a, float); v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]]); p = np.array(b, float)
    d.polygon([tuple(p), tuple(p - 16 * v + 8 * n), tuple(p - 16 * v - 8 * n)], fill=color)


# --- scenes ------------------------------------------------------------------------------
def s_title(t):
    im, d = canvas()
    d.text((48, 210), "vla-serve", font=font(BOLD, 84), fill=INK)
    d.text((52, 320), "Serve a trained robot policy with one command,", font=font(SANS, 34), fill=INK2)
    d.text((52, 364), "and drive a robot with it.", font=font(SANS, 34), fill=INK2)
    d.text((52, 450), "A 4-minute walkthrough. Every output shown is real — recorded October 2026 on one RTX A6000.",
           font=font(SANS, 19), fill=MUTED)
    return im


def s_problem(t):
    im, d = canvas()
    header(d, "", "Why deploying a VLA is hard")
    rows = [("Every policy family loads differently", "loads any LeRobot checkpoint from its own files"),
            ("Library upgrades break old checkpoints", "compatibility layer, files never modified"),
            ("config.json can be stale after fine-tuning", "trusts the dataset statistics first"),
            ("Cameras and joints must match training", "vla-serve check, before anything moves"),
            ("Model: ~300 ms per call. Robot: 33 ms per step", "action chunks + ChunkExecutor")]
    d.text((70, 128), "Problem", font=font(BOLD, 20), fill=MUTED)
    d.text((700, 128), "What vla-serve does", font=font(BOLD, 20), fill=MUTED)
    span = TIMING["s02_problem"]["duration"]
    for i, (p, s) in enumerate(rows):
        y = 172 + 78 * i
        if t < LEAD + 1.5 + i * (span - 4) / len(rows):
            break
        d.rounded_rectangle((60, y, 620, y + 60), 10, fill=(255, 255, 255), outline=GRID, width=2)
        d.text((80, y + 17), p, font=font(SANS, 20), fill=INK)
        arrow(d, (630, y + 30), (686, y + 30), BLUE, 3)
        d.rounded_rectangle((696, y, 1222, y + 60), 10, fill=(232, 240, 251))
        d.text((716, y + 17), s, font=font(SANS, 20), fill=BLUE)
    return im


def s_overview(t):
    im, d = canvas()
    header(d, "", "Two parts")
    boxes = [((70, 170, 520, 450), "Robot computer", ["pip install vla-serve  (~80 MB)", "", "reads cameras + joints",
                                                       "plays actions at 30 Hz", "(ChunkExecutor)"]),
             ((760, 170, 1210, 450), "GPU machine", ["docker run vla-serve", "", "loads a LeRobot checkpoint",
                                                     "(pi0.5, SmolVLA, …)", "returns chunks of actions"])]
    for (x0, y0, x1, y1), title, lines in boxes:
        d.rounded_rectangle((x0, y0, x1, y1), 16, outline=GRID, width=3, fill=(255, 255, 255))
        d.text((x0 + 28, y0 + 24), title, font=font(BOLD, 28), fill=INK)
        for i, l in enumerate(lines):
            d.text((x0 + 28, y0 + 80 + 34 * i), l, font=font(MONO if "install" in l or "docker" in l else SANS, 20),
                   fill=INK2)
    if t > 6:
        arrow(d, (530, 260), (750, 260), BLUE)
        d.text((548, 216), "frames, joints, task", font=font(SANS, 18), fill=BLUE)
    if t > 11:
        arrow(d, (750, 360), (530, 360), ORANGE)
        d.text((552, 372), "next 50 actions", font=font(SANS, 18), fill=ORANGE)
    return im


def json_lines(name):
    js = json.loads((DATA / "out" / name).read_text())
    sp = js["spec"]
    pick = [("policy_type", sp["policy_type"]), ("cameras", list(sp["cameras"])), ("state_dim", sp["state_dim"]),
            ("action_dim", sp["action_dim"]), ("chunk_size", sp["chunk_size"]), ("language", sp["language"]),
            ("source", sp["source"]), ("image_input", sp.get("image_input")), ("license", js["license"])]
    return ["{"] + [f'  "{k}": {json.dumps(v)},' for k, v in pick] + ["  …", "}"]


def step_scene(step, title, cmds, pace=1.0):
    def f(t):
        im, d = canvas()
        header(d, step, title)
        terminal(d, cmds, t - LEAD, pace=pace)
        return im
    return f


ASK = ["from vlaserve.client import PolicyClient",
       "",
       "client = PolicyClient(\"http://gpu-box:11450\", model=CKPT)",
       "# defaults: resize=\"auto\", encoding=\"jpeg\"",
       "chunk = client.act(images={\"head\": head, \"wrist\": wrist},",
       "                   state=q, task=\"open the bottom drawer\")",
       "chunk.actions        # (50, 8): next 50 targets, robot units"]


def s_ask(t):
    im, d = canvas()
    header(d, "Step 5", "Ask for actions from Python")
    code_panel(d, ASK, (48, 112, W - 48, 300))
    terminal(d, [("python ask.py", txt("client.txt"))], t - LEAD - 2.0, box=(48, 316, W - 48, 590), size=16,
             pace=1.6)
    return im


def s_chunks(t):
    im, d = canvas()
    header(d, "", "Action chunks: the robot never waits")
    fig = Image.open(DATA / "executor-timeline.png").convert("RGB")
    fig.thumbnail((W - 96, 420))
    im.paste(fig, ((W - fig.width) // 2, 130))
    return im


def s_sim_factory(narr):
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
    t_succ, task, n = float(ep["t_success"]), str(ep["task"]), len(frames)
    dur = max(narr + LEAD + TAIL, n / 30 + 2.0)
    speed = min(1.0, (n / 30) / (dur - LEAD - 1.5))            # slow down to fit the narration

    def f(t):
        i = int(np.clip((t - LEAD) * 30 * speed, 0, n - 1))
        im, d = canvas()
        header(d, "Step 6", "Run it: the robot opens the drawer")
        fr = frames[i]
        head = Image.fromarray(fr[:, : fr.shape[1] // 2]).resize((768, 435))
        wrist = Image.fromarray(fr[:, fr.shape[1] // 2:]).resize((420, 238))
        im.paste(head, (48, 110)); im.paste(wrist, (830, 110))
        d.text((56, 116), "head camera", font=font(BOLD, 16), fill=(255, 255, 255))
        d.text((838, 116), "wrist camera", font=font(BOLD, 16), fill=(255, 255, 255))
        if speed < 0.99:
            d.text((740, 116), f"{speed:.1f}×", font=font(BOLD, 16), fill=(255, 255, 255))
        x, y = 830, 362
        k = max(j for j, s in enumerate(steps) if s <= i)
        fresh = i - steps[k] < 8
        d.text((x, y), f'task  "{task}"', font=font(MONO, 18), fill=INK)
        d.text((x, y + 30), f"time  {i / 30:4.1f} s     step {i:3d}", font=font(MONO, 18), fill=INK2)
        d.rounded_rectangle((x, y + 62, x + 420, y + 96), 8, fill=BLUE if fresh else (232, 240, 251))
        d.text((x + 12, y + 68), f"chunk {k + 1}/{len(steps)} · server {ms[k]:.0f} ms",
               font=font(MONOB if fresh else MONO, 18), fill=(255, 255, 255) if fresh else BLUE)
        o = max(0.0, float(opened[min(i, len(opened) - 1)]))   # contact jitter can read -0.1 cm
        d.text((x, y + 110), f"drawer opened  {o:4.1f} cm", font=font(MONO, 18), fill=INK)
        d.rounded_rectangle((x, y + 140, x + 420, y + 160), 6, fill=(240, 239, 236))
        d.rounded_rectangle((x, y + 140, x + max(12, int(420 * min(o, 28) / 28)), y + 160), 6, fill=ORANGE)
        gx = x + int(420 * 20 / 28)
        d.line([(gx, y + 134), (gx, y + 166)], fill=INK, width=2)
        d.text((gx - 30, y + 168), "20 cm = success", font=font(SANS, 14), fill=MUTED)
        if i / 30 >= t_succ:
            d.rounded_rectangle((48, 110, 230, 152), 8, fill=AQUA)
            d.text((60, 117), "✓ success", font=font(BOLD, 22), fill=(255, 255, 255))
        return im
    return f, dur


def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - r) / dd, 100 * (c + r) / dd


def s_results(t):
    res = json.loads((DATA / "results.json").read_text())
    im, d = canvas()
    header(d, "", "Does serving change the policy?")
    # 1. parity
    d.rounded_rectangle((48, 112, W - 48, 196), 12, fill=(232, 246, 240))
    d.text((70, 126), "1. Same input, same seed", font=font(BOLD, 20), fill=INK)
    d.text((70, 158), "served chunk vs LeRobot's own: max |difference| = 0, on pi0.5 (2 checkpoints) and SmolVLA",
           font=font(SANS, 18), fill=INK2)
    # 2. closed loop
    d.text((70, 214), "2. Closed loop: 50 identical simulated episodes per row", font=font(BOLD, 20), fill=INK)
    if t < LEAD + 9:
        return im
    rows, y0 = res["rows"], 258
    x0, x1 = 600, 1180
    X = lambda v: x0 + (x1 - x0) * (v - 60) / 40
    for j, r in enumerate(rows):
        y = y0 + 64 * j
        d.text((70, y), r["label"], font=font(SANS, 19), fill=INK)
        d.text((70, y + 24), r["sub"], font=font(SANS, 14), fill=MUTED)
        d.line([(x0, y + 18), (x1, y + 18)], fill=GRID, width=2)
        lo, hi = wilson(r["k"], r["n"])
        d.line([(X(max(lo, 60)), y + 18), (X(hi), y + 18)], fill=(195, 194, 183), width=8)
        p = 100 * r["k"] / r["n"]
        c = BLUE if r["served"] else MUTED
        d.ellipse((X(p) - 9, y + 9, X(p) + 9, y + 27), fill=c, outline=BG, width=3)
        d.text((X(p) - 20, y - 14), f'{r["k"]}/{r["n"]}', font=font(BOLD, 16), fill=INK)
    yb = y0 + 64 * len(rows)
    for v in (60, 70, 80, 90, 100):
        d.text((X(v) - 14, yb - 14), f"{v}%", font=font(SANS, 14), fill=MUTED)
    d.text((x0, yb + 6), "dot = success rate · bar = 95 % interval", font=font(SANS, 13), fill=MUTED)
    return im


def s_end(t):
    im, d = canvas()
    d.text((48, 190), "Try it", font=font(BOLD, 56), fill=INK)
    for i, (k, v) in enumerate([("Docs and VLA course", "mtbui2010.github.io/vla-serve"),
                                ("Code", "github.com/mtbui2010/vla-serve"),
                                ("Every command", "make help")]):
        d.text((52, 300 + 62 * i), k, font=font(SANS, 24), fill=MUTED)
        d.text((380, 298 + 62 * i), v, font=font(MONOB, 26), fill=BLUE)
    return im


# --- timeline ----------------------------------------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[w.getsampwidth()]).astype(np.float32) / \
            {2: 32768.0, 4: 2147483648.0}[w.getsampwidth()]


def main():
    nd = lambda sid: TIMING[sid]["duration"]
    sim, sim_dur = s_sim_factory(nd("s10_sim"))
    scenes = [
        ("s01_title", s_title),
        ("s02_problem", s_problem),
        ("s03_overview", s_overview),
        ("s04_build", step_scene("Step 1", "Build the server image (once)",
                                 [("make docker", ["…", "#13 DONE 6.8s"]),
                                  ("docker images vla-serve", txt("images.txt"))], pace=1.6)),
        ("s05_inspect", step_scene("Step 2", "Look at a checkpoint before loading it",
                                   [("vla-serve inspect $CKPT", json_lines("inspect.json"))], pace=2.0)),
        ("s06_check", step_scene("Step 3", "Check the robot matches the policy",
                                 [("vla-serve check $CKPT --profile kaair", txt("check_ok.txt")),
                                  ("vla-serve check $CKPT --profile libero", txt("check_bad.txt"))], pace=2.2)),
        ("s07_serve", step_scene("Step 4", "Start the server",
                                 [("make docker-run GPU=3 CKPT_ROOT=/mnt/nas/…/checkpoints MODEL=$CKPT",
                                   txt("serverlog.txt")[-2:]),
                                  ("make health", txt("health.txt")),
                                  ("make ps", txt("ps.txt"))], pace=1.4)),
        ("s08_ask", s_ask),
        ("s09_chunks", s_chunks),
        ("s10_sim", sim),
        ("s11_results", s_results),
        ("s12_end", s_end),
    ]
    durs = [sim_dur if sid == "s10_sim" else nd(sid) + LEAD + TAIL for sid, _ in scenes]
    print(f"{len(scenes)} scenes, {sum(durs):.1f} s")

    # audio: each scene's narration starts LEAD seconds into the scene
    track = np.zeros(int(sum(durs) * SR) + SR, np.float32)
    t0 = 0.0
    for (sid, _), dur in zip(scenes, durs):
        a = read_wav(DATA / "voice" / f"{sid}.wav")
        s = int((t0 + LEAD) * SR)
        track[s:s + len(a)] = a
        t0 += dur
    wav = OUT.with_suffix(".wav")
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())

    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow",
                           "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(OUT)],
                          stdin=subprocess.PIPE)
    fade = 0.3
    for (sid, fn), dur in zip(scenes, durs):
        for k in range(int(dur * FPS)):
            t = k / FPS
            im = fn(t)
            subtitles(im, sid, t)
            a = min(1.0, t / fade, (dur - t) / fade)
            if a < 1:
                im = Image.blend(Image.new("RGB", (W, H), BG), im, max(a, 0))
            ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    wav.unlink()
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
