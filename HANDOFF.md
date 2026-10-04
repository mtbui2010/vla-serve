# Handoff

Started 2026-10-04. Read `CLAUDE.md` (rules and settled decisions) first, then this.

## Where it stands

**Works and is verified**

- Load any LeRobot 0.6 checkpoint by alias / HF repo id / local path (`registry.py`), spec read
  from `config.json` without loading weights (`spec.py`).
- `compat.py`: checkpoints saved by LeRobot 0.5.x that name `delta_actions_processor` load under
  0.6.1 (renamed to `relative_actions_processor`; fields verified identical). Without it the KETI
  OpenArm checkpoint fails with `Processor step 'delta_actions_processor' not found` — observed.
- HTTP server + msgpack codec + client + `ChunkExecutor`; 34 fast tests.
- GPU parity against LeRobot's policy server on two real pi0.5 checkpoints — see
  *Measurements* below.
- **Client/server split.** `pip install vla-serve` = robot-side client only (83 MB, no torch;
  verified in a fresh venv and by `tests/test_client_is_light.py`). `[server]` adds LeRobot.
- **Container** `vla-serve:gpu-cu128` (8.5 GB, `deploy/Dockerfile`, guide in `deploy/README.md`).
  Fast tests pass inside it; GPU parity inside it: see *Measurements*.

**Not done**

1. **Closed-loop check in simulation** (`vla-serve check --sim libero`). The single most useful
   next feature: parity proves we match LeRobot, not that the policy works on the robot setup.
   `lerobot/pi05_libero_finetuned` + the LIBERO install in the old `lerobot` env
   (`remote_dir/LIBERO`) are the starting point. LIBERO is not installed in `vlaserve` yet.
2. **Other policy types.** Only pi0.5 is verified. Next: SmolVLA (needs `lerobot/smolvla_base`
   downloaded, ~1 GB, and a fine-tuned checkpoint to be meaningful), then GR00T (`lerobot[groot]`).
   Add each to `VLASERVE_PARITY_MODELS` once it passes.
3. **ONNX / TensorRT backend.** Interface exists (`backends/base.py`). Build it only for a model
   that has a working export, and gate it on parity *and* a closed-loop check — an export can
   match tensors and still lose task success.
4. **Smoother chunk merging.** `ActionQueue.on_chunk` is latest-wins. Real-time chunking /
   temporal ensembling (LeRobot has `policies/rtc/`) can replace it in one place.
5. **JPEG transport** for WiFi robots (raw frames are ~8 MB per 3x720p request).
6. **ROS 2 node** wrapping `ChunkExecutor`.
7. `examples/robot_loop.py` has stub robot I/O; it has not been run against a real robot.
8. **Jetson images** (Orin/JetPack 6, Thor/JetPack 7): need an L4T PyTorch base per JetPack and a
   device to test on. Image not pushed to any registry.
9. Not git-initialised; no GitHub remote. The docs site (`website/`, workflow in
   `.github/workflows/docs.yml`) builds with `mkdocs build --strict` but is not published: it needs
   the repository on GitHub (assumed `mtbui2010/vla-serve`; change `site_url`/`repo_url` in
   `website/mkdocs.yml` otherwise) and Pages set to "GitHub Actions".
10. Docs figures come from data recorded 2026-10-05 (`website/tools/README.md`). Re-record them
   if the served models or the executor change.

Docker on this host: data-root stays on the SSD (`/var/lib/docker`). `/data` was considered and
rejected: it is an HDD (WD Ultrastar, ROTA=1) shared with the `ketiserver` account. Old images
and stopped containers were cleaned instead (root 26 GB → 146 GB free, 2026-10-04).

## Things that will bite you

- **torch build vs driver.** The driver supports CUDA 12.8; `pip install lerobot` pulls a cu130
  torch that reports `cuda.is_available() == False` with only a warning. Install torch from the
  cu128 index first, and pin it again whenever you add a LeRobot extra (pip will try to upgrade).
- **`config.json` lies after fine-tuning from a base model.** kaair_sim's SmolVLA keeps
  smolvla_base's `camera1..3` 256x256 + 6-d state; the real inputs are `head`/`wrist` + 8-d state,
  bridged by `rename_map`. The spec reads normaliser stats first for exactly this reason, and
  the server never resizes to the declared shape (LeRobot's own policy server does, and would
  squash 480x848 frames to 256x256 for this checkpoint).
- **Postprocessing is per step.** LeRobot's server applies the postprocessor to each of the H
  actions separately (it is written for `(B, D)`). We do the same. Don't batch it without
  re-running parity.
- **Policy cameras nobody sends** (`empty_camera_*`, or `camera3` on a renamed SmolVLA) are listed
  in `spec.policy_fills`; the policy pads them itself.
- **Relative actions** (`use_relative_actions`): the preprocessor caches the state and the
  postprocessor adds it back. Pre and post must be called as a pair, under the model's lock.
- **Loading from the NAS is slow** (~4-5 min per pi0.5 checkpoint cold).

## Measurements

2026-10-04, one RTX A6000 (shared machine; GPU otherwise idle), bf16, eager, batch 1.

**Parity** (`tests/test_gpu_parity.py`) — our chunk vs LeRobot 0.6.1's
`PolicyServer._predict_action_chunk`, same objects, same seed, frames at a non-native resolution:

| checkpoint | chunk | max \|diff\| |
|---|---|---:|
| `lerobot/pi05_libero_finetuned` | 50 × 7 | **0** (seeds 0, 1) |
| `KETI-IRRC/pi05_openarm_handover_v0_clean_alpha2prime` (via `compat`) | 30 × 16 | **0** (seeds 0, 1) |

Before forcing eager, `pi05_libero_finetuned` (which ships `compile_model: true, max-autotune`)
differed by 9.3e-4 and its first call took 46 s. Hence eager by default.

**Latency per chunk**, server-side `timings_ms.total`: pi05-libero ~306-339 ms eager (one compiled
sample: 180 ms — not measured properly, do that before recommending compile); KETI OpenArm
~315 ms (3 cameras, preprocess ~19 ms of it).

**End to end** (`examples/robot_loop.py`, stub robot, pi05-libero, 20 Hz, 15 s, two runs):
300 steps, **0 stalls**, 7 startup ticks, 12 chunks, ~6 stale actions dropped per chunk
(= 0.32 s × 20 Hz, as expected), HTTP round trip median 321-328 ms, p95 341-380 ms. Bad msgpack →
400, unknown model → 409.

**SmolVLA, kaair_sim open-drawer** (`smolvla_open_drawer_500_20260924`, step 020000;
`examples/kaair_sim_eval.py`, server in the `vla-serve:gpu-cu128` container, client in
`kaair_sim_humble`, kaair's own `eval_policy.run()` with only the policy swapped):

| check | result |
|---|---|
| served (HTTP, container, GPU) vs in-process LeRobot 0.6.1 GPU, same obs + seed | max\|diff\| **0** (seeds 0, 1) |
| LeRobot 0.4.4 (kaair's training/eval version) vs 0.6.1, CPU, same obs + noise | max\|diff\| **0** |
| closed loop, 50 episodes, seeds 10⁹+k, `task_open_drawer_generic.yaml` | vla-serve **47/50 (94 %)**; in-process CPU baseline today 49/50 (98 %); recorded 2026-09-24 48/50 (96 %) |
| episode 4 alone, 5 fresh runs each | in-process **0/5**, vla-serve 1/5 — a hard scene for the policy, not a serving error |

The 94 vs 98 % gap is within noise (Fisher p ≈ 0.6) and every failing episode is explained by
SmolVLA's sampling noise: unseeded runs differ by up to 0.06 rad between noise draws. The
recorded run predates kaair's 2026-09-28 changes to `randomize.py`/`drawer_ready.py`/`eval_policy.py`.
HTTP round trip median 273 ms (SmolVLA, raw 2 × 480x848 frames); irrelevant here because the
sim is synchronous. Note: `kaair_sim_humble` had lost GPU access during these runs ("Failed to
initialize NVML"), so MuJoCo rendered on CPU (~31 s/episode instead of ~9 s).

This checkpoint is what forced the stats-first spec and the removal of server-side resizing.

The first end-to-end run exposed a bug, since fixed and regression-tested: stale actions were
counted in control ticks, so a slow first chunk (warmup, 92 ticks) was discarded entirely.
Staleness is now counted in *executed actions*; idle time does not age a plan.
