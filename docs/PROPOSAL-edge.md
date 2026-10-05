# Proposal: inspect, convert and verify VLAs for edge devices

**Status: proposal only — nothing here is implemented.** Written 2026-10-05 after reading
VisionServe's inspect / converter / precision work (`../vision_serve`) and the measurements made in
this repo. Read `CLAUDE.md` and `HANDOFF.md` first.

The one idea to carry over from VisionServe: **every speed-up passes an automatic verification
gate before it is installed.** For VLAs the gate must include a closed-loop check, because an
export can match tensors and still lose task success (one SmolVLA ONNX study: LIBERO-Spatial 70 %
→ 41 % from a truncated prompt and a tokenizer mismatch).

---

## 1. What VisionServe does (reference)

| Part | How | Where |
|---|---|---|
| Inspect | Pure-Go ONNX header reader: graph I/O without reading weights (ms, even for a 695 MB file). Contract check: file shapes vs manifest. | `internal/engine/onnxheader.go`, guide `website/docs/guides/inspect.md` |
| Convert | Separate converter image (PyTorch/TF), so the server stays Python-free. License gate → export → contract check → install → verify tiers → report. | `clients/python/visionserve/convert/`, `deploy/Dockerfile.convert`, `internal/cli/convert.go` |
| Verification tiers | **A** ONNX vs framework · **B1** server preprocessing vs reference on real images, with a *diagnosis* of the cause (letterbox, mean/std, missing /255, RGB/BGR, resampling) · **B2** served outputs vs original pipeline · **C** mAP / top-1 on the user's data, reference vs served | `verify.py`, `diagnose.py`, `evaluate.py` |
| Precision | FP16, INT8 (QDQ), per-layer sensitivity, mixed precision (`--keep-fp-top N`). RF-DETR INT8: error 0.36; 69/120 layers had to stay float. | `precision.py`, guide `precision.md` |
| Bench | Framework / ONNX / server, p50 and p95 | `bench.py` |
| Report | FAIL > ERROR > WARN > PASS; `convert-report.json`; a FAIL uninstalls the model | `report.py` |
| Edge | TensorRT opt-in with an engine cache keyed by sha256; `Dockerfile.edge` for Jetson | `internal/engine/trt.go` |

---

## 2. Proposal for vla-serve

### 2.1 `inspect --deep` and `doctor`

- `vla-serve inspect MODEL --deep`
  - processor pipeline, `rename_map`, tokenizer and its max length, weight dtypes, estimated VRAM;
  - LeRobot version the checkpoint was saved with vs installed; whether `compat.py` is needed;
  - **statistics sanity**: dimensions with near-zero std (sensor noise gets amplified after
    normalisation), state/action dims that disagree with `config.json`.
- Inspect an exported ONNX / TRT engine: I/O and fixed shapes (prompt length, image size, number
  of denoising steps).
- `vla-serve doctor`: torch CUDA build vs driver (the cu130-on-a-12.8-driver failure met in this
  repo), GPU visible, TensorRT version, free RAM/VRAM. Run before anything else fails.

### 2.2 `vla-serve convert` (separate converter image, like VisionServe)

```
vla-serve convert CKPT --format onnx|trt --precision fp32|fp16|bf16|int8 [--target orin|thor]
```

Graph design, following exports others have made work (Arm's SmolVLA ONNX, NVIDIA's pi0.5/GR00T TRT):

- **Unroll the 10 Euler steps** into one graph; **noise is a graph input**, so runs stay seedable
  and parity stays testable.
- **Fixed shapes**: prompt length = `tokenizer_max_length` (the processor already pads to it:
  48 for SmolVLA, 200 for pi0.5); images after the policy's own resize-with-pad.
- **Processors stay outside the graph** (tokenizer, normaliser, relative/absolute actions) and keep
  using LeRobot's own pipeline. An ONNX backend = Python + onnxruntime / TensorRT EP behind the
  existing `backends.Backend` interface: still general, runs on Jetson.
- Fallback if the unrolled graph is too large: two graphs, *VLM prefix* (once per chunk, emits the
  KV cache) and *action expert step* (run N times).

### 2.3 Verification tiers for VLAs

| Tier | VisionServe | vla-serve |
|---|---|---|
| A | ONNX vs framework | ONNX/TRT vs PyTorch, **same noise**, max\|Δ\| per joint in robot units |
| B1 | preprocessing vs reference, with diagnosis | tokens, normalised state, image tensors: server vs LeRobot via a `/v1/debug/preprocess` endpoint; diagnose truncated prompt, wrong tokenizer, wrong stats |
| B2 | served vs original pipeline | exists (bit-exact parity tests); turn into `vla-serve verify`, on **real dataset frames** instead of random inputs |
| **B3 (new)** | — | **open-loop replay** on held-out dataset episodes: action error vs recorded actions. Cheap, no simulator |
| C | mAP on user data | **paired closed loop**: same scenes, *noise seeded per episode*, McNemar test (as done for kaair_sim). Plugins: kaair_sim, LIBERO, gym |

**Acceptance threshold relative to noise spread.** Measured here: with the *same* noise, GPU vs
CPU chunks differ by ≤ 0.018 rad; with *different* noise by up to 0.48 rad. Proposal: an export
passes when its same-noise error is **below ~10 % of the across-seed spread** per joint, instead
of fixed numbers like VisionServe's provisional 0.05 / 0.25.

### 2.4 Precision per component, with sensitivity

Reuse VisionServe's `--sensitivity` / `--keep-fp-top`, measured on *action* error. Expected (from
RF-DETR) most sensitive: attention softmax·V. Order to try: vision tower + VLM in FP16/BF16 →
action expert FP16 → INT8 only on large VLM MatMuls. Avoid full INT8: post-training INT8 bought
little on a transformer detector.

### 2.5 Bench in control terms

Besides p50/p95 at three levels:

```
vla-serve bench MODEL --fps 30 --chunk 50
```

answers *"does it keep up"*: latency vs chunk duration, minimum `refill_at` = latency × fps,
expected stalls.

### 2.6 Edge (Jetson)

- One image per JetPack: Orin = JP6, Thor = JP7. **Build TRT engines on the device** (not portable).
- Engine cache key = *checkpoint weights hash + shapes + TRT version + GPU*. VisionServe hit a
  collision between two checkpoints of the same graph, so the weights hash must be in the key.
- Convert report saved next to the checkpoint; FAIL = not installed.

---

## 3. Roadmap

| # | Step | Cost | Risk |
|---|---|---|---|
| 1 | `doctor` + `inspect --deep` | days | none |
| 2 | Pilot: SmolVLA → ONNX FP32 → FP16; tiers A → B1 → B2 → B3 → C on kaair_sim | 1–2 weeks | medium |
| 3 | TensorRT FP16 for SmolVLA on the A6000, then on a Jetson when available | 1 week | medium |
| 4 | pi0.5 along NVIDIA's path (Thor, FP8) when a Thor is available | weeks | high |

**Biggest risk:** an export that is right in tensors but loses closed-loop success. So tier C and
the noise-relative threshold come **before** speed work, not after.

## 4. Inputs worth reusing

- VisionServe converter: report/status model, sensitivity tooling, diagnosis approach.
- This repo: `tests/test_gpu_parity.py` (tier B2), `examples/kaair_sim_eval.py` (tier C),
  `website/tools/probe_figs.py` (seed-spread measurement for the threshold).
- Published edge numbers (indicative, vendor/author-reported): pi0.5 on Thor ≈ 132 ms PyTorch →
  ≈ 54 ms TRT FP8; GR00T N1.7 ≈ 40 ms TRT; pi0.5 on Orin ≈ 1420 ms PyTorch → ≈ 413 ms (Jetson-PI).
  Sources in `website/docs/learn/08-deployment.md`.

## 5. Open questions

- Does SmolVLA's GPU run on the office scene (36/50 vs 42/50 on CPU, p = 0.11) reflect a real
  device effect? Settle with an in-process GPU baseline before trusting any GPU-side comparison.
- Unrolled single graph vs prefix + expert split: which exports cleanly for pi0.5 with ONNX opset
  support in the current exporters?
- Where should the ONNX backend's tokenizer run on a Jetson without the full LeRobot install?
