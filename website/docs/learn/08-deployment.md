# 8. Deploying to hardware

!!! abstract "In this chapter"
    - Matching PyTorch, CUDA and the GPU driver.
    - What edge devices can do today, with published numbers.
    - What exporting a VLA to ONNX/TensorRT involves.
    - Licences.

## PyTorch, CUDA and the driver

A PyTorch wheel is built for one CUDA version, and the GPU driver supports CUDA up to some
version. If the wheel's CUDA is newer than the driver's, `torch.cuda.is_available()` returns
`False` with only a warning — and everything quietly runs on the CPU.

!!! measured "Measured"
    On the development machine (driver supporting CUDA 12.8), `pip install lerobot` pulled
    torch 2.11 built for CUDA 13.0: no GPU. Installing torch 2.10 from the `cu128` index first
    fixed it. `make env` and the Dockerfile do this for you; for an older driver use
    `TORCH_CUDA=cu126`.

## Containers

The container image fixes that combination once and is tested end to end: the GPU parity tests
pass inside it. See [Run it in a container](../guides/container.md).

## Edge devices

Published numbers for pi0.5-class models on NVIDIA Jetson (vendor or author measurements, single
configuration, batch 1 — treat as indicative):

| device | setup | time per chunk | source |
|---|---|---:|---|
| Jetson AGX Orin | pi0.5, plain PyTorch | ≈ 1420 ms | [Jetson-PI](https://arxiv.org/abs/2607.12659) |
| Jetson AGX Orin | pi0.5, Jetson-PI optimised | ≈ 413 ms | same |
| Jetson AGX Thor | pi0.5, PyTorch BF16 | ≈ 132 ms | [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor) |
| Jetson AGX Thor | pi0.5, TensorRT FP8 | ≈ 54 ms | same |
| Jetson AGX Thor | GR00T N1.7, TensorRT | ≈ 40 ms | [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/) |

On Orin-class hardware a large VLA only works with asynchronous execution
([chapter 6](06-real-time.md)) and long chunks; Thor-class hardware is fast enough to keep a
30 Hz robot comfortably fed. vla-serve's image is x86-64 only for now.

## Exporting to ONNX / TensorRT

The speed-ups above come from compiling the network for the device. For a flow-matching VLA that
means:

1. **Unroll the denoising loop**: the 10 Euler steps become part of one graph, and the initial
   noise becomes a graph *input* (so it can still be seeded).
2. **Fix the shapes**: prompt length (e.g. 48 or 200 tokens), image size, chunk length.
   Truncating the prompt to a shorter fixed width is exactly the kind of change that broke
   closed-loop success in the study cited in [chapter 7](07-evaluation.md).
3. **Keep the pre/postprocessing outside**: tokenizer, normalisation statistics, relative
   actions. Those must be re-implemented around the graph, per checkpoint.

vla-serve keeps a backend interface for this
([`backends/base.py`](https://github.com/mtbui2010/vla-serve/blob/main/vlaserve/backends/base.py))
but has no ONNX backend yet: any export must pass the parity tests **and** a closed-loop check
before it replaces PyTorch.

## Licences

The weights' licence comes from the backbone. pi0 / pi0.5 / pi0-FAST build on PaliGemma and
fall under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms): usable in research,
but whoever redistributes a fine-tune must pass the use restrictions on. Check every model card
before shipping; `vla-serve inspect` reports what it can infer.

## Checklist

- [ ] torch's CUDA build ≤ the driver's CUDA (`torch.cuda.is_available()` is `True`)
- [ ] `vla-serve check` passes for the exact robot profile
- [ ] round-trip latency × fps < `refill_at`; `stalls == 0` in a dry run
- [ ] closed-loop success measured with seeded noise, counts + interval reported
- [ ] the robot's own velocity / workspace limits and an emergency stop are active
- [ ] the licence of every model you ship is known
