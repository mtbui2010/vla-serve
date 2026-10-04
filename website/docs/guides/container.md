# Run it in a container

The server needs PyTorch, CUDA libraries and LeRobot: about 8.5 GB that must match each other and
the GPU driver. The container ships that combination already tested, so the host only needs
Docker and an NVIDIA driver.

## Build

```bash
make docker                          # → vla-serve:gpu-cu128
make docker TORCH_CUDA=cu126         # host driver older than 570
make docker-test                     # fast tests inside the image
```

The image uses `python:3.12-slim` as base, not `nvidia/cuda`: the CUDA and cuDNN libraries come
inside the torch wheel, and the driver is injected by the NVIDIA runtime at `docker run`.
**No model weights are baked in.**

## Run

```bash
make docker-run MODEL=<name or path> [GPU=3] [CKPT_ROOT=/host/checkpoints] [PORT=11450] [OFFLINE=1]
```

which runs, for example:

```bash
docker run -d --name vla-serve --gpus '"device=3"' -p 127.0.0.1:11450:11450 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nas/lerobot_shared/kaair_sim/checkpoints:/ckpt:ro \
  vla-serve:gpu-cu128 serve --host 0.0.0.0 \
  /ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model
```

| Part | Why |
|---|---|
| `--gpus '"device=3"'` | pick a GPU; on a shared machine check `nvidia-smi` first |
| `-p 127.0.0.1:11450:11450` | only this machine can call it. Drop `127.0.0.1:` to let a robot on the LAN connect — anyone who reaches the port can move the robot |
| `-v ~/.cache/huggingface:…` | **always needed**, even for local checkpoints: tokenizers (PaliGemma, SmolVLM2) are fetched by name |
| `-v …:/ckpt:ro` | your checkpoints, read-only. The model name is then the path **inside** the container (`/ckpt/...`) |
| `--host 0.0.0.0` | required inside a container |
| `-e HF_HUB_OFFLINE=1` (`OFFLINE=1`) | no internet; everything must already be in the cache |

## Watch it start

```bash
make docker-logs        # wait for: "loaded <model> on cuda in 98.7s"
make health
make ps
```

`docker ps` shows `(health: starting)` until the model is in memory; the healthcheck allows ten
minutes.

## Manage models without restarting

```bash
make load  MODEL=lerobot/pi05_libero_finetuned
make rm    MODEL=lerobot/pi05_libero_finetuned
make reset MODEL=...                       # start of a new episode
make docker-shell                          # then: vla-serve inspect /ckpt/...
make docker-stop
```

## Another machine

No registry is published yet. Copy the image by hand:

```bash
make docker-save                           # → vla-serve-gpu-cu128.tar.gz
docker load < vla-serve-gpu-cu128.tar.gz   # on the other machine
```

## Verify an image end to end

```bash
make docker-test-gpu GPU=3 HF_CACHE=/mnt/nas/huggingface
```

runs the [parity tests](../how-it-works/correctness.md) inside the image: each served chunk
must equal the chunk LeRobot's own policy server produces, bit for bit.

!!! warning "Jetson"
    The image is x86-64 only. Jetson Orin (JetPack 6) and Thor (JetPack 7) need NVIDIA's L4T
    PyTorch base for each JetPack version; see [Deploying to hardware](../learn/08-deployment.md).
