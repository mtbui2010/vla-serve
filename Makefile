# Makefile — vla-serve
# `make help` lists every target. Parameters go on the command line: make serve MODEL=pi05-libero GPU=3

# --- Environment (local, non-container targets) ---
CONDA    ?= conda
ENV      ?= vlaserve
ENV_BIN  ?= $(shell $(CONDA) env list 2>/dev/null | awk '$$1=="$(ENV)" {print $$NF}')/bin
# The torch build must match the host driver (this host: CUDA 12.8). PyPI's default is newer.
TORCH_CUDA ?= cu128
TORCH_VERSION ?= 2.10.0
TORCHVISION_VERSION ?= 0.25.0

# --- Defaults for serve / inspect / check / bench ---
MODEL    ?= pi05-libero
PROFILE  ?=
HOST     ?= 127.0.0.1
PORT     ?= 11450
URL      ?= http://127.0.0.1:$(PORT)
# GPU index; empty = all GPUs (docker) / whatever CUDA picks (local).
GPU      ?=
HF_CACHE ?= $(HOME)/.cache/huggingface
OFFLINE  ?= 0

# --- Docker ---
IMAGE     ?= vla-serve:gpu-$(TORCH_CUDA)
CONTAINER ?= vla-serve
# Optional host directory of checkpoints, mounted read-only at /ckpt; then MODEL=/ckpt/<...>.
CKPT_ROOT ?=
DOCKER_GPUS := $(if $(GPU),'"device=$(GPU)"',all)

LOCAL_ENV := $(if $(GPU),CUDA_VISIBLE_DEVICES=$(GPU)) $(if $(filter 1,$(OFFLINE)),HF_HUB_OFFLINE=1) HF_HOME=$(HF_CACHE)

.PHONY: help env install-client list inspect check serve bench ps load rm reset health \
        test test-gpu docker docker-run docker-logs docker-stop docker-shell docker-test \
        docker-test-gpu docker-save kaair-eval clean

help: ## Print the list of targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## --- Setup ---

env: ## Create the conda env ENV=vlaserve with torch (TORCH_CUDA=cu128) + server + tests
	$(CONDA) create -y -n $(ENV) python=3.12
	$(MAKE) --no-print-directory _install ENV_BIN="$$($(CONDA) env list | awk '$$1=="$(ENV)" {print $$NF}')/bin"

_install:
	$(ENV_BIN)/python -m pip install torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION) \
		--index-url https://download.pytorch.org/whl/$(TORCH_CUDA)
	$(ENV_BIN)/python -m pip install -e '.[server,test]' torch==$(TORCH_VERSION) \
		torchvision==$(TORCHVISION_VERSION) --extra-index-url https://download.pytorch.org/whl/$(TORCH_CUDA)
	$(ENV_BIN)/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

install-client: ## Robot side: install only the light client into the current Python (no torch)
	python -m pip install .

## --- Inspect (no GPU, works with the client-only install) ---

list: ## List model aliases and robot profiles
	$(LOCAL_ENV) $(ENV_BIN)/vla-serve list

inspect: ## Cameras, state/action size, chunk, licence: make inspect MODEL=<alias|repo|path>
	$(LOCAL_ENV) $(ENV_BIN)/vla-serve inspect $(MODEL)

check: ## Robot profile vs policy: make check MODEL=... PROFILE=openarm-bimanual
	@test -n "$(PROFILE)" || { echo "usage: make check MODEL=... PROFILE=<name|file.yaml>"; exit 2; }
	$(LOCAL_ENV) $(ENV_BIN)/vla-serve check $(MODEL) --profile $(PROFILE)

## --- Run locally (conda env) ---

serve: ## Start the server: make serve MODEL=... [GPU=3] [PORT=11450] [HOST=127.0.0.1] [OFFLINE=1]
	$(LOCAL_ENV) $(ENV_BIN)/vla-serve serve --host $(HOST) --port $(PORT) $(MODEL)

bench: ## In-process latency, synthetic inputs: make bench MODEL=... [GPU=3]
	$(LOCAL_ENV) $(ENV_BIN)/vla-serve bench $(MODEL)

## --- Talk to a running server (local or container) ---

health: ## GET /health [URL=http://127.0.0.1:11450]
	@curl -sf $(URL)/health && echo || { echo "no server at $(URL)"; exit 1; }

ps: ## Models loaded in the running server
	@curl -sf $(URL)/v1/models | python3 -c "import json,sys; [print(f\"{m['model']}  {m['device']}  calls={m['calls']}  licence={m['license']}\") for m in json.load(sys.stdin)['models']] or print('(none loaded)')"

load: ## Load a model into the running server: make load MODEL=...
	curl -sf -X POST $(URL)/v1/models/load -H 'content-type: application/json' -d '{"model":"$(MODEL)"}' >/dev/null && echo "loaded $(MODEL)"

rm: ## Unload a model from the running server: make rm MODEL=...
	curl -sf -X POST $(URL)/v1/models/unload -H 'content-type: application/json' -d '{"model":"$(MODEL)"}' && echo

reset: ## Reset a model's episode state: make reset MODEL=...
	curl -sf -X POST $(URL)/v1/models/reset -H 'content-type: application/json' -d '{"model":"$(MODEL)"}' && echo

## --- Tests ---

test: ## Fast tests (no GPU)
	HF_HUB_OFFLINE=1 $(ENV_BIN)/python -m pytest -q

test-gpu: ## Parity vs LeRobot on real checkpoints: make test-gpu GPU=3 [HF_CACHE=/mnt/nas/huggingface]
	$(LOCAL_ENV) HF_HUB_OFFLINE=1 $(ENV_BIN)/python -m pytest -m gpu -s -q tests/test_gpu_parity.py

## --- Docker ---

docker: ## Build the GPU server image IMAGE=vla-serve:gpu-cu128 [TORCH_CUDA=cu126]
	docker build -f deploy/Dockerfile --build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
		-t $(IMAGE) .

docker-run: ## Start the server container: make docker-run MODEL=... [GPU=3] [CKPT_ROOT=/host/ckpts] [PORT] [OFFLINE=1]
	@if docker ps -a --format '{{.Names}}' | grep -qx '$(CONTAINER)'; then \
		echo "container '$(CONTAINER)' exists; run 'make docker-stop' first or set CONTAINER=<name>"; exit 1; fi
	docker run -d --name $(CONTAINER) --gpus $(DOCKER_GPUS) \
		-p $(HOST):$(PORT):11450 \
		-v $(HF_CACHE):/root/.cache/huggingface \
		$(if $(CKPT_ROOT),-v $(CKPT_ROOT):/ckpt:ro) \
		$(if $(wildcard $(HOME)/.config/vla-serve),-v $(HOME)/.config/vla-serve:/root/.config/vla-serve:ro) \
		$(if $(filter 1,$(OFFLINE)),-e HF_HUB_OFFLINE=1) \
		$(IMAGE) serve --host 0.0.0.0 $(MODEL)
	@echo "starting; loading takes 1.5-5 min. Watch: make docker-logs   Ready: make health"

docker-logs: ## Follow the server container's log
	docker logs -f $(CONTAINER)

docker-stop: ## Stop and remove the server container
	-docker stop $(CONTAINER) >/dev/null && docker rm $(CONTAINER) >/dev/null && echo "removed $(CONTAINER)"

docker-shell: ## Shell inside the running server container
	docker exec -it $(CONTAINER) bash

docker-test: ## Fast tests inside the image
	docker run --rm -e HF_HUB_OFFLINE=1 --entrypoint python $(IMAGE) -m pytest -q -p no:warnings tests

docker-test-gpu: ## Parity tests inside the image: make docker-test-gpu GPU=3 [HF_CACHE=/mnt/nas/huggingface]
	docker run --rm --gpus $(DOCKER_GPUS) -e HF_HUB_OFFLINE=1 -v $(HF_CACHE):/root/.cache/huggingface \
		--entrypoint python $(IMAGE) -m pytest -m gpu -s -q -p no:warnings tests/test_gpu_parity.py

docker-save: ## Export the image for another machine: vla-serve-<tag>.tar.gz (~8.5 GB before gzip)
	docker save $(IMAGE) | gzip > $(subst :,-,$(IMAGE)).tar.gz
	@echo "on the other machine: docker load < $(subst :,-,$(IMAGE)).tar.gz"

## --- kaair_sim closed-loop test (see examples/kaair_sim_eval.py) ---

KAAIR_CONTAINER ?= kaair_sim_humble
KAAIR_CKPT ?= smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model
N   ?= 50
TAG ?= vlaserve_$(shell date +%Y%m%d)

kaair-eval: ## Sim eval through a server started with CKPT_ROOT=/mnt/nas/lerobot_shared/kaair_sim/checkpoints [KAAIR_CKPT=... N=50]
	docker exec $(KAAIR_CONTAINER) mkdir -p /tmp/vlaclient
	docker cp vlaserve $(KAAIR_CONTAINER):/tmp/vlaclient/
	docker cp examples/kaair_sim_eval.py $(KAAIR_CONTAINER):/tmp/vlaclient/
	docker exec $(KAAIR_CONTAINER) /sim/.venv/bin/python -m pip install -q --target /tmp/vlaclient/deps msgpack
	docker exec $(KAAIR_CONTAINER) bash -lc 'cd /sim && PYTHONPATH=/tmp/vlaclient:/tmp/vlaclient/deps \
		/sim/.venv/bin/python /tmp/vlaclient/kaair_sim_eval.py --url $(URL) \
		--server-model /ckpt/$(KAAIR_CKPT) --ckpt /nas/checkpoints/$(KAAIR_CKPT) \
		--task task_open_drawer_generic.yaml --tag $(TAG) --videos 0 --n $(N)'
	docker exec $(KAAIR_CONTAINER) rm -rf /tmp/vlaclient

clean: ## Remove caches and build artefacts
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist logs
