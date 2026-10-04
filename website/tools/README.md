# Figures for the documentation site

Every figure on the site is drawn from data recorded on real runs. To regenerate:

1. **Start a server** with the kaair SmolVLA checkpoint (GPU 3 on the development machine):

   ```bash
   make docker-run GPU=3 CKPT_ROOT=/mnt/nas/lerobot_shared/kaair_sim/checkpoints \
     MODEL=/ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model
   ```

2. **Record one episode** inside the `kaair_sim_humble` container (client copied to `/tmp`):

   ```bash
   docker exec kaair_sim_humble mkdir -p /tmp/vlaclient /tmp/docsfig
   docker cp vlaserve kaair_sim_humble:/tmp/vlaclient/
   docker cp website/tools/record_episode.py kaair_sim_humble:/tmp/docsfig/
   docker exec kaair_sim_humble /sim/.venv/bin/python -m pip install -q --target /tmp/vlaclient/deps msgpack
   docker exec kaair_sim_humble bash -lc 'cd /sim && PYTHONPATH=/tmp/vlaclient:/tmp/vlaclient/deps \
     /sim/.venv/bin/python /tmp/docsfig/record_episode.py'
   docker cp kaair_sim_humble:/tmp/docsfig/episode.npz DATA/
   ```

3. **Probe** the flow-matching steps, seed spread and a live executor timeline (host, `vlaserve`
   env; needs the server from step 1 and the checkpoint path inside the script):

   ```bash
   CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python website/tools/probe_figs.py DATA
   ```

4. **Latency and statistics**: `DATA/latency.json` holds the median `total` of
   `vla-serve bench <model> -n 20` for each model; `DATA/stats.npz` holds `mean`/`std` of
   `observation.state` from the checkpoint's normaliser file.

5. **Draw**: `python website/tools/figures.py DATA` → `website/docs/assets/img/*.png`.

Recorded 2026-10-05 on one RTX A6000 (shared machine). Episode: scene seed 10⁹, success at 7.8 s,
drawer opened 25.8 cm, 265 control steps, 6 chunks.
