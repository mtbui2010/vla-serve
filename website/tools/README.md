# Figures for the documentation site

Every figure on the site is drawn from data recorded on real runs. To regenerate:

1. **Start a server** with the office-scene SmolVLA (GPU 3 on the development machine):

   ```bash
   make docker-run GPU=3 CKPT_ROOT=/mnt/nas/lerobot_shared/kaair_sim/checkpoints \
     MODEL=/ckpt/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model
   ```

2. **Record one episode** inside the `kaair_sim_humble` container (client copied to `/tmp`):

   ```bash
   docker exec kaair_sim_humble mkdir -p /tmp/vlaclient /tmp/docsfig
   docker cp vlaserve kaair_sim_humble:/tmp/vlaclient/
   docker cp website/tools/record_episode.py kaair_sim_humble:/tmp/docsfig/
   docker exec kaair_sim_humble /sim/.venv/bin/python -m pip install -q --target /tmp/vlaclient/deps msgpack
   docker exec kaair_sim_humble bash -lc 'cd /sim && PYTHONPATH=/tmp/vlaclient:/tmp/vlaclient/deps \
     /sim/.venv/bin/python /tmp/docsfig/record_episode.py --camera-model --seed 1000000000 \
     --model /ckpt/smolvla_open_drawer_real500_multidrawer_20260928/checkpoints/020000/pretrained_model'
   docker cp kaair_sim_humble:/tmp/docsfig/. DATA/          # episode.npz + episode.mp4
   ```

3. **Probe** the flow-matching steps, seed spread and a live executor timeline (host, `vlaserve`
   env; needs the server from step 1 and the checkpoint path inside the script):

   ```bash
   CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python website/tools/probe_figs.py DATA seeds,flow
   CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python website/tools/probe_figs.py DATA timeline   # idle server only
   ```

4. **Latency and statistics**: `DATA/latency.json` holds the median `total` of
   `vla-serve bench <model> -n 20` for each model; `DATA/stats.npz` holds `mean`/`std` of
   `observation.state` from the checkpoint's normaliser file.

5. **Draw**: `python website/tools/figures.py DATA` → `website/docs/assets/img/*.png`.

6. **Video**: capture the command outputs into `DATA/out/` (`inspect.json`, `check_ok.txt`,
   `check_bad.txt`, `health.txt`, `serverlog.txt`, `client.txt`, `images.txt`), copy
   `video_results.json` to `DATA/results.json`, then
   `python website/tools/make_video.py DATA` → `website/docs/assets/video/vla-serve-tutorial.mp4`.
   Poster: `ffmpeg -ss 55 -i …mp4 -frames:v 1 …-poster.png`.

Recorded 2026-10-05 on one RTX A6000 (shared machine), office scene (photo textures, camera
model on), scene seed 10⁹, task "open the bottom drawer": success at 9.6 s, drawer opened
25.4 cm, 319 control steps, 7 chunks.
