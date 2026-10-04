# Models, aliases and licences

## Three ways to name a model

```bash
vla-serve serve lerobot/pi05_libero_finetuned          # Hugging Face repo id (through the HF cache)
vla-serve serve /data/runs/x/checkpoints/last/pretrained_model   # local directory
vla-serve serve pi05-libero                            # alias from a registry file
```

Anything in LeRobot's checkpoint format works without registration. An alias only adds a short
name, a pinned revision and a licence.

## Your own aliases

```yaml title="~/.config/vla-serve/models.yaml"
models:
  kaair-drawer:
    source: /ckpt/smolvla_open_drawer_500_20260924/checkpoints/020000/pretrained_model
    description: SmolVLA, open drawer, kaair_sim, step 20k
    profile: kaair
  openarm-handover:
    source: KETI-IRRC/pi05_openarm_handover_v0_clean_alpha2prime
    revision: d12580bbf417e58777bb16cbe1bc212e8d486ef8     # pin exactly what you tested
```

Your file overrides the bundled one. In a container, mount it:
`-v ~/.config/vla-serve:/root/.config/vla-serve:ro` (`make docker-run` does this automatically if
the directory exists).

## Licences: recorded, warned, never blocked

Model weights carry their own licence, independent of the code that runs them. vla-serve reports
one for every model, from — in order — your registry entry, the checkpoint's config, or what it
can infer:

| what it finds | reported as |
|---|---|
| `license:` in your registry entry | that value |
| `license` in `config.json` | that value (rarely set; every LeRobot checkpoint seen so far has `null`) |
| a PaliGemma backbone (pi0, pi0.5, pi0-FAST) | `gemma-terms` + a warning |
| nothing | `unknown` + a warning |

Why PaliGemma matters: under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
fine-tunes are "Model Derivatives", and whoever redistributes them must pass the use restrictions
on. That is not Apache, MIT or BSD. It does not stop you using pi0.5 in the lab; it matters when
you ship a product. (Gemma 4 moved to Apache-2.0; PaliGemma-based policies today are on the older
terms.)

vla-serve does not block anything, by design: it serves your own lab's checkpoints, and the goal
is that whoever ships a robot knows what they ship. Its sibling VisionServe takes the opposite,
stricter line for a different reason; see its documentation.
