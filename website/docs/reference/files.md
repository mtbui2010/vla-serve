# Registry and robot profiles

## Registry — `models.yaml`

Bundled: `vlaserve/data/models.yaml`. Yours: `~/.config/vla-serve/models.yaml` (or
`$VLASERVE_REGISTRY`), which overrides bundled entries of the same name.

```yaml
models:
  <alias>:
    source: <HF repo id or local path>      # required
    revision: <git revision>                # optional; pin what you tested
    backend: lerobot                        # default; the only backend today
    license: <id>                           # optional; leave out rather than guess
    description: <text>
    profile: <robot profile name>           # default profile for this model
    options: {compile: false}               # passed to the backend
```

Unknown keys are an error, so a typo is caught at load time.

## Robot profile

Bundled: `vlaserve/data/profiles/*.yaml` (`libero`, `openarm-bimanual`). Use your own by path.

```yaml
name: kaair
fps: 30                         # control rate the training data was recorded at
cameras:                        # policy camera name -> your camera driver's name
  head: femto_color
  wrist: d405_color
state_dim: 8
action_dim: 8
action_names: [joint1, ...]     # optional; if both sides list them, order is checked
notes: free text
```

`vla-serve check MODEL --profile kaair.yaml` compares the two and lists every disagreement.
