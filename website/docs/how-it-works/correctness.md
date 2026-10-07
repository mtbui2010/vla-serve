# Proving it is correct

A serving layer for robot policies has one job: return the action the policy would have returned.
"Looks reasonable" is not enough — a policy that is slightly off can still move the arm
smoothly into the wrong place. vlaserve is checked at three levels.

## Level 1 — contract tests (no GPU)

`pytest` runs 36 tests in about 8 s: codec bounds, spec derivation (including a fake checkpoint
whose `config.json` is stale, as found in practice), profile mismatches, HTTP status codes, and
the chunk queue's timing logic. Timing-sensitive tests are run five times in a row to make sure
they are not flaky.

## Level 2 — parity with LeRobot (GPU)

`pytest -m gpu` loads real checkpoints and compares our chunk with the chunk produced by
`lerobot.async_inference.PolicyServer._predict_action_chunk` — LeRobot's own server — using the
same policy object, observation and random seed.

!!! measured "Measured"
    | checkpoint | chunk | max \|difference\| |
    |---|---|---:|
    | `lerobot/pi05_libero_finetuned` | 50 × 7 | **0** |
    | KETI pi0.5 OpenArm (via the compatibility layer) | 30 × 16 | **0** |
    | kaair SmolVLA, served over HTTP from the container vs in-process | 50 × 8 | **0** |
    | kaair SmolVLA, LeRobot 0.4.4 vs 0.6.1, same noise, CPU | 50 × 8 | **0** |

Bit-exact, not "close": with the same seed, a flow-matching policy is deterministic, so any
difference at all means something in the path changed.

Getting to zero found three real issues:

1. **Resize order.** Resizing a `uint8` frame then dividing by 255 differs from dividing first
   by up to one grey level. (Later the server stopped resizing at all — see below.)
2. **`torch.compile` with autotuning** made two loads of the same checkpoint differ by ~1e-3.
   Hence eager mode by default.
3. **A checkpoint saved by LeRobot 0.5** named a processor step that 0.6 renamed. Loading
   failed outright until `compat.py` registered the old name.

## Level 3 — closed loop

Parity says "the same as LeRobot". It cannot say "the robot succeeds". That needs the policy
acting in a world: see [Test in simulation](../guides/simulation.md). On kaair_sim's open-drawer
task the served SmolVLA opened 47/50 drawers against 49/50 in-process (plain scene), and in the
photo-textured office scene **42/50 against 42/50** with both on CPU.

The closed-loop run also changed the design. That SmolVLA checkpoint's `config.json` declared the
base model's cameras (`camera1..3`, 256×256, 6-d state) while the policy actually consumed
`head`/`wrist` at 480×848 and an 8-d state. A server that trusted `config.json` — including
LeRobot's own policy server — would have squashed every frame to 256×256. vlaserve now reads the
dataset statistics first and passes frames through unresized.

## Level 0 — the end-to-end loop

`examples/robot_loop.py` with stub I/O against a live server, pi0.5 at 20 Hz for 15 s:
300 steps, **0 stalls**, about 6 stale actions dropped per chunk (= 0.32 s × 20 Hz, as predicted).
The first version of that test exposed a real bug: staleness was counted in clock ticks, so a
slow first chunk (warm-up) was discarded entirely. It is now counted in *executed actions*.
