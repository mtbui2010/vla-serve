# 7. Evaluating a policy

!!! abstract "In this chapter"
    - Three questions an evaluation can answer, and which test answers which.
    - Why random seeds decide individual episodes.
    - How many episodes you need before two success rates differ.

## Three questions

| Question | Test | Example in this repo |
|---|---|---|
| Does my serving code compute the same thing as training code? | **parity**: same input, same seed, compare outputs bit for bit | `tests/test_gpu_parity.py` — difference 0 |
| Does the policy succeed at the task? | **closed loop** in simulation | kaair_sim open-drawer, 50 scenes |
| Does it succeed on the robot? | closed loop on hardware | not done yet |

Parity alone is not enough. An external study exported SmolVLA to ONNX: per-step outputs agreed
with PyTorch to cosine similarity ≈ 0.98, yet LIBERO-Spatial success fell from 70 % to 41 %,
traced to a truncated prompt and a tokenizer mismatch
([arXiv 2609.14146](https://arxiv.org/abs/2609.14146), a single-author preprint). Small numeric
errors compound over hundreds of closed-loop steps.

## One episode, end to end

![One episode in kaair_sim](../assets/img/episode-filmstrip.png)

![Commanded actions and measured joints over the episode](../assets/img/episode-actions.png)

The blue line is what the server commanded, the orange line where the joint actually was; the
gap is the controller tracking the target. Grey lines mark where a new chunk took over.

## Seeds decide individual episodes

A flow-matching policy samples noise for every chunk ([chapter 3](03-action-heads.md)):

![Same observation, eight seeds](../assets/img/seeds.png)

If the evaluator does not seed that noise, each episode's result depends on how much randomness
the previous episodes consumed.

!!! measured "Measured"
    kaair_sim scene 4: succeeded inside a 50-episode run, but failed **5 of 5** times when run
    alone from a fresh process (in-process LeRobot) and 4 of 5 through vla-serve. A fresh process
    starts from the same default random state, so it drew the same unlucky noise every time.

**Seed the policy's noise per episode** (e.g. `torch.manual_seed(scene_seed)`) to make runs
repeatable, and run several noise seeds per scene to measure robustness on purpose.

## How many episodes?

Success over \(n\) episodes is a coin-flip count. Its 95 % interval (Wilson) for 50 episodes:

![Success rates with 95 % intervals](../assets/img/evaluation.png)

47/50 and 49/50 overlap heavily (Fisher exact test p = 0.62): **50 episodes cannot tell 94 %
from 98 %.** Running the *same* 50 scenes in both conditions helps: compare episode by episode
(McNemar's test on the scenes where the two disagree).

!!! measured "A worked example: is the server, or the device, to blame?"
    Office scene, same 50 scenes. Through vla-serve on GPU: 36/50. In-process on CPU: 42/50.
    Two things differ — the serving path *and* the device. Run vla-serve on CPU as well: 42/50,
    with the same five hard scenes failing as in-process. The serving path is cleared; what
    remains is device and noise, and that 36 vs 42 is itself not significant (p = 0.11).
    Change one thing at a time.

To pin a rate near 95 % to ±5 points you need about 73 episodes; to compare two policies that
differ by a few points, several hundred.

## Compare against today

The 2026-09-24 number (48/50) was produced before kaair_sim's scene scripts changed on
2026-09-28. Re-running yesterday's evaluation code is part of the experiment: compare a new
result only with a baseline run on the same code, the same scenes, the same day.

## What this means when you deploy

* Gate releases on closed-loop success, not on tensor agreement.
* Report counts and intervals ("47/50, 84–98 %"), not a bare percentage.
* Keep a list of hard scenes (like scene 4) and re-run them for every new checkpoint.

**Next:** [Deploying to hardware →](08-deployment.md)
