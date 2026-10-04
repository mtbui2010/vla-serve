# Learn VLAs

A short course on Vision-Language-Action models, built on this repository. Each chapter explains
one idea, shows it in real code or a real checkpoint you can open, and ends with what it means
when you deploy a policy.

!!! abstract "Who this is for"
    You know Python and some PyTorch, and you have trained or used a computer-vision model. You do
    not need to know robotics or reinforcement learning. Where a chapter relies on a published
    result, it links the paper; where it relies on a number measured here, it says so in a green
    **Measured** box.

## The path

| # | Chapter | You will be able to… |
|---|---|---|
| 1 | [What a VLA is](01-what-is-a-vla.md) | explain what goes in, what comes out, and how pi0.5, SmolVLA and OpenVLA differ |
| 2 | [Actions and action chunks](02-actions-and-chunks.md) | read an action vector, and say why policies predict 30–50 steps at once |
| 3 | [How the action head works](03-action-heads.md) | follow flow matching from random noise to a trajectory, and see why seeds matter |
| 4 | [Normalisation and statistics](04-normalisation.md) | compute what the network actually sees, and spot a wrong-scale bug |
| 5 | [Anatomy of a LeRobot checkpoint](05-checkpoints.md) | open a checkpoint directory and know which file decides what |
| 6 | [Real-time control](06-real-time.md) | run a 300 ms policy on a 30 Hz robot without stalls or stale actions |
| 7 | [Evaluating a policy](07-evaluation.md) | design an evaluation that does not fool you |
| 8 | [Deploying to hardware](08-deployment.md) | choose GPU, edge device, export path and licence knowingly |

Read 1 → 3 for the model, 4 → 5 for the data plumbing, 6 → 8 for running it on a robot.

## Words used throughout

Observation
:   What the robot senses at one instant: camera images, the *state* (joint positions, gripper
    width), and the *task* sentence.

Action
:   What the robot is told to do at one control step, e.g. 7 target joint angles and a gripper
    width.

Chunk
:   A sequence of \(H\) future actions predicted from one observation. \(H\) is `chunk_size`.

Control rate
:   How many actions per second the robot executes (`fps`), typically 10–50 Hz.

Episode
:   One attempt at the task, from reset to success or timeout.

Closed loop
:   The policy acts, the world changes, the policy sees the result and acts again. The opposite,
    *open loop*, replays actions without looking.

Checkpoint
:   A directory holding the trained weights *and* everything needed to use them: configuration,
    tokenizer name, normalisation statistics.
