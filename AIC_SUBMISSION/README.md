# Perception-Guided Cable Insertion

This directory contains the policy, perception models, training tools, scene
generators, experiment configurations, and container definition developed for
the 2026 Intrinsic AI for Industry Challenge qualification task.

## Result

The best randomized local evaluation used three target-specific relative-pose
specialists, a plug-aware correction model, target-specific bias correction,
and a small lateral insertion search:

- randomized configuration: `random3_rich_seed501.yaml`
- completed tasks: 3/3
- total local score: `130.50678519523254`
- official leaderboard status: not submitted before the deadline

The score should be treated as one local evaluation, not a statistically
complete robustness claim.

## Approach

Teacher trajectories were recorded in randomized Gazebo scenes using the
ground-truth information permitted during training. Camera observations and
task metadata were converted into supervised examples for compact ResNet-based
pose regressors.

The final runtime policy:

1. selects an SFP-port-0, SFP-port-1, or SC specialist from the requested task;
2. predicts the target port pose relative to the robot;
3. optionally corrects the estimated plug position with a plug-aware model;
4. filters predictions over time and applies measured target-specific biases;
5. executes an approach trajectory followed by a small lateral search near the
   connector mouth.

The project also contains earlier general pose models, keypoint experiments,
target-pose regressors, analysis tools, and fixed-scene debugging policies.

## Layout

- `aic_submission/`: runtime policies, perception models, datasets, and
  training scripts
- `models/`: five checkpoints required by the randomized policy
- `generated_configs/`: teacher-data and randomized evaluation scenes
- `perception_submission/`: experiments and the self-contained model image
- `randomized_submission/`: the final randomized candidate and local compose
  entry point
- `tools/`: dataset generation and ROS bag analysis utilities
- `MENTAL_PATH.md`: development plan and technical reasoning
- `SUM.md`: experiment history and outcome summary

## Clone

Model weights are stored with Git LFS:

```bash
git clone git@github.com:Noah727/AI_For_Industry.git
cd AI_For_Industry
git lfs pull
```

## Build And Evaluate

The host needs Docker Engine, NVIDIA Container Toolkit, an NVIDIA GPU, and the
Docker Compose plugin. Build and run the self-contained randomized candidate:

```bash
docker compose \
  -f AIC_SUBMISSION/randomized_submission/docker-compose.yaml \
  build

docker compose \
  -f AIC_SUBMISSION/randomized_submission/docker-compose.yaml \
  up --abort-on-container-exit
```

The included compose file runs a headless three-trial randomized evaluation.
Generated results are written to a Docker volume rather than committed.

## Generate More Training Scenes

```bash
pixi run python AIC_SUBMISSION/tools/generate_teacher_config.py \
  --num-sfp 70 \
  --num-sc 30 \
  --seed 101 \
  --output AIC_SUBMISSION/generated_configs/teacher_100_trials.yaml
```

Training and evaluation commands for the individual model families are
documented in `commands.md` and implemented under
`aic_submission/training/`.

## Main Limitation

The strongest likely cause of the performance ceiling was dataset size and
coverage. Approximately 100 randomized teacher episodes were divided across
multiple plug and port types, leaving each specialist with relatively few
examples. The policy usually moved in the correct direction but retained
centimeter-scale pose errors, while reliable insertion requires much tighter
alignment.

The data also emphasized successful teacher trajectories rather than recovery
from near-contact failures. A stronger continuation would collect hundreds of
additional balanced episodes, concentrate new samples around the model's
failure cases, and use closed-loop visual or contact feedback during the final
centimeters of insertion.

## Attribution And License

The base toolkit comes from
[`intrinsic-dev/aic`](https://github.com/intrinsic-dev/aic). It is distributed
under Apache-2.0, with the upstream `aic_isaac` subtree under BSD-3-Clause.
The files added under `AIC_SUBMISSION/` are project modifications distributed
under Apache-2.0 unless a file states otherwise.

No challenge credentials, portal logs, ROS bags, or private evaluation
artifacts are included.
