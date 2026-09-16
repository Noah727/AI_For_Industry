# AIC Submission Commands

## Validate Teacher Dataset

```bash
cd /home/noah/aic-main
pixi run python AIC_SUBMISSION/tools/inspect_teacher_dataset.py \
  --root /tmp/aic_teacher_dataset
```

## Generate Randomized Teacher Config

```bash
cd /home/noah/aic-main
pixi run python AIC_SUBMISSION/tools/generate_teacher_config.py \
  --num-sfp 4 \
  --num-sc 2 \
  --seed 11 \
  --output AIC_SUBMISSION/generated_configs/smoke_randomized.yaml
```

## Run Eval Container With Config

Run inside `aic_eval`:

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/noah/aic-main/AIC_SUBMISSION/generated_configs/smoke_randomized.yaml
```

## Run Teacher Recorder

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_TEACHER_DATASET_DIR=/tmp/aic_teacher_dataset \
AIC_RECORDER_SAMPLE_EVERY=5 \
AIC_RECORDER_IMAGE_STRIDE=4 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.TeacherRecorderPolicy
```

## Tiny Port-Pose Training Smoke Test

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/train_port_pose_smoke.py \
  --root /tmp/aic_teacher_dataset \
  --output-dir AIC_SUBMISSION/runs/port_pose_smoke \
  --epochs 3 \
  --batch-size 16 \
  --max-samples 512 \
  --num-workers 0
```

On the host RTX 3090 this should automatically use CUDA if `torch.cuda.is_available()` is true.

Latest verified tiny run:

- dataset: `/tmp/aic_teacher_dataset`
- episodes: 9
- samples: 990
- tiny run: 128 samples, 1 epoch, CPU from Codex sandbox
- result: train Smooth L1 `0.082169`, validation Smooth L1 `0.081966`
- checkpoint: `AIC_SUBMISSION/runs/port_pose_smoke_9ep/best_model.pt`

## Train On All Current Samples

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/train_port_pose_smoke.py \
  --root /tmp/aic_teacher_dataset \
  --output-dir AIC_SUBMISSION/runs/port_pose_all_9ep \
  --epochs 20 \
  --batch-size 32 \
  --max-samples 0 \
  --num-workers 2
```

`--max-samples 0` means use every discovered sample.

## Train On 100-Episode Dataset

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/train_port_pose_smoke.py \
  --root /tmp/aic_teacher_dataset_100 \
  --output-dir AIC_SUBMISSION/runs/port_pose_100ep_stratified \
  --epochs 30 \
  --batch-size 64 \
  --max-samples 0 \
  --num-workers 4
```

Latest verified 100-episode run:

- dataset: `/tmp/aic_teacher_dataset_100`
- samples: 11,000
- split: 8,140 train / 2,860 validation
- device: CUDA
- best validation Smooth L1: `0.007138`
- checkpoint: `AIC_SUBMISSION/runs/port_pose_100ep_stratified/best_model.pt`

## Evaluate Port-Pose Checkpoint

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/evaluate_port_pose_checkpoint.py \
  --checkpoint AIC_SUBMISSION/runs/port_pose_100ep_stratified/best_model.pt \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --output AIC_SUBMISSION/runs/port_pose_100ep_stratified/eval_val.json
```

Latest verified 100-episode eval:

- mean XYZ error: `23.33 mm`
- median XYZ error: `20.77 mm`
- p90 XYZ error: `41.08 mm`
- `sfp->sfp` mean: `23.57 mm`
- `sc->sc` mean: `22.77 mm`
- approach-stage mean: `27.68 mm`

## Train Relative Multi-Camera Model

This is the stronger perception model:

- input: left, center, and right camera images
- target: normalized `port_xyz_base - tcp_xyz_base`
- heads: separate SFP and SC output heads

Run from the host:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/train_relative_port_pose.py \
  --root /tmp/aic_teacher_dataset_100 \
  --output-dir AIC_SUBMISSION/runs/relative_port_pose_100ep \
  --epochs 30 \
  --batch-size 32 \
  --max-samples 0 \
  --num-workers 4
```

Latest verified run:

- checkpoint: `AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt`
- best epoch: `29`
- validation Smooth L1: `0.033082`

Evaluate it:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/evaluate_relative_port_pose.py \
  --checkpoint AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt \
  --split val \
  --batch-size 32 \
  --num-workers 2 \
  --output AIC_SUBMISSION/runs/relative_port_pose_100ep/eval_val.json
```

Latest verified relative-model eval:

- mean XYZ error: `5.05 mm`
- median XYZ error: `3.17 mm`
- p90 XYZ error: `11.17 mm`
- `sfp->sfp` mean: `5.11 mm`
- `sc->sc` mean: `4.91 mm`
- approach-stage mean: `12.68 mm`
- insert-stage mean: `3.37 mm`
- stabilize-stage mean: `3.12 mm`

## Run Perception-Guided Policy

Start eval without ground-truth TF. Use the 3-trial debug config while iterating:

```bash
/entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/noah/aic-main/AIC_SUBMISSION/generated_configs/debug_3_trials.yaml
```

Then run from a second host terminal:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_PERCEPTION_CHECKPOINT="$PWD/AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt" \
AIC_PERCEPTION_DEVICE=auto \
AIC_PERCEPTION_IMAGE_STRIDE=4 \
AIC_PORT_FILTER_ALPHA=0.35 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.PerceptionGuidedPolicy
```

For policy debugging, prefer `debug_3_trials.yaml` because it covers both SFP and
SC while keeping turnaround time lower. Use the 6-trial `smoke_randomized.yaml`
only when checking repeatability.

## Debug Perception-Guided Policy

Use this to compare learned port estimates against simulator ground truth during
local debugging. Do not use ground truth for final evaluation.

Start eval with ground truth:

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/noah/aic-main/AIC_SUBMISSION/generated_configs/debug_3_trials.yaml
```

Then run the learned policy with debug logging:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_PERCEPTION_CHECKPOINT="$PWD/AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt" \
AIC_PERCEPTION_DEVICE=auto \
AIC_PERCEPTION_IMAGE_STRIDE=4 \
AIC_PORT_FILTER_ALPHA=0.35 \
AIC_DEBUG_GROUND_TRUTH=1 \
AIC_DEBUG_LOG_EVERY=20 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.PerceptionGuidedPolicy
```

The policy will print learned-vs-ground-truth port error in millimeters and a
summary at the end of each trial.

To isolate controller error from perception error, run the same policy but force
the port position to ground truth:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_PERCEPTION_CHECKPOINT="$PWD/AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt" \
AIC_PERCEPTION_DEVICE=auto \
AIC_PERCEPTION_IMAGE_STRIDE=4 \
AIC_PORT_FILTER_ALPHA=0.35 \
AIC_DEBUG_GROUND_TRUTH=1 \
AIC_USE_GROUND_TRUTH_PORT=1 \
AIC_DEBUG_LOG_EVERY=20 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.PerceptionGuidedPolicy
```

Interpretation:

- learned debug fails and ground-truth-port debug succeeds: perception/runtime distribution is the main issue.
- both fail: controller geometry, orientation, or insertion sequence is the main issue.
- both succeed: learned policy is probably sensitive to trial variation; collect more runtime-state data.

Latest controller isolation note:

- The old port-only controller scored `94.60` over the 3-trial debug eval even when the true port position was forced.
- The fix is to keep the learned port position model, but compute plug alignment in the TCP frame:
  - fixed per-plug TCP-to-plug translation and rotation from teacher data
  - fixed per-port nominal orientation and z height
  - CheatCode-style gripper orientation alignment
  - CheatCode-style XY insertion integrator

## Train Target-Pose Imitation Model

This is an alternate behavior-cloning branch that predicts the teacher's next
TCP target pose directly.

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/train_target_pose.py \
  --root /tmp/aic_teacher_dataset_100 \
  --output-dir AIC_SUBMISSION/runs/target_pose_100ep \
  --epochs 30 \
  --batch-size 64 \
  --max-samples 0 \
  --num-workers 4
```

Evaluate it:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
pixi run python AIC_SUBMISSION/aic_submission/training/evaluate_target_pose.py \
  --checkpoint AIC_SUBMISSION/runs/target_pose_100ep/best_model.pt \
  --root /tmp/aic_teacher_dataset_100 \
  --split val \
  --batch-size 64 \
  --num-workers 0 \
  --output AIC_SUBMISSION/runs/target_pose_100ep/eval_val.json
```

Latest verified target-pose eval:

- checkpoint: `AIC_SUBMISSION/runs/target_pose_100ep/best_model.pt`
- best epoch: `12`
- validation Smooth L1: `0.221924`
- mean target position error: `29.02 mm`
- median target position error: `12.74 mm`
- p90 target position error: `84.67 mm`
- mean orientation error: `2.85 deg`

Run it in simulation:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_TARGET_POSE_CHECKPOINT="$PWD/AIC_SUBMISSION/runs/target_pose_100ep/best_model.pt" \
AIC_TARGET_POSE_DEVICE=auto \
AIC_TARGET_POSE_IMAGE_STRIDE=4 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.TargetPosePolicy
```

## Generate A Larger Teacher Dataset

Generate 100 randomized trials:

```bash
cd /home/noah/aic-main
pixi run python AIC_SUBMISSION/tools/generate_teacher_config.py \
  --num-sfp 70 \
  --num-sc 30 \
  --seed 101 \
  --output AIC_SUBMISSION/generated_configs/teacher_100_trials.yaml
```

Run inside `aic_eval`:

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/noah/aic-main/AIC_SUBMISSION/generated_configs/teacher_100_trials.yaml
```

Then leave that terminal running. In a second host terminal, start the teacher recorder policy:

```bash
cd /home/noah/aic-main
PYTHONPATH="$PWD/AIC_SUBMISSION:$PYTHONPATH" \
AIC_TEACHER_DATASET_DIR=/tmp/aic_teacher_dataset_100 \
AIC_RECORDER_SAMPLE_EVERY=5 \
AIC_RECORDER_IMAGE_STRIDE=4 \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_submission.TeacherRecorderPolicy
```

This writes the 100-episode collection to `/tmp/aic_teacher_dataset_100`.

If the eval terminal appears to pause at:

```text
Sending InsertCable goal for task [task_1]
TrialState: TasksExecuting
Waiting for result...
```

that usually means the eval engine is waiting for the policy node in the second terminal. Check progress from a third host terminal:

```bash
find /tmp/aic_teacher_dataset_100 -mindepth 1 -maxdepth 1 -type d | wc -l
find /tmp/aic_teacher_dataset_100 -name 'sample_*.npz' | wc -l
du -sh /tmp/aic_teacher_dataset_100
```

If the sample count grows, the trial is running normally. If it stays at zero, the teacher recorder policy is not running or crashed in the second terminal.

## Train Keypoint Heatmap Perception

This is the new perception direction after the direct XYZ regressor plateaued.
It projects teacher TF labels into camera pixels and trains a tiny CNN to predict
port and plug heatmaps. No manual labels are needed.

Quick train on the current 100-episode teacher dataset:

```bash
cd /home/noah/aic-main
env PYTHONPATH=/home/noah/aic-main/AIC_SUBMISSION \
  /home/noah/.pixi/bin/pixi run python \
    AIC_SUBMISSION/aic_submission/training/train_keypoint_smoke.py \
    --root /tmp/aic_teacher_dataset \
    --output-dir AIC_SUBMISSION/runs/keypoint_center_100ep \
    --epochs 15 \
    --batch-size 32 \
    --max-samples 990 \
    --num-workers 2 \
    --heatmap-stride 4
```

Latest result:

- checkpoint: `AIC_SUBMISSION/runs/keypoint_center_100ep/best_model.pt`
- train/val samples: `660 / 330`
- device: CUDA
- output: center-camera port/plug heatmaps at `64x72`
- best validation pixel error: about `9.45 px` in the original `256x288` image

Next wiring step:

1. Add runtime keypoint inference that decodes heatmap argmax/soft-argmax.
2. Back-project center-camera pixel to a ray using `Observation.center_camera_info`.
3. Either combine multiple cameras for triangulation, or use current TCP/camera
   geometry and the learned plug/port pixel alignment for visual servoing.
4. Hand the refined pose/alignment to the existing compliant insertion loop.

Run the keypoint-servo policy:

```bash
cd /home/noah/aic-main
env PYTHONPATH=/home/noah/aic-main/AIC_SUBMISSION \
  AIC_PLUG_AWARE_CHECKPOINT=/home/noah/aic-main/AIC_SUBMISSION/runs/plug_aware_pose_100ep/best_model.pt \
  AIC_PLUG_AWARE_FREEZE_PORT_AFTER=1 \
  AIC_PLUG_AWARE_DEBUG_LOG_EVERY=60 \
  AIC_KEYPOINT_CHECKPOINT=/home/noah/aic-main/AIC_SUBMISSION/runs/keypoint_center_100ep/best_model.pt \
  AIC_KEYPOINT_DEVICE=auto \
  AIC_KEYPOINT_IMAGE_STRIDE=4 \
  AIC_KEYPOINT_CORRECTION_GAIN=0.30 \
  AIC_KEYPOINT_MAX_CORRECTION_M=0.012 \
  AIC_KEYPOINT_MIN_CONFIDENCE=0.45 \
  AIC_KEYPOINT_FILTER_ALPHA=0.30 \
  AIC_KEYPOINT_UPDATE_MIN_Z_OFFSET=0.050 \
  AIC_KEYPOINT_MAX_DELTA_PX=300 \
  AIC_KEYPOINT_MAX_RAW_CORRECTION_M=0.025 \
  /home/noah/.pixi/bin/pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true -p policy:=aic_submission.KeypointServoPolicy
```

Run a local Docker 3-random-trial robustness eval with the keypoint policy:

```bash
cd /home/noah/aic-main
docker compose \
  -f AIC_SUBMISSION/final_submission/docker-compose.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.keypoint-local.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.random3.yaml \
  up --abort-on-container-exit --exit-code-from eval
```

For a faster one-trial shakeout:

```bash
cd /home/noah/aic-main
docker compose \
  -f AIC_SUBMISSION/final_submission/docker-compose.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.keypoint-local.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.debug1.yaml \
  up --abort-on-container-exit --exit-code-from eval
```

Clean up afterward:

```bash
docker compose \
  -f AIC_SUBMISSION/final_submission/docker-compose.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.keypoint-local.yaml \
  -f AIC_SUBMISSION/final_submission/docker-compose.random3.yaml \
  down --remove-orphans
```

For the one-trial shakeout, swap `docker-compose.random3.yaml` for
`docker-compose.debug1.yaml` in the cleanup command.
