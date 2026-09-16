# AIC Submission Mental Path

## Direction

We will build a perception-assisted insertion policy instead of a fully end-to-end policy first.

The core idea:

1. Use simulator ground truth only during training/data collection.
2. Train a perception model to infer the target port pose or plug-to-port alignment from official observations.
3. Reuse the existing scripted compliant insertion behavior for approach, alignment, descent, and stabilization.
4. Submit only the perception model plus controller logic. No ground-truth TF access during evaluation.

This should train faster and be easier to debug than ACT/diffusion because the learned part is mostly perception, not the whole contact-rich manipulation behavior.

## Current Rethink: Keypoints Over Direct XYZ

The direct 3D pose regressors learned a coarse direction but were not accurate
enough for insertion. The better next target is image-space keypoints:

- predict target port pixel and current plug pixel
- use simulator TF only to auto-project labels during training
- use camera calibration and robot kinematics at runtime to turn pixels into
  a visual-servo or triangulated alignment correction

This gives the CNN a simpler visual task than absolute `base_link` XYZ
regression. It also matches the failure mode we observed: the robot was moving
in the right general direction, but the final port/plug alignment was off by a
small visual gap.

Implemented pieces:

- `TeacherRecorderPolicy` now saves camera intrinsics and camera poses for new
  recordings.
- `keypoint_dataset.py` can also retro-project labels for old samples by using
  the fixed wrist-camera geometry and saved `tcp_pose_base`.
- `TinyKeypointUNet` predicts two heatmaps: port and plug.
- `train_keypoint_smoke.py` trains the heatmap model.

Latest keypoint training result:

- dataset: `/tmp/aic_teacher_dataset`
- samples: `990`
- split: `660 train / 330 val`
- checkpoint: `AIC_SUBMISSION/runs/keypoint_center_100ep/best_model.pt`
- validation error: about `9.45 px` on `256x288` center-camera images

Next step: wire this into runtime inference and use it as a visual alignment
correction before/during the compliant insertion descent.

## Do We Need Manual Labels?

No, not initially.

We can auto-label perception data from the training simulator by launching with `ground_truth:=true`. `CheatCode` already reads the relevant TF frames:

- target port frame: `task_board/{task.target_module_name}/{task.port_name}_link`
- grasped plug frame: `{task.cable_name}/{task.plug_name}_link`
- TCP frame: `gripper/tcp`

For each frame in an episode, we can save:

- left, center, right camera images
- task fields: plug type, port type, port name, target module name
- robot state: TCP pose, velocity, joints, wrench
- ground-truth labels:
  - target port pose in `base_link`
  - plug pose in `base_link`
  - relative plug-to-port transform
  - teacher TCP target pose from the `CheatCode.calc_gripper_pose(...)` logic

There does not appear to be a bundled labeled perception dataset in this repo. The repo provides the simulation, ground-truth TF option, `CheatCode`, and LeRobot recording utilities. So our best dataset is one we generate automatically.

## Evaluation Environment First

The official quick-start path requires these tools on the host:

- Docker
- Distrobox
- Pixi pinned to `0.67.2`
- NVIDIA Container Toolkit if using GPU acceleration

Current status: Docker, Distrobox, NVIDIA GPU containers, and the WaveArm/CheatCode smoke tests work on the host. Codex itself may still not see GPU device nodes because it runs in a sandbox.

When those tools are available, the first environment smoke test is:

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

In another terminal:

```bash
cd /home/noah/aic-main
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm
```

For teacher data collection, run the eval environment with ground truth:

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

## Implementation Plan

### Phase 1: Submission Workspace

Create our code under `AIC_SUBMISSION/` and keep challenge source edits minimal.

Likely structure:

```text
AIC_SUBMISSION/
  MENTAL_PATH.md
  aic_submission/
    __init__.py
    TeacherRecorderPolicy.py
    PerceptionGuidedPolicy.py
    perception/
      model.py
      transforms.py
      inference.py
    data/
      record_teacher_dataset.py
      dataset.py
    training/
      train_port_pose.py
      config.yaml
  docker/
    Dockerfile
```

### Phase 2: Teacher Dataset Recorder

Build a recorder policy derived from `CheatCode`. First version is now:

- `AIC_SUBMISSION/aic_submission/TeacherRecorderPolicy.py`
- default dataset output: `/tmp/aic_teacher_dataset`
- default image downsample stride: `4`
- default recorder rate: every `5` teacher ticks
- output format: per-episode folder with `metadata.json`, `summary.json`, and `sample_*.npz`

For each control tick:

1. Call `get_observation()`.
2. Read ground-truth TFs for the target port, plug, and TCP.
3. Compute the same target pose `CheatCode` would command.
4. Save images, robot state, task fields, and labels.
5. Continue sending the teacher command so the episode succeeds.

Start with HDF5 or Zarr locally. Convert to PyTorch dataset directly. LeRobot is optional for this perception route.

Run it with the eval environment started using ground truth:

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

Then from the host:

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

After the run:

```bash
find /tmp/aic_teacher_dataset -maxdepth 2 -type f | head
```

Validate the dataset:

```bash
cd /home/noah/aic-main
pixi run python AIC_SUBMISSION/tools/inspect_teacher_dataset.py \
  --root /tmp/aic_teacher_dataset
```

Current smoke dataset status:

- 9 successful teacher-recorder episodes
- 990 total samples
- 6 `sfp->sfp` episodes
- 3 `sc->sc` episodes
- sample image shape: `256x288x3`
- stage counts: 180 approach, 774 insert, 36 stabilize
- validation passed

This is enough to prove the recorder and a tiny training pipeline work. It is still small for a robust final perception model, so the next real data target should be 50-120 randomized episodes.

Generate a randomized multi-trial teacher config:

```bash
cd /home/noah/aic-main
pixi run python AIC_SUBMISSION/tools/generate_teacher_config.py \
  --num-sfp 4 \
  --num-sc 2 \
  --seed 11 \
  --output AIC_SUBMISSION/generated_configs/smoke_randomized.yaml
```

Run a small randomized smoke collection:

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=/home/noah/aic-main/AIC_SUBMISSION/generated_configs/smoke_randomized.yaml
```

Once that passes, generate a larger collection config:

```bash
pixi run python AIC_SUBMISSION/tools/generate_teacher_config.py \
  --num-sfp 80 \
  --num-sc 40 \
  --seed 101 \
  --output AIC_SUBMISSION/generated_configs/teacher_120_trials.yaml
```

Then run the same recorder policy against that config.

### Phase 3: Perception Model

First model should be intentionally small:

- three image encoders, shared ResNet-18 or MobileNetV3
- proprioception MLP for TCP pose, joints, wrench
- task embedding for SFP vs SC and target port/module
- output head:
  - target port translation in `base_link`
  - target port orientation as 6D rotation representation
  - optional confidence/uncertainty

Loss:

- L1 or Huber for translation
- geodesic/6D rotation loss for orientation
- heavier weight near final insertion frames

Train separately for:

- SFP into NIC/SFP port
- SC plug into SC port

Then either share one model with task conditioning or keep two heads.

Current tiny training status:

- `AIC_SUBMISSION/aic_submission/perception/dataset.py` loads teacher samples by episode.
- `AIC_SUBMISSION/aic_submission/perception/model.py` defines a ResNet-18 image encoder plus state/task conditioning.
- `AIC_SUBMISSION/aic_submission/training/train_port_pose_smoke.py` trains a first port-XYZ regressor.
- `AIC_SUBMISSION/aic_submission/training/evaluate_port_pose_checkpoint.py` loads a saved checkpoint and reports XYZ error in millimeters.
- Smoke run on 128 samples passed on CPU:
  - 100 train samples, 28 validation samples
  - epoch 1 train Smooth L1: `0.082169`
  - epoch 1 validation Smooth L1: `0.081966`
  - checkpoint: `AIC_SUBMISSION/runs/port_pose_smoke_9ep/best_model.pt`
- First CUDA run in `AIC_SUBMISSION/runs/port_pose_all_9ep` used 512 samples because the training script defaulted to `--max-samples 512`.
  - best validation loss: `0.012251`
  - validation XYZ error: mean `35.05 mm`, median `31.19 mm`, p90 `59.93 mm`
  - this verifies the pipeline, but is not accurate enough for final insertion.
- 100-episode dataset collected at `/tmp/aic_teacher_dataset_100`:
  - 100 episodes, 11,000 samples
  - 70 `sfp->sfp` episodes, 30 `sc->sc` episodes
  - dataset size: about `766 MB`
- The first 100-episode run exposed a bad validation split because the generated trials were ordered by task. `split_by_episode(...)` now stratifies validation episodes by task.
- Corrected 100-episode run:
  - checkpoint: `AIC_SUBMISSION/runs/port_pose_100ep_stratified/best_model.pt`
  - train/val split: 8,140 train samples, 2,860 validation samples
  - best validation Smooth L1: `0.007138`
  - validation XYZ error: mean `23.33 mm`, median `20.77 mm`, p90 `41.08 mm`
  - task breakdown: `sfp->sfp` mean `23.57 mm`, `sc->sc` mean `22.77 mm`
  - stage breakdown: approach mean `27.68 mm`, insert mean `22.36 mm`, stabilize mean `22.32 mm`
  - task-mean baseline on the same split is about `57.63 mm` mean, so the model is learning useful signal, but not yet enough for final insertion.

Next perception step: improve the perception target/model before wiring it into control. The current single-center-camera absolute XYZ regressor is useful but too coarse; the next model should use either multi-camera inputs, a relative port-to-TCP target, a task-specific head, or a heatmap/crop-based target.

Implemented stronger perception variant:

- `AIC_SUBMISSION/aic_submission/perception/relative_dataset.py`
- `AIC_SUBMISSION/aic_submission/perception/relative_model.py`
- `AIC_SUBMISSION/aic_submission/training/train_relative_port_pose.py`
- `AIC_SUBMISSION/aic_submission/training/evaluate_relative_port_pose.py`

This variant uses:

- left, center, and right images
- relative target: `port_xyz_base - tcp_xyz_base`
- train-set label normalization stored in the checkpoint
- separate SFP and SC output heads

Verified 100-episode result:

- checkpoint: `AIC_SUBMISSION/runs/relative_port_pose_100ep/best_model.pt`
- best epoch: `29`
- validation Smooth L1: `0.033082`
- validation XYZ error: mean `5.05 mm`, median `3.17 mm`, p90 `11.17 mm`
- task breakdown: `sfp->sfp` mean `5.11 mm`, `sc->sc` mean `4.91 mm`
- stage breakdown: approach mean `12.68 mm`, insert mean `3.37 mm`, stabilize mean `3.12 mm`

This is now accurate enough to justify building the perception-guided hybrid policy. Approach frames are still weaker than final insertion frames, but the scripted controller only needs rough approach and precise final alignment.

Runtime policy wiring:

- `AIC_SUBMISSION/aic_submission/perception/inference.py` loads the relative model checkpoint and converts live observations into tensors.
- `AIC_SUBMISSION/aic_submission/PerceptionGuidedPolicy.py` is the first hybrid runtime policy.
- It runs with `ground_truth:=false`.
- It predicts `port_xyz_base = tcp_xyz_base_from_observation + predicted_relative_xyz`.
- It downsamples live images with stride `4`, matching the recorder images used during training.
- It keeps a light exponential moving average over predicted port positions to reduce frame-to-frame jitter.
- It uses dataset-estimated TCP-to-plug offsets by plug type:
  - SFP z offset: about `55.7 mm`
  - SC z offset: about `20.5 mm`
- It preserves the CheatCode-style sequence:
  - approach above the predicted port
  - descend slowly through the insertion range
  - hold/stabilize

Current limitation: orientation is not learned yet. The first runtime policy keeps the current TCP orientation and relies on the initial task setup being close. If insertion fails despite good XYZ, the next perception target should add port orientation or a teacher TCP target pose head.

First runtime smoke result with `ground_truth:=false`:

- 6 trials completed, but no full insertions.
- The cable moved generally toward the correct area, but final plug-port gaps were still about `3-7 cm`.
- One trial hit an off-limit contact penalty from gripper-to-task-board contact.
- This means offline validation accuracy is not yet the whole story; runtime closed-loop states differ from teacher states, or the simplified controller/pose geometry is not accurate enough.

Debug switches added to `PerceptionGuidedPolicy`:

- `AIC_DEBUG_GROUND_TRUTH=1`: logs learned-vs-ground-truth port error in millimeters when eval is launched with `ground_truth:=true`.
- `AIC_USE_GROUND_TRUTH_PORT=1`: forces the runtime policy to use ground-truth port position while keeping the same simplified controller. This isolates controller error from perception error.

Next diagnostic:

1. Use `AIC_SUBMISSION/generated_configs/debug_3_trials.yaml` for quick debug loops. It has 2 SFP trials and 1 SC trial.
2. Run learned policy with `ground_truth:=true` and `AIC_DEBUG_GROUND_TRUTH=1`.
3. Run ground-truth-port isolation with `AIC_USE_GROUND_TRUTH_PORT=1`.
4. If the ground-truth-port isolation fails, improve controller orientation/offset/insertion behavior.
5. If it succeeds, collect runtime-state data or train a policy head that predicts the controller target directly.

### Phase 4: Hybrid Policy

At runtime:

1. Read official observation only.
2. Predict target port pose.
3. Use known plug/gripper geometry and current TCP state to compute a desired TCP pose.
4. Execute the scripted sequence:
   - approach above port
   - align orientation
   - descend slowly
   - use wrench feedback/search for small errors
   - stabilize and return success

This keeps contact behavior deterministic and makes the learned model responsible only for finding the port accurately enough.

### Phase 5: Robustness

Generate data across the actual qualification variations:

- all five NIC rails, not just sample config NIC 0/1
- SFP port 0 and 1
- SC port 0 and 1
- board x/y/yaw variation
- rail translation variation
- small grasp offsets around the documented 2 mm / 0.04 rad range
- camera/lighting/image augmentations

Keep a validation set of held-out board and rail poses.

### Phase 6: Scoring Targets

The scoring rewards:

- successful correct insertion
- fast completion
- smooth and efficient paths
- low force
- no off-limit robot contacts

So the policy should prefer:

- direct smooth approach
- slow compliant final insertion
- force threshold guard
- short timeout and retry/search behavior instead of pushing hard

## Fallback Path

If perception-pose accuracy is not enough, use the generated dataset to train ACT as a second policy. The same teacher data can be converted into action chunks later, so the data collection work is not wasted.
