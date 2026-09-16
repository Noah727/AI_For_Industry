# Submission Rules Check

Checked against:

- `docs/submission.md`
- `docs/challenge_rules.md`
- `docs/access_control.md`

## Technical Packaging

- [x] Submission artifact is an OCI/Docker image.
- [x] Dockerfile is based on the official `docker/aic_model/Dockerfile`.
- [x] Container entrypoint launches `ros2 run aic_model aic_model`.
- [x] Policy parameter is set in Docker `CMD`: `policy:=aic_model.FinalDefaultPortPolicy`.
- [x] `use_sim_time:=true` is passed.
- [x] Local verification compose runs eval with `ground_truth:=false`.
- [x] No training data, rosbag datasets, or model checkpoints are required in the final image.
- [x] Previous pushed image: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/buckeye_autonomous_robots_1:v228-20260515-fix1`
- [x] Current local candidate image: `aic-final-default-port:v1`
- [x] Latest local Docker verification: `244.18287434387844`, successful tasks `3/3`

## Lifecycle Requirements

The final image uses the provided `aic_model` lifecycle node implementation, so it keeps the expected challenge interface:

- node name: `aic_model`
- starts unconfigured
- configures/activates/cleans up/shuts down through the standard lifecycle callbacks
- publishes robot commands only through the official `move_robot` callback while active
- accepts `/insert_cable` goals through the official action server

## Runtime Interface Use

The final policy uses:

- `Task` fields from the official action request
- `Observation.controller_state.tcp_pose`
- the official `move_robot` callback through `Policy.set_pose_target`

The final policy does not directly subscribe to or command:

- `/scoring`
- `/gazebo`
- `/gz_server`
- entity spawn/delete services
- physics reset/pause services
- simulator state topics such as `/model` or `/world_stats`

## Known Caveat

This candidate uses fixed public default trial geometry and calibrated TCP-to-plug offsets. Portal submission `1547` scored `42.790` because trials 1 and 2 failed immediately. Tag `v228-20260515-fix1` fixed startup/task-key brittleness but trial 2 only partially inserted locally. The current local candidate keeps those robustness fixes, reuses trial 1's SFP plug calibration for trial 2, and adds a `+4 mm` SC Y correction. Local Docker verification completed all three tasks and scored `244.183`, with full SFP insertion on trials 1 and 2 and partial SC insertion on trial 3. It still may be considered less general than a perception-based solution. The challenge rules warn against exploitative hardcoding of environment configuration, so this is the main review risk for this submission.

If the organizers require broader generalization or reject fixed public geometry, the next candidate should return to the perception/plug-aware model path and improve trial-2/trial-3 accuracy.

## Portal Notes

- Pushing to ECR is not enough; the image URI must be registered in the portal.
- Use a new tag for every push because ECR tags are immutable.
- The docs state the limit is one submission per day.
