# Randomized Candidate Rules Check

Checked for the perception-guided randomized candidate.

## Packaging

- [x] Artifact is an OCI/Docker image.
- [x] Entrypoint launches `ros2 run aic_model aic_model`.
- [x] Policy is selected with Docker `CMD`: `policy:=aic_submission.PerceptionGuidedPolicy`.
- [x] `use_sim_time:=true` is passed.
- [x] No host-mounted code or checkpoints are required by the submitted image.
- [x] Model starts through the official `aic_model` lifecycle node.
- [x] Runtime communicates through the official AIC model interface.

## Runtime Inputs

The policy uses:

- `Task` metadata: plug type, port type, port name, target module name
- `Observation` camera images
- `Observation.controller_state.tcp_pose`
- official `move_robot` callback through `Policy.set_pose_target`

The policy does not use:

- ground-truth TF port lookup
- `/scoring` outputs
- Gazebo spawn/delete/reset services
- simulator state topics

## Known Performance

This candidate optimizes for randomized robustness rather than the public fixed
default eval.

Best comparable local randomized score:

```text
130.50678519523254
```

That result used:

- rich generic perception checkpoint
- SFP port 0 / SFP port 1 / SC specialist checkpoints
- target-specific bias map
- small final lateral search

## Current ECR Status

Local tag:

```bash
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/buckeye_autonomous_robots_1:v130-randomized-20260516
```

Push status:

```text
pushed successfully
```

Digest:

```text
sha256:ed3a7f38264462e8d99aa6d3412874dc2dcaa5642800de215add747597064836
```
