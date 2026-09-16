# Randomized Submission Candidate

This folder records the best randomized-evaluation candidate. It uses the
perception-guided policy rather than the fixed public-default policy.

## Included Runtime

- `aic_submission.PerceptionGuidedPolicy`
- general rich relative-port-pose checkpoint
- SFP port 0 specialist
- SFP port 1 specialist
- SC specialist
- plug-aware checkpoint for late XY correction
- target-specific port bias correction
- small final lateral search
- keypoint correction disabled

The five required checkpoints are stored in `../models/` with Git LFS and are
copied into the self-contained OCI image during the build.

## Local Result

Best comparable randomized run:

- configuration: `random3_rich_seed501.yaml`
- completed tasks: 3/3
- total: `130.50678519523254`

The later keypoint experiment scored `126.17379268415645`, so it was not used
in this candidate. These are local measurements and not official leaderboard
scores.

## Run

From the repository root:

```bash
git lfs pull

docker compose \
  -f AIC_SUBMISSION/randomized_submission/docker-compose.yaml \
  build

docker compose \
  -f AIC_SUBMISSION/randomized_submission/docker-compose.yaml \
  up --abort-on-container-exit
```

`Dockerfile.snapshot` preserves the final image definition used for the
candidate. The actively referenced copy is
`../perception_submission/docker/aic_model_perception/Dockerfile`.
