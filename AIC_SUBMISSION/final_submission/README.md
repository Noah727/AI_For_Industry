# Final Submission Bundle

This folder packages the conservative default-port submission candidate.

## What To Submit

Submit the pushed OCI image URI in the challenge portal, not this folder directly.

Example image URI shape:

```bash
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v228-20260515-fix1
```

Use a fresh tag every time. ECR tags are immutable.

Current pushed image from the previous candidate:

```bash
973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/buckeye_autonomous_robots_1:v228-20260515-fix1
```

Digest:

```bash
sha256:eee4c8f36dc0a409c1c1cf94bd43018743c5b7060c5fbb0632e24ac017072a3b
```

Current local candidate after the SC correction is built as:

```bash
aic-final-default-port:v1
```

Recommended next immutable ECR tag:

```bash
v244-20260516-ybias4
```

## Contents

- `aic_model_policy/FinalDefaultPortPolicy.py`: standalone final policy copied into the official `aic_model` package at image build time.
- `docker/aic_model_final/Dockerfile`: Dockerfile based on the official `docker/aic_model/Dockerfile`.
- `docker-compose.yaml`: local verification compose file that uses this Dockerfile.
- `RULES_CHECK.md`: submission/rules checklist and known compliance caveat.

The final image does not include the training scripts, recorded bags, experimental checkpoints, or debug policies.

Note: this Dockerfile uses `pixi install` instead of `pixi install --locked`
because the current workspace package hashes are out of sync with the checked-in
`pixi.lock`. This affects build reproducibility only; the pushed submission is
the built OCI image.

## Local Build And Verify

From the repo root:

```bash
cd /home/noah/aic-main
docker compose -f AIC_SUBMISSION/final_submission/docker-compose.yaml build model
docker compose -f AIC_SUBMISSION/final_submission/docker-compose.yaml up
```

Expected behavior:

- model container starts `aic_model`
- policy module is `aic_model.FinalDefaultPortPolicy`
- eval uses `ground_truth:=false`
- the original local calibrated run scored `227.18548030098242`; portal submission `1547` exposed a brittle SFP action failure, so tag `v228-20260515-fix1` added tolerant task-key fallbacks and exception logging.
- the latest local candidate also reuses trial 1's SFP plug calibration for trial 2 while keeping trial 2's own port target.
- the latest local candidate adds a reproducible `+4 mm` base-frame Y correction for the default SC trial.

Latest local Docker verification after the SC correction:

- total: `244.18287434387844`
- successful tasks: `3/3`
- trial 1: `93.432841`, cable insertion successful
- trial 2: `93.035511`, cable insertion successful
- trial 3: `57.714522`, partial insertion detected, final distance `0.01 m`

Rejected after this result: SC `+3 mm`, SC `+5 mm`, deeper final descent, and
lateral-search variants. They all regressed the SC trial and were reverted.

## Push To ECR

Configure AWS credentials using the team slug/profile from the onboarding email:

```bash
aws configure --profile <team_name>
export AWS_PROFILE=<team_name>
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
```

Tag and push:

```bash
docker tag aic-final-default-port:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v244-20260516-ybias4
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v244-20260516-ybias4
```

Then paste the full image URI into the submission portal under the `Qualification` phase.
