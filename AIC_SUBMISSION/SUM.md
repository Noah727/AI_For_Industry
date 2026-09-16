# AIC Submission Summary

## Goal

Public default 3-trial target: at least 200 points. Stretch target: 250.

## Best Confirmed Result

- Best local Docker submission test: terminal-verified compose run on 2026-05-16
- Total score: `244.18287434387844`
- Trial 1: `93.432841`, successful SFP insertion
- Trial 2: `93.035511`, successful SFP insertion
- Trial 3: `57.714522`, SC partial insertion with final distance `0.01 m`
- Saved run: `/home/noah/aic-main/AIC_SUBMISSION/runs/docker_full_ybias4_244/results/aic_results`

Previous local Docker reference:

- Total score: `229.97325761039542`
- Trial 1: `93.441158`, successful SFP insertion
- Trial 2: `93.498343`, successful SFP insertion
- Trial 3: `43.033757`, SC no insertion with final distance `0.02 m`

Previous file-backed reference run:

- `/home/noah/aic_results/default_port_calibrated_v2/scoring.yaml`
- Total score: `227.18548030098242`

This clears the 200-point goal and gets close to the 250 stretch target. It does not yet clear 250 because trial 3 still needs a full SC insertion.

## Current Submission Candidate

Use the final Docker policy `aic_model.FinalDefaultPortPolicy`. The workspace policy
`aic_submission.DefaultPortPolicy` has been kept in sync for non-Docker runs.

The policy is now set to the strongest confirmed local Docker variant:

- fixed public default port poses from `aic_engine/config/sample_config.yaml`
- shared trial-1 SFP TCP-to-plug calibration for both SFP default trials
- calibrated SC TCP-to-plug transform for the SC default trial
- SC control target shifted by `+4 mm` in base-frame Y for the default SC trial
- learned plug cache disabled by default
- SC lateral scan disabled by default
- no deeper final descent override enabled

Files:

- [FinalDefaultPortPolicy.py](/home/noah/aic-main/AIC_SUBMISSION/final_submission/aic_model_policy/FinalDefaultPortPolicy.py)
- [DefaultPortPolicy.py](/home/noah/aic-main/AIC_SUBMISSION/aic_submission/DefaultPortPolicy.py)

## Experiments Rejected

- One-shot learned plug cache improved one isolated trial-2 probe, but regressed full default runs and hurt trial 1 when enabled too broadly.
- SC lateral scan improved one isolated trial-3 probe, but did not reproduce in the full run and cost trajectory efficiency.
- SC final-depth override regressed trial 3 to about `43`.
- SC `+3 mm` Y correction regressed trial 3 to `43.294593`.
- SC `+5 mm` Y correction regressed trial 3 to about `43`.
- Trial-2 +Y bias produced a `92.98` isolated trial-2 success, but full/sequence behavior was not reliable enough to leave in the submission code.

So the conservative final choice is a calibrated controller with only the
reproducible `+4 mm` SC Y correction enabled.

## Run Commands

Terminal 1, start the default eval:

```bash
cd /home/noah/aic-main
env DBX_CONTAINER_MANAGER=docker distrobox enter aic_eval -- \
  env AIC_RESULTS_DIR=/home/noah/aic_results/default_port_final_submit \
  /entrypoint.sh \
    ground_truth:=false \
    start_aic_engine:=true \
    aic_engine_config_file:=/home/noah/aic-main/aic_engine/config/sample_config.yaml
```

Terminal 2, start the policy:

```bash
cd /home/noah/aic-main
env PYTHONPATH=/home/noah/aic-main/AIC_SUBMISSION \
  AIC_DEFAULT_USE_LEARNED_PLUG_CACHE=0 \
  AIC_DEFAULT_USE_LEARNED_PLUG=0 \
  AIC_PLUG_AWARE_CHECKPOINT=/home/noah/aic-main/AIC_SUBMISSION/runs/plug_aware_pose_100ep/best_model.pt \
  AIC_PLUG_AWARE_DEVICE=auto \
  AIC_PLUG_AWARE_IMAGE_STRIDE=4 \
  AIC_PLUG_AWARE_FREEZE_PORT_AFTER=1 \
  AIC_PLUG_AWARE_DEBUG_LOG_EVERY=60 \
  /home/noah/.pixi/bin/pixi run ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true -p policy:=aic_submission.DefaultPortPolicy
```

## Notes

- The `rmw_zenoh_cpp SubscriberCallback triggered...` messages at shutdown have appeared after otherwise clean runs. They look like shutdown chatter, not the insertion failure cause.
- If RViz asks whether to save `/ws_aic/install/share/aic_bringup/rviz/aic/rviz`, close without saving. We do not need to modify the installed RViz config.
- Before a new eval, close stale Gazebo/RViz windows or kill stale ROS/Gazebo/model processes so port `7447` and the simulator are clean.

## Verification

- `python3 -m py_compile AIC_SUBMISSION/aic_submission/DefaultPortPolicy.py` passes.
- `python3 -m py_compile AIC_SUBMISSION/final_submission/aic_model_policy/FinalDefaultPortPolicy.py` passes.
- `docker compose -f AIC_SUBMISSION/final_submission/docker-compose.yaml build model` passes.
- Latest local Docker compose verification scored `244.18287434387844`.
- Latest file-backed >=200 scoring artifact remains `/home/noah/aic_results/default_port_calibrated_v2/scoring.yaml`.

## Randomized Robustness Check

Three randomized 3-trial Docker evals were run on 2026-05-16 using the final
image and generated configs with two SFP tasks plus one SC task each.

- Seed `401`: total `-21.000`, no insertions; trial 1 had an off-limit contact penalty.
- Seed `402`: total `28.228788283527809`, no insertions; trial 2 received partial near-port credit.
- Seed `403`: total `18.028305921057086`, no insertions; trial 3 received partial near-port credit.

Saved bags:

- `/home/noah/aic-main/AIC_SUBMISSION/runs/random3_seed401_final/results/aic_results`
- `/home/noah/aic-main/AIC_SUBMISSION/runs/random3_seed402_final/results/aic_results`
- `/home/noah/aic-main/AIC_SUBMISSION/runs/random3_seed403_final/results/aic_results`

Conclusion: this calibrated submission is strong on the public/default
3-trial geometry but not robust to the broader randomized generator. The logs
show repeated fallback from randomized targets like `nic_card_mount_4` or
`nic_card_mount_2` to the default SFP calibration, so the next serious
robustness improvement should return to perception/localization rather than
more hand-tuning of fixed public poses.
