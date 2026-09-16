FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.67.2 bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY robostack_mapping.json /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install

# Copy the runtime perception package and the five checkpoints needed by the
# randomized-robust policy. Public checkpoints are stored with Git LFS.
COPY AIC_SUBMISSION/aic_submission /ws_aic/src/aic/aic_submission
COPY AIC_SUBMISSION/models/relative_port_pose_rich.pt /ws_aic/src/aic/models/relative_port_pose_rich.pt
COPY AIC_SUBMISSION/models/relative_port_pose_sfp_port0.pt /ws_aic/src/aic/models/relative_port_pose_sfp_port0.pt
COPY AIC_SUBMISSION/models/relative_port_pose_sfp_port1.pt /ws_aic/src/aic/models/relative_port_pose_sfp_port1.pt
COPY AIC_SUBMISSION/models/relative_port_pose_sc.pt /ws_aic/src/aic/models/relative_port_pose_sc.pt
COPY AIC_SUBMISSION/models/plug_aware_pose.pt /ws_aic/src/aic/models/plug_aware_pose.pt

ENV PYTHONPATH=/ws_aic/src/aic \
    AIC_PERCEPTION_CHECKPOINT=/ws_aic/src/aic/models/relative_port_pose_rich.pt \
    AIC_SFP_PORT0_PERCEPTION_CHECKPOINT=/ws_aic/src/aic/models/relative_port_pose_sfp_port0.pt \
    AIC_SFP_PORT1_PERCEPTION_CHECKPOINT=/ws_aic/src/aic/models/relative_port_pose_sfp_port1.pt \
    AIC_SC_PERCEPTION_CHECKPOINT=/ws_aic/src/aic/models/relative_port_pose_sc.pt \
    AIC_PLUG_AWARE_CHECKPOINT=/ws_aic/src/aic/models/plug_aware_pose.pt \
    AIC_PERCEPTION_DEVICE=auto \
    AIC_PERCEPTION_IMAGE_STRIDE=4 \
    AIC_PERCEPTION_USE_LEARNED_PLUG=1 \
    AIC_PERCEPTION_LEARNED_PLUG_MODE=xy \
    AIC_PERCEPTION_LEARNED_PLUG_AFTER_Z=0.03 \
    AIC_PERCEPTION_LEARNED_PLUG_GATE_M=0.025 \
    AIC_PERCEPTION_PLUG_FILTER_ALPHA=0.80 \
    AIC_PORT_FILTER_ALPHA=0.45 \
    AIC_PERCEPTION_FREEZE_PORT_AFTER=20 \
    AIC_PERCEPTION_ORIENTATION_SOURCE=fixed \
    AIC_PERCEPTION_USE_KEYPOINT=0 \
    AIC_PORT_BIAS_BY_TARGET=sfp_port_1:-0.024,-0.006,0.000;sfp_port_0:0.021,-0.004,0.000;sc_port_base:-0.030,0.000,0.000 \
    AIC_PORT_BIAS_AFTER_Z=0.20 \
    AIC_SFP_FINAL_Z_OFFSET=-0.045 \
    AIC_SC_FINAL_Z_OFFSET=-0.028 \
    AIC_LATERAL_SEARCH_HOLD_STEPS=5 \
    AIC_LATERAL_SEARCH_RETRACT_STEPS=5 \
    AIC_LATERAL_SEARCH_INSERT_STEP=0.003 \
    AIC_SFP_LATERAL_SEARCH_ENTRY_Z_OFFSET=0.045 \
    AIC_SC_LATERAL_SEARCH_ENTRY_Z_OFFSET=0.030 \
    AIC_LATERAL_SEARCH_KEEP_LAST_BIAS=0 \
    AIC_SFP_LATERAL_SEARCH_PATTERN=0.000,0.000,0.000;0.012,0.000,0.000;-0.012,0.000,0.000;0.000,0.012,0.000;0.000,-0.012,0.000;0.020,0.000,0.000;-0.020,0.000,0.000;0.012,0.012,0.000;0.012,-0.012,0.000;-0.012,0.012,0.000;-0.012,-0.012,0.000;0.000,0.000,0.000 \
    AIC_SC_LATERAL_SEARCH_PATTERN=0.000,0.000,0.000;0.010,0.000,0.000;-0.010,0.000,0.000;0.000,0.010,0.000;0.000,-0.010,0.000;0.016,0.000,0.000;-0.016,0.000,0.000;0.010,0.010,0.000;0.010,-0.010,0.000;-0.010,0.010,0.000;-0.010,-0.010,0.000;0.000,0.000,0.000 \
    AIC_DEBUG_LOG_EVERY=20

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_submission.PerceptionGuidedPolicy", "-p", "use_sim_time:=true"]
