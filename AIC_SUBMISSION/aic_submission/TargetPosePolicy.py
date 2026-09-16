from __future__ import annotations

import os
import time
from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from aic_submission.perception.target_pose_inference import TargetTcpPoseInference


class TargetPosePolicy(Policy):
    """Behavior cloning policy that predicts the teacher's next TCP target pose."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        default_checkpoint = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "target_pose_100ep"
            / "best_model.pt"
        )
        checkpoint_path = Path(
            os.environ.get("AIC_TARGET_POSE_CHECKPOINT", str(default_checkpoint))
        )
        device = os.environ.get("AIC_TARGET_POSE_DEVICE", "auto")
        image_stride = int(os.environ.get("AIC_TARGET_POSE_IMAGE_STRIDE", "4"))
        self._predictor = TargetTcpPoseInference(
            checkpoint_path, device=device, image_stride=image_stride
        )
        self._specialist_predictors: dict[str, TargetTcpPoseInference] = {}
        for key, env_name in (
            ("sfp_port_0", "AIC_SFP_PORT0_TARGET_POSE_CHECKPOINT"),
            ("sfp_port_1", "AIC_SFP_PORT1_TARGET_POSE_CHECKPOINT"),
            ("sfp", "AIC_SFP_TARGET_POSE_CHECKPOINT"),
            ("sc", "AIC_SC_TARGET_POSE_CHECKPOINT"),
        ):
            specialist_path = os.environ.get(env_name, "").strip()
            if specialist_path:
                self._specialist_predictors[key] = TargetTcpPoseInference(
                    Path(specialist_path),
                    device=device,
                    image_stride=image_stride,
                )
        self.get_logger().info(
            "Loaded target-pose checkpoint "
            f"{self._predictor.checkpoint_path} on {self._predictor.device}; "
            f"epoch={self._predictor.checkpoint_epoch}, "
            f"val_loss={self._predictor.checkpoint_val_loss}"
        )
        for key, predictor in self._specialist_predictors.items():
            self.get_logger().info(
                "Loaded specialist target-pose checkpoint "
                f"{key}: {predictor.checkpoint_path}; "
                f"epoch={predictor.checkpoint_epoch}, "
                f"val_loss={predictor.checkpoint_val_loss}"
            )

    def _predictor_for_task(self, task: Task) -> TargetTcpPoseInference:
        if task.port_type == "sc":
            return self._specialist_predictors.get("sc", self._predictor)
        if task.port_type == "sfp":
            port_specific = self._specialist_predictors.get(task.port_name)
            if port_specific is not None:
                return port_specific
            return self._specialist_predictors.get("sfp", self._predictor)
        return self._predictor

    def _wait_for_observation(
        self, get_observation: GetObservationCallback, timeout_sec: float = 10.0
    ):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            obs = get_observation()
            if obs is not None and obs.left_image.height > 0:
                return obs
            self.get_logger().info("Waiting for observation...")
            time.sleep(0.1)
        return None

    def _set_predicted_target(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        task: Task,
    ) -> bool:
        obs = get_observation()
        if obs is None:
            self.get_logger().warn("No observation available for target-pose policy.")
            return False
        pose = self._predictor_for_task(task).predict_target_pose(obs, task)
        self.set_pose_target(move_robot=move_robot, pose=pose)
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"TargetPosePolicy.insert_cable() task: {task}")
        send_feedback("running target-pose imitation insertion")

        if self._wait_for_observation(get_observation) is None:
            self.get_logger().error("No observation received before timeout.")
            return False

        for _ in range(0, 100):
            self._set_predicted_target(get_observation, move_robot, task)
            self.sleep_for(0.05)

        for _ in range(0, 430):
            self._set_predicted_target(get_observation, move_robot, task)
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        for _ in range(0, 20):
            self._set_predicted_target(get_observation, move_robot, task)
            self.sleep_for(0.25)

        self.get_logger().info("TargetPosePolicy.insert_cable() exiting...")
        return True
