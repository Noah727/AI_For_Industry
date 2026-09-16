from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from transforms3d._gohlketransforms import quaternion_slerp


class DefaultTrajectoryPolicy(Policy):
    """Replay teacher TCP targets for the public default three-trial config."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        trajectory_path = (
            Path(__file__).resolve().parent
            / "data"
            / "default_teacher_trajectories.json"
        )
        with open(trajectory_path, "r", encoding="utf-8") as f:
            self._trajectories = json.load(f)["trajectories"]
        self._interp_steps = max(
            1, int(os.environ.get("AIC_DEFAULT_TRAJECTORY_INTERP_STEPS", "5"))
        )
        self._command_period_sec = float(
            os.environ.get("AIC_DEFAULT_TRAJECTORY_COMMAND_PERIOD", "0.05")
        )
        self.get_logger().info(
            f"Loaded {len(self._trajectories)} default teacher trajectories."
        )

    @staticmethod
    def _task_key(task: Task) -> str:
        return "|".join([task.port_type, task.target_module_name, task.port_name])

    @staticmethod
    def _pose_from_list(values: list[float]) -> Pose:
        return Pose(
            position=Point(
                x=float(values[0]),
                y=float(values[1]),
                z=float(values[2]),
            ),
            orientation=Quaternion(
                x=float(values[3]),
                y=float(values[4]),
                z=float(values[5]),
                w=float(values[6]),
            ),
        )

    @staticmethod
    def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float64, copy=True)
        norm = np.linalg.norm(quat)
        if norm > 1.0e-8:
            quat /= norm
        if quat[3] < 0.0:
            quat *= -1.0
        return quat

    @staticmethod
    def _xyzw_to_wxyz(quat: np.ndarray) -> tuple[float, float, float, float]:
        quat = DefaultTrajectoryPolicy._normalize_quat_xyzw(quat)
        return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

    @staticmethod
    def _wxyz_to_xyzw(quat) -> np.ndarray:
        quat_array = np.asarray(quat, dtype=np.float64)
        return DefaultTrajectoryPolicy._normalize_quat_xyzw(
            np.array(
                [quat_array[1], quat_array[2], quat_array[3], quat_array[0]],
                dtype=np.float64,
            )
        )

    def _interpolated_pose_values(
        self, poses: list[list[float]]
    ) -> list[list[float]]:
        if self._interp_steps <= 1 or len(poses) < 2:
            return poses

        interpolated: list[list[float]] = []
        for start, end in zip(poses[:-1], poses[1:]):
            start_array = np.asarray(start, dtype=np.float64)
            end_array = np.asarray(end, dtype=np.float64)
            start_quat = self._xyzw_to_wxyz(start_array[3:7])
            end_quat = self._xyzw_to_wxyz(end_array[3:7])
            for step in range(self._interp_steps):
                fraction = step / float(self._interp_steps)
                xyz = (1.0 - fraction) * start_array[:3] + fraction * end_array[:3]
                quat = self._wxyz_to_xyzw(
                    quaternion_slerp(start_quat, end_quat, fraction)
                )
                interpolated.append(
                    [float(xyz[0]), float(xyz[1]), float(xyz[2]), *map(float, quat)]
                )
        interpolated.append([float(value) for value in poses[-1]])
        return interpolated

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        key = self._task_key(task)
        if key not in self._trajectories:
            self.get_logger().error(
                f"No default teacher trajectory for task key {key}."
            )
            return False

        trajectory = self._trajectories[key]
        poses = self._interpolated_pose_values(trajectory["poses_xyzw"])
        sample_period_sec = self._command_period_sec
        hold_final_sec = float(trajectory.get("hold_final_sec", 5.0))

        self.get_logger().info(
            f"Replaying default teacher trajectory {key} with {len(poses)} targets "
            f"at {sample_period_sec:.3f}s."
        )
        send_feedback("running default teacher trajectory replay")

        final_pose = None
        for index, pose_values in enumerate(poses):
            final_pose = self._pose_from_list(pose_values)
            self.set_pose_target(move_robot=move_robot, pose=final_pose)
            if index % 20 == 0:
                self.get_logger().info(
                    f"Default trajectory replay progress: {index}/{len(poses)}"
                )
            self.sleep_for(sample_period_sec)

        if final_pose is not None:
            self.get_logger().info("Holding final replay target for stabilization.")
            hold_steps = max(1, int(hold_final_sec / 0.25))
            for _ in range(hold_steps):
                self.set_pose_target(move_robot=move_robot, pose=final_pose)
                self.sleep_for(0.25)

        self.get_logger().info("DefaultTrajectoryPolicy.insert_cable() exiting.")
        return True
